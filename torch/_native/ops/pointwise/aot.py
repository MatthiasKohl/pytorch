"""AOT instantiations for the initial floating pointwise TMA operations.

The structured hook runs after ATen meta, so its output layout is authoritative.
Unsupported calls return false and continue through ATen's original CUDA kernel.
This module must remain torch-free at import time for stage-1 code generation.
"""

from __future__ import annotations

import itertools
from typing import Any


_DTYPES = {
    "float16": "at::kHalf",
    "bfloat16": "at::kBFloat16",
    "float32": "at::kFloat",
    "float64": "at::kDouble",
}
_OPS = (
    ("add", "_add", 2),
    ("sub", "_sub", 2),
    ("mul", "_mul", 2),
    ("div.Tensor", "_div", 2),
    ("neg", "_neg", 1),
    ("gt.Tensor", "_gt", 2),
)


def _axes(inputs: tuple[Any, ...], out: Any, boolean: bool) -> dict[str, Any]:
    import torch

    ref = inputs[0]
    if (
        not all(isinstance(t, torch.Tensor) for t in inputs)
        or not ref.is_cuda
        or ref.dim() != 2
        or ref.dtype
        not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        or all(t.is_contiguous() for t in inputs)
        or torch.cuda.get_device_capability(ref.device) != (10, 7)
    ):
        return {"eligible": False}
    box = 16 if ref.dtype == torch.float64 else 32
    fast = 0 if ref.stride(0) == 1 else 1
    if out is not None:
        if (
            out.layout != torch.strided
            or out.device != ref.device
            or out.shape != ref.shape
            or out.dtype != (torch.bool if boolean else ref.dtype)
            or out.is_neg()
            or out.is_conj()
            or out.storage_offset()
            or out.const_data_ptr() % 16
            or any(torch._C._overlaps(out, t) for t in inputs)
        ):
            return {"eligible": False}
        fast = 0 if out.stride(0) == 1 else 1
        if out.stride(fast) != 1 or out.stride(1 - fast) != out.size(fast):
            return {"eligible": False}
    rows, cols = ref.size(1 - fast), ref.size(fast)
    if (
        rows <= 0
        or cols <= 0
        or rows % box
        or cols % box
        or rows * cols > 2**31 - 1
        or rows // box > 65535
        or len(inputs) * (6 * box * box * ref.element_size() + 1024) + 48
        > torch.cuda.get_device_properties(ref.device).shared_memory_per_block_optin
    ):
        return {"eligible": False}
    mask = []
    for t in inputs:
        if (
            t.layout != torch.strided
            or t.device != ref.device
            or t.dtype != ref.dtype
            or t.shape != ref.shape
            or t.is_neg()
            or t.is_conj()
            or t.storage_offset()
            or sum((n - 1) * s for n, s in zip(t.shape, t.stride())) > 2**31 - 1
        ):
            return {"eligible": False}
        transposed = t.stride(fast) != 1
        inner = 1 - fast if transposed else fast
        pitch = t.stride(1 - inner)
        if (
            t.stride(inner) != 1
            or pitch < t.size(inner)
            or pitch * t.element_size() % 16
            or t.const_data_ptr() % (128 if transposed else 16)
        ):
            return {"eligible": False}
        mask.append(transposed)
    return {"eligible": True, "dtype": ref.dtype, "transposed": tuple(mask)}


class _Declaration:
    DISPATCH_KEY = "CUDA"
    KERNEL_MODULE = "aot_kernels"
    ARCHS = ("sm_107",)

    def __init__(self, aten: str, fn: str, nin: int) -> None:
        self.ATEN_OP = aten
        self.fn = fn
        self.nin = nin
        self.boolean = fn == "_gt"
        self.alpha = fn in ("_add", "_sub")

    def kernel_precompile_grid(self) -> list[dict[str, Any]]:
        return [
            {
                "dtype": list(_DTYPES),
                "transposed": mask,
                "fn": self.fn,
                "boolean": self.boolean,
                "eligible": True,
            }
            for mask in itertools.product((False, True), repeat=self.nin)
        ]

    def covered_axes(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        alpha = kwargs.get("alpha", args[self.nin] if len(args) > self.nin else 1)
        if self.alpha and isinstance(alpha, complex):
            return {"eligible": False}
        return _axes(args[: self.nin], kwargs.get("out"), self.boolean)

    def _inputs(self) -> str:
        return "{self}" if self.nin == 1 else "{self, other}"

    def cpp_helpers(self) -> str:
        return _CPP_HELPERS

    def cpp_dispatch_prelude(self) -> str:
        alpha_gate = "if (alpha.isComplex()) return false;" if self.alpha else ""
        return f"""
{alpha_gate}
auto cfg = pointwise_tma_config({self._inputs()}, &out, {str(self.boolean).lower()},
                                _naot_props->sharedMemPerBlockOptin);
if (cfg.mask < 0) return false;
"""

    def cpp_dispatch(self, spec: dict[str, Any]) -> str:
        mask = sum(int(value) << i for i, value in enumerate(spec["transposed"]))
        return f"self.scalar_type() == {_DTYPES[spec['dtype']]} && cfg.mask == {mask}"

    def cpp_launch(self, spec: dict[str, Any], launch_fn: str) -> str:
        views = []
        for i, transposed in enumerate(spec["transposed"]):
            name = "self" if i == 0 else "other"
            shape = "{cfg.rows, cfg.cols}"
            stride = (
                f"{{1, {name}.stride(cfg.fast)}}"
                if transposed
                else f"{{{name}.stride(1 - cfg.fast), 1}}"
            )
            views.append(f"auto x{i} = {name}.as_strided({shape}, {stride});")
        output = (
            "auto result = out.as_strided({1, cfg.rows, cfg.cols}, {0, cfg.cols, 1});"
        )
        alpha = "alpha.toDouble()" if self.alpha else "1.0"
        operands = "x0" if self.nin == 1 else "x0, x1"
        return "\n".join(
            [
                *views,
                output,
                f"{launch_fn}({operands}, result, {alpha}, at::cuda::getCurrentCUDAStream());",
            ]
        )

    def cpp_covers(self) -> str:
        alpha_gate = "if (alpha.isComplex()) return false;" if self.alpha else ""
        return f"""
{alpha_gate}
return pointwise_tma_config({self._inputs()}, out ? &*out : nullptr,
                            {str(self.boolean).lower()}, _naot_props->sharedMemPerBlockOptin).mask >= 0;
"""


def declarations() -> list[_Declaration]:
    return [_Declaration(*op) for op in _OPS]


_CPP_HELPERS = r"""
struct PointwiseTmaConfig {
  int mask = -1;
  int fast = 1;
  int64_t rows = 0, cols = 0;
};

PointwiseTmaConfig pointwise_tma_config(
    std::initializer_list<at::Tensor> inputs, const at::Tensor* out,
    bool boolean, size_t smem_limit) {
  PointwiseTmaConfig cfg;
  const auto& ref = *inputs.begin();
  if (!ref.is_cuda() || ref.layout() != at::kStrided || ref.dim() != 2) return cfg;
  auto dt = ref.scalar_type();
  if (dt != at::kHalf && dt != at::kBFloat16 && dt != at::kFloat && dt != at::kDouble) return cfg;
  if (std::all_of(inputs.begin(), inputs.end(), [](const auto& t) { return t.is_contiguous(); })) return cfg;
  cfg.fast = ref.stride(0) == 1 ? 0 : 1;
  if (out) {
    if (out->layout() != at::kStrided || out->device() != ref.device() || out->sizes() != ref.sizes() ||
        out->scalar_type() != (boolean ? at::kBool : dt) || out->storage_offset() ||
        out->is_neg() || out->is_conj() || reinterpret_cast<uintptr_t>(out->const_data_ptr()) % 16) return cfg;
    cfg.fast = out->stride(0) == 1 ? 0 : 1;
    if (out->stride(cfg.fast) != 1 || out->stride(1 - cfg.fast) != out->size(cfg.fast)) return cfg;
    for (const auto& t : inputs) if (out->is_alias_of(t)) return cfg;
  }
  cfg.rows = ref.size(1 - cfg.fast);
  cfg.cols = ref.size(cfg.fast);
  int box = dt == at::kDouble ? 16 : 32;
  if (cfg.rows <= 0 || cfg.cols <= 0 || cfg.rows % box || cfg.cols % box ||
      ref.numel() > INT32_MAX || cfg.rows / box > 65535 ||
      inputs.size() * (6 * box * box * ref.element_size() + 1024) + 48 > smem_limit) return cfg;
  int mask = 0, i = 0;
  for (const auto& t : inputs) {
    if (!t.is_cuda() || t.layout() != at::kStrided || t.device() != ref.device() ||
        t.scalar_type() != dt || t.sizes() != ref.sizes() || t.storage_offset() ||
        t.is_neg() || t.is_conj()) return cfg;
    bool transposed = t.stride(cfg.fast) != 1;
    int inner = transposed ? 1 - cfg.fast : cfg.fast;
    int64_t pitch = t.stride(1 - inner);
    if (t.stride(inner) != 1 || pitch < t.size(inner) || pitch > INT32_MAX ||
        pitch * t.element_size() % 16 ||
        reinterpret_cast<uintptr_t>(t.const_data_ptr()) % (transposed ? 128 : 16) ||
        (t.size(0) - 1) * t.stride(0) + (t.size(1) - 1) * t.stride(1) > INT32_MAX) return cfg;
    mask |= int(transposed) << i++;
  }
  cfg.mask = mask;
  return cfg;
}
"""

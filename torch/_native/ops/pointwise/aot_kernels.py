"""Flat, dynamic-shape C ABI for the shared pointwise TMA executor."""

from typing import Any

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute

from . import kernel, ops


class _AotTma:
    def __init__(
        self, fn: str, dtype: Any, mask: tuple[bool, ...], boolean: bool
    ) -> None:
        self.compute = cutlass.Float64 if dtype is cutlass.Float64 else cutlass.Float32
        box, block = (16, 32) if dtype is cutlass.Float64 else (32, 128)
        config = kernel._TmaConfig(mask, box, block, (), ((), ()))
        self.op = kernel._ElementwiseTma2D(
            ops.get_fn(fn),
            len(mask),
            int(fn in ("_add", "_sub")),
            self.compute,
            cutlass.Int8 if boolean else dtype,
            dtype,
            config,
        )

    @cute.jit
    def binary(
        self,
        x: cute.Tensor,
        y: cute.Tensor,
        out: cute.Tensor,
        alpha: cutlass.Float64,
        stream: cuda.CUstream,
    ) -> None:
        self.op(
            [x, y], out, self.compute(alpha), self.compute(1), self.compute(1), stream
        )

    @cute.jit
    def unary(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        alpha: cutlass.Float64,
        stream: cuda.CUstream,
    ) -> None:
        self.op([x], out, self.compute(alpha), self.compute(1), self.compute(1), stream)


def build(spec: dict[str, Any]) -> dict[str, Any]:
    dtype = getattr(
        cutlass,
        {
            "float16": "Float16",
            "bfloat16": "BFloat16",
            "float32": "Float32",
            "float64": "Float64",
        }[spec["dtype"]],
    )
    mask = tuple(spec["transposed"])
    op = _AotTma(spec["fn"], dtype, mask, spec["boolean"])
    box = op.op.box
    shapes = (cute.sym_int(divisibility=box), cute.sym_int(divisibility=box))
    fakes = []
    tensor_args = []
    names = ("x",) if len(mask) == 1 else ("x", "y")
    for name, transposed in zip(names, mask):
        pitch = cute.sym_int64(divisibility=16 // (dtype.width // 8))
        stride = (1, pitch) if transposed else (pitch, 1)
        fakes.append(
            cute.runtime.make_fake_tensor(
                dtype,
                shapes,
                stride=stride,
                assumed_align=128 if transposed else 16,
            )
        )
        tensor_args.append(
            {
                "name": name,
                "dynamic_sizes": [0, 1],
                "dynamic_strides": [1 if transposed else 0],
                "read_only": True,
            }
        )
    output_type = cutlass.Int8 if spec["boolean"] else dtype
    fakes.append(
        cute.runtime.make_fake_tensor(
            output_type,
            (1, *shapes),
            stride=(0, cute.sym_int64(divisibility=box), 1),
            assumed_align=16,
        )
    )
    tensor_args.append({"name": "out", "dynamic_sizes": [1, 2], "dynamic_strides": [1]})
    tag = "".join("t" if value else "n" for value in mask)
    return {
        "kind": "cutedsl",
        "fn": op.unary if len(mask) == 1 else op.binary,
        "prefix": f"pointwise_{spec['fn'].lstrip('_')}_{spec['dtype']}_{tag}",
        "fake_args": [*fakes, cutlass.Float64(1), cute.runtime.make_fake_stream()],
        "tensor_args": tensor_args,
        "scalar_args": [{"name": "alpha", "ctype": "double"}],
    }

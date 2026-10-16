# Register only TMA-eligible pointwise calls; other layouts keep their ATen kernels.

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND as PromotionKind

from ... import cutedsl_utils as cu
from ...utils import capability as cap
from ...utils.lazy import LazyModule
from .table import (
    POINTWISE_DEF_TABLE,
    POINTWISE_VARIANTS,
    PointwiseDef,
    PointwiseVariant,
    variant_aten_name,
)


if TYPE_CHECKING:
    from ...cutedsl import dtypes as dt
    from . import kernel as K, ops
else:
    dt = LazyModule("torch._native.cutedsl.dtypes")
    K = LazyModule("torch._native.ops.pointwise.kernel")
    ops = LazyModule("torch._native.ops.pointwise.ops")


_SUPPORTED = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _scalars(
    row: PointwiseDef,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    compute: torch.dtype,
) -> tuple[Any, ...]:
    # Resolve the row's named scalar args, which the caller may pass either
    # positionally (after the nin tensors) or by keyword (e.g. torch.add(x, y,
    # alpha=2)). Missing -> the row's aten default (row.scalar_defaults, aligned with
    # row.scalars; unset -> 1, the add/sub-alpha convention). Tuple in row.scalars order.
    #
    # A name in row.optional_defaults is an aten `Scalar?`/`float?` the caller may omit
    # INDEPENDENTLY, and its fill-in is a callable of the RESULT dtype rather than a fixed
    # value (an omitted nan_to_num posinf is that dtype's max, an omitted clamp bound its
    # infinity). Resolve these values before admission as well as before launch.
    pos = args[row.nin :]
    opt = row.optional_defaults or {}
    out = []
    for i, name in enumerate(row.scalars):
        if name in kwargs and kwargs[name] is not None:
            out.append(kwargs[name])
            continue
        if i < len(pos) and pos[i] is not None:
            out.append(pos[i])
            continue
        if name in opt:
            out.append(opt[name](compute))
            continue
        out.append(row.scalar_defaults[i] if row.scalar_defaults else 1)
    return tuple(out)


def _mode_of(row: PointwiseDef, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    # The row's mode-kwarg VALUE for this call (gelu's approximate, div's
    # rounding_mode). aten allows it positionally too, after the tensors and any
    # numeric scalars, so fall back to scanning the positional tail for a str/None.
    if row.mode_kwarg in kwargs:
        return kwargs[row.mode_kwarg]
    tail = args[row.nin + len(row.scalars) :]
    return tail[0] if tail else None


def _function(
    row: PointwiseDef, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[str | None, PromotionKind]:
    if row.mode_kwarg is None:
        return row.fn, row.promotion
    mode = _mode_of(row, args, kwargs)
    return (row.mode_fns or {}).get(mode), row.promotion


def _make_cond(row: PointwiseDef, variant: PointwiseVariant) -> Any:
    dtypes = _SUPPORTED if row.dtypes is None else row.dtypes

    def cond(*args: Any, **kwargs: Any) -> bool:
        ins = args[: row.nin]
        if len(ins) != row.nin or any(
            not isinstance(t, torch.Tensor) or t.dtype not in dtypes or cap.is_traced(t)
            for t in ins
        ):
            return False
        if all(t.is_contiguous() for t in ins):
            return False
        ref = ins[0]
        if not cap.device_ok(ref, (10,)) or not cap.on_current_device(ref):
            return False
        fn_name, promotion = _function(row, args, kwargs)
        scalars = _scalars(row, args, kwargs, ref.dtype)
        if fn_name is None or any(
            not isinstance(s, (bool, int, float)) for s in scalars
        ):
            return False
        if row.aten == "softshrink" and not (scalars[0] >= 0):
            return False
        result_dtype = (
            torch.bool if promotion is PromotionKind.ALWAYS_BOOL else ref.dtype
        )
        target = kwargs.get("out") if variant.out_from == "out_kw" else None
        if variant.out_from == "out_kw" and (
            not isinstance(target, torch.Tensor)
            or cap.is_traced(target)
            or not torch.can_cast(result_dtype, target.dtype)
        ):
            return False
        if row.precondition is not None and not row.precondition(ins, target):
            return False
        out_dtype = result_dtype if target is None else target.dtype
        return K.can_run(ins, tuple(ref.shape), out_dtype, target)

    return cond


def _make_impl(row: PointwiseDef, variant: PointwiseVariant) -> Any:
    def impl(*args: Any, **kwargs: Any) -> torch.Tensor:
        ins = args[: row.nin]
        ref = ins[0]
        fn_name, promotion = _function(row, args, kwargs)
        if fn_name is None:
            raise AssertionError(
                "pointwise TMA implementation requires a supported mode"
            )
        result_dtype = (
            torch.bool if promotion is PromotionKind.ALWAYS_BOOL else ref.dtype
        )
        out = (
            kwargs["out"]
            if variant.out_from == "out_kw"
            else torch.empty(ref.shape, device=ref.device, dtype=result_dtype)
        )
        compute_dtype = torch.float64 if ref.dtype == torch.float64 else torch.float32
        compute = dt.torch2cute[compute_dtype]
        consts = tuple(compute(s) for s in _scalars(row, args, kwargs, ref.dtype))
        K.run(ops.get_fn(fn_name), row.aten, ins, out, consts, compute)
        return out

    return impl


def register_pointwise_overrides() -> None:
    for row in POINTWISE_DEF_TABLE:
        for variant in POINTWISE_VARIANTS:
            if row.skip_out_variant and variant.out_from == "out_kw":
                continue
            aten_name = variant_aten_name(row.aten, variant)
            if aten_name is not None:
                cu.register_op_override(
                    "aten",
                    aten_name,
                    "CUDA",
                    cond=_make_cond(row, variant),
                    impl=_make_impl(row, variant),
                )

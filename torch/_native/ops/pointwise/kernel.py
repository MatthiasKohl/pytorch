# TMA pointwise execution for transposed and row-strided CUDA inputs.
# Contiguous inputs and unsupported layouts stay on ATen's CUDA kernels.

from __future__ import annotations

import math
import os
from typing import Any, NamedTuple, TYPE_CHECKING

import cuda.bindings.driver as cuda  # noqa: TC002 - CuTeDSL resolves kernel annotations

import cutlass
import cutlass.cute as cute
from cutlass import const_expr
from cutlass.cute.nvgpu import cpasync

import torch

from ...cutedsl import hw_caps as _hw, launch as _L
from ...cutedsl.dtypes import torch2cute
from ...cutedsl.plan_cache import cached_plan


if TYPE_CHECKING:
    from collections.abc import Sequence


_ELEMENTS_PER_THREAD = 8
_TILES_PER_BLOCK = 6
_CONST_SLOTS = 3
_RUBIN_CC = (10, 7)
_ENABLED = os.getenv("TORCH_NATIVE_CUTEDSL_POINTWISE_TUNING", "1") != "0"
_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_PLAN: dict[tuple[Any, ...], _Plan] = {}
_KERNELS: dict[tuple[Any, ...], Any] = {}


class _TmaConfig(NamedTuple):
    transposed: tuple[bool, ...]
    box: int
    block: int
    input_views: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    output_view: tuple[tuple[int, ...], tuple[int, ...]]


class _Plan(NamedTuple):
    config: _TmaConfig
    fn: Any


def _pointer(t: torch.Tensor) -> int:
    with torch._C.DisableTorchFunctionSubclass():
        return t.const_data_ptr()


def _tma_geometry(element_bytes: int) -> tuple[int, int] | None:
    if element_bytes in (2, 4):
        return 32, 128
    if element_bytes == 8:
        return 16, 32
    return None


def _physical_dim_order(strides: tuple[int, ...]) -> tuple[int, ...] | None:
    if not strides or any(stride <= 0 for stride in strides):
        return None
    order = tuple(sorted(range(len(strides)), key=strides.__getitem__))
    if strides[order[0]] != 1:
        return None
    if any(strides[order[i]] <= strides[order[i - 1]] for i in range(1, len(order))):
        return None
    return order


def _dense_run(
    shape: tuple[int, ...], strides: tuple[int, ...], dims: tuple[int, ...]
) -> bool:
    return all(
        shape[dims[i - 1]] * strides[dims[i - 1]] == strides[dims[i]]
        for i in range(1, len(dims))
    )


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [0] * len(shape)
    stride = 1
    for dim in range(len(shape) - 1, -1, -1):
        strides[dim] = stride
        stride *= shape[dim]
    return tuple(strides)


def _tma2d_config(
    inputs: Sequence[torch.Tensor],
    shape: tuple[int, ...],
    output_address: int,
) -> _TmaConfig | None:
    if len(shape) < 2 or not inputs or all(t.is_contiguous() for t in inputs):
        return None
    dtype = inputs[0].dtype
    geometry = _tma_geometry(dtype.itemsize)
    if geometry is None or _hw.caps(inputs[0].device).cc < (9, 0):
        return None
    if (
        math.prod(shape) > torch.iinfo(torch.int32).max
        or any(size <= 0 or size > torch.iinfo(torch.int32).max for size in shape)
        or output_address % 16
        or any(tuple(t.shape) != shape for t in inputs)
    ):
        return None

    output_strides = _contiguous_strides(shape)
    output_order = _physical_dim_order(output_strides)
    if output_order is None:
        return None
    output_positions = {dim: position for position, dim in enumerate(output_order)}
    input_orders = []
    for t in inputs:
        order = _physical_dim_order(t.stride())
        if order is None:
            return None
        input_orders.append(tuple(output_positions[dim] for dim in order))

    box, block = geometry
    tile_bytes = box * box * dtype.itemsize
    smem_bytes = len(inputs) * (_TILES_PER_BLOCK * tile_bytes + 1024)
    smem_bytes += _TILES_PER_BLOCK * 8
    if smem_bytes > _hw.caps(inputs[0].device).smem_per_block_optin:
        return None

    ndim = len(shape)
    identity_order = tuple(range(ndim))
    for a_end in range(1, ndim):
        for b_end in range(a_end + 1, ndim + 1):
            swapped_order = (
                tuple(range(a_end, b_end))
                + tuple(range(a_end))
                + tuple(range(b_end, ndim))
            )
            transposed = []
            for order in input_orders:
                if order == identity_order:
                    transposed.append(False)
                elif order == swapped_order:
                    transposed.append(True)
                else:
                    break
            else:
                a_dims = output_order[:a_end]
                b_dims = output_order[a_end:b_end]
                batch_dims = output_order[b_end:]
                shape_a = math.prod(shape[dim] for dim in a_dims)
                shape_b = math.prod(shape[dim] for dim in b_dims)
                batch = math.prod(shape[dim] for dim in batch_dims)
                grid_x = (shape_a + box * _TILES_PER_BLOCK - 1) // (
                    box * _TILES_PER_BLOCK
                )
                if (
                    shape_a * batch > torch.iinfo(torch.int32).max
                    or shape_b * batch > torch.iinfo(torch.int32).max
                    or grid_x > torch.iinfo(torch.int32).max
                    or shape_b // box > 65535
                    or batch > 65535
                    or shape_a % box
                    or shape_b % box
                    or shape_a * dtype.itemsize % 16
                ):
                    continue

                input_views = []
                valid = True
                for t, is_transposed in zip(inputs, transposed):
                    first = b_dims if is_transposed else a_dims
                    second = a_dims if is_transposed else b_dims
                    second_and_batch = second + batch_dims
                    if not _dense_run(shape, t.stride(), first) or not _dense_run(
                        shape, t.stride(), second_and_batch
                    ):
                        valid = False
                        break
                    fast_extent = shape_b if is_transposed else shape_a
                    higher_stride = t.stride()[second[0]]
                    required_alignment = 128 if is_transposed else 16
                    if (
                        _pointer(t) % required_alignment
                        or higher_stride < fast_extent
                        or fast_extent * dtype.itemsize % 16
                        or higher_stride * dtype.itemsize % 16
                        or higher_stride * dtype.itemsize >= 1 << 40
                    ):
                        valid = False
                        break
                    if is_transposed:
                        view = (
                            (shape_b, batch * shape_a),
                            (t.stride()[first[0]], higher_stride),
                        )
                    else:
                        view = (
                            (batch * shape_b, shape_a),
                            (higher_stride, t.stride()[first[0]]),
                        )
                    input_views.append(view)
                if not valid:
                    continue

                output_stride_a = output_strides[a_dims[0]]
                output_stride_b = output_strides[b_dims[0]]
                output_batch_stride = (
                    output_strides[batch_dims[0]]
                    if batch_dims
                    else shape_b * output_stride_b
                )
                if (
                    output_stride_b * dtype.itemsize % 16
                    or output_batch_stride * dtype.itemsize % 16
                ):
                    continue
                return _TmaConfig(
                    tuple(transposed),
                    box,
                    block,
                    tuple(input_views),
                    (
                        (batch, shape_b, shape_a),
                        (output_batch_stride, output_stride_b, output_stride_a),
                    ),
                )
    return None


def _tma_smem_layout(dtype: Any, box: int, transposed: bool) -> Any:
    inner_bytes = box * dtype.width // 8
    if not transposed:
        swizzle_bits = 0
    elif inner_bytes <= 32:
        swizzle_bits = 1
    elif inner_bytes <= 64:
        swizzle_bits = 2
    else:
        swizzle_bits = 3
    swizzle = cute.make_swizzle(swizzle_bits, 4, 3)
    atom_bytes = 1 << (7 + swizzle_bits)
    atom_elements = atom_bytes * 8 // (8 * dtype.width)
    if transposed:
        outer = cute.make_layout((atom_elements, 8), stride=(1, atom_elements))
        order = (1, 0, 2)
    else:
        outer = cute.make_layout((8, atom_elements), stride=(atom_elements, 1))
        order = (0, 1, 2)
    atom = cute.make_composed_layout(swizzle, 0, outer)
    return cute.tile_to_shape(atom, (box, box, _TILES_PER_BLOCK), order=order)


class _ElementwiseTma2D:
    def __init__(
        self,
        fn: Any,
        ninputs: int,
        nconsts: int,
        compute: Any,
        out_type: Any,
        in_dtype: Any,
        config: _TmaConfig,
    ) -> None:
        self.fn = fn
        self.ninputs = ninputs
        self.nconsts = nconsts
        self.compute = compute
        self.out_type = out_type
        self.in_dtype = in_dtype
        self.transposed = config.transposed
        self.box = config.box
        self.block = config.block

    @cute.jit
    def __call__(
        self,
        mIns: list,
        mOut: cute.Tensor,
        const0,
        const1,
        const2,
        stream: cuda.CUstream,
    ) -> None:
        tma_inputs = []
        layouts = []
        for k in cutlass.range_constexpr(self.ninputs):
            layout = _tma_smem_layout(self.in_dtype, self.box, self.transposed[k])
            info = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(), mIns[k], layout, (self.box, self.box)
            )
            tma_inputs.append(
                (info, cute.zipped_divide(info.tma_tensor, (self.box, self.box)))
            )
            layouts.append(layout)
        batch, rows, cols = mOut.shape
        self.kernel(
            tma_inputs,
            mOut,
            const0,
            const1,
            const2,
            layouts,
        ).launch(
            grid=[
                cute.ceil_div(cols, self.box * _TILES_PER_BLOCK),
                cute.ceil_div(rows, self.box),
                batch,
            ],
            block=[self.block, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tma_inputs,
        mOut: cute.Tensor,
        const0,
        const1,
        const2,
        layouts: Any,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        fast_group, slow_tile, batch = cute.arch.block_idx()
        fast_tile = fast_group * _TILES_PER_BLOCK
        fast_tiles = mOut.shape[2] // const_expr(self.box)
        slow_tiles = mOut.shape[1] // const_expr(self.box)
        valid_stages = cutlass.min(fast_tiles - fast_tile, _TILES_PER_BLOCK)

        smem = cutlass.memory.SmemAllocator()
        buffers = [
            smem.allocate_tensor(
                self.in_dtype,
                layouts[k].outer,
                byte_alignment=1024,
                swizzle=layouts[k].inner,
            )
            for k in range(const_expr(self.ninputs))
        ]
        barriers = smem.allocate_array(cutlass.Int64, num_elems=_TILES_PER_BLOCK)
        tx_bytes = const_expr(
            self.ninputs * self.box * self.box * self.in_dtype.width // 8
        )

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            for k in cutlass.range_constexpr(self.ninputs):
                cpasync.prefetch_descriptor(tma_inputs[k][0].atom)
            for stage in cutlass.range_constexpr(_TILES_PER_BLOCK):
                if stage < valid_stages:
                    barrier = barriers + stage
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_init(barrier, 1)
                        cute.arch.fence_proxy(kind="async.shared", space="cta")
                    for k in cutlass.range_constexpr(self.ninputs):
                        info, tiled = tma_inputs[k]
                        if const_expr(self.transposed[k]):
                            tile_coord = (
                                slow_tile,
                                batch * fast_tiles + fast_tile + stage,
                            )
                        else:
                            tile_coord = (
                                batch * slow_tiles + slow_tile,
                                fast_tile + stage,
                            )
                        g_tile = tiled[((None, None), tile_coord)]
                        s_tile = buffers[k][None, None, stage]
                        cutlass.block.block_copy(
                            info.atom,
                            cute.group_modes(g_tile, 0, 2),
                            cute.group_modes(s_tile, 0, 2),
                            tma_bar_ptr=barrier,
                        )
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(barrier, tx_bytes)
        cute.arch.barrier()

        lane = tidx & 31
        warp_iteration = (tidx >> 5) * _ELEMENTS_PER_THREAD
        slow_base = slow_tile * const_expr(self.box)
        for stage in cutlass.range_constexpr(_TILES_PER_BLOCK):
            if stage < valid_stages:
                cute.arch.mbarrier_wait(barriers + stage, 0)
                fast_base = (fast_tile + stage) * const_expr(self.box)
                for i in cutlass.range_constexpr(_ELEMENTS_PER_THREAD):
                    wi = warp_iteration + i
                    if const_expr(self.in_dtype.width == 32):
                        fast = lane
                        slow = wi ^ (lane >> 3)
                    elif const_expr(self.in_dtype.width == 16):
                        fast = lane
                        slow = wi ^ ((lane >> 4) << 1)
                    else:
                        local = wi * 32 + lane
                        fast = local & 15
                        slow = local >> 4
                    values = []
                    for k in cutlass.range_constexpr(self.ninputs):
                        values.append(self.compute(buffers[k][slow, fast, stage]))
                    if const_expr(self.nconsts == 0):
                        value = self.fn(*values)
                    elif const_expr(self.nconsts == 1):
                        value = self.fn(*values, const0)
                    elif const_expr(self.nconsts == 2):
                        value = self.fn(*values, const0, const1)
                    else:
                        value = self.fn(*values, const0, const1, const2)
                    mOut[batch, slow_base + slow, fast_base + fast] = self.out_type(
                        value
                    )


def _fake_tensor(
    dtype: Any, view: tuple[tuple[int, ...], tuple[int, ...]], align: int
) -> Any:
    shape, stride = view
    return cute.runtime.make_fake_tensor(
        dtype, tuple(shape), stride=tuple(stride), assumed_align=align
    )


def can_run(
    inputs: Sequence[torch.Tensor],
    shape: Sequence[int],
    out_dtype: torch.dtype,
    target: torch.Tensor | None = None,
) -> bool:
    if not _ENABLED or not inputs:
        return False
    ref = inputs[0]
    if ref.device.type != "cuda" or _hw.caps(ref.device).cc != _RUBIN_CC:
        return False
    if ref.dtype not in _DTYPES or out_dtype not in (ref.dtype, torch.bool):
        return False
    limit = torch.iinfo(torch.int32).max
    if any(
        t.device != ref.device
        or t.dtype != ref.dtype
        or t.numel() == 0
        or t.storage_offset() != 0
        or t.is_neg()
        or t.is_conj()
        or sum((n - 1) * s for n, s in zip(t.shape, t.stride())) > limit
        for t in inputs
    ):
        return False
    if target is not None and (
        target.device != ref.device
        or target.dtype != out_dtype
        or tuple(target.shape) != tuple(shape)
        or not target.is_contiguous()
        or target.storage_offset() != 0
        or target.is_neg()
        or target.is_conj()
        or torch._C._is_cow_tensor(target)  # pyrefly: ignore[missing-attribute]
        or any(torch._C._overlaps(target, t) for t in inputs)
    ):
        return False
    address = 0 if target is None else _pointer(target)
    return _tma2d_config(inputs, tuple(shape), address) is not None


def _compile_plan(
    fn: Any,
    aten_name: str,
    inputs: Sequence[torch.Tensor],
    out: torch.Tensor,
    consts: Sequence[Any],
    compute: type[cutlass.Numeric],
) -> _Plan:
    config = _tma2d_config(inputs, tuple(out.shape), _pointer(out))
    if config is None:
        raise AssertionError("pointwise override selected an unsupported TMA layout")
    in_type = torch2cute[inputs[0].dtype]
    out_type = cutlass.Int8 if out.dtype == torch.bool else torch2cute[out.dtype]
    op = _ElementwiseTma2D(
        fn, len(inputs), len(consts), compute, out_type, in_type, config
    )
    fakes = [
        _fake_tensor(in_type, view, 128 if config.transposed[k] else 16)
        for k, view in enumerate(config.input_views)
    ]
    fake_out = _fake_tensor(out_type, config.output_view, 16)
    compile_args = (fakes, fake_out, *([compute(1)] * _CONST_SLOTS), _L.stream())
    caps = _hw.caps(inputs[0].device)
    kernel_key = (
        id(fn),
        len(consts),
        compute,
        in_type,
        out_type,
        config,
        inputs[0].device.index,
        caps.cc,
        caps.smem_per_block_optin,
    )
    compiled = cached_plan(
        _KERNELS,
        kernel_key,
        lambda: _L.compile_kernel(op, *compile_args),
        op=f"aten::{aten_name}",
    )
    return _Plan(config, compiled)


def run(
    fn: Any,
    aten_name: str,
    inputs: Sequence[torch.Tensor],
    out: torch.Tensor,
    consts: Sequence[Any],
    compute: type[cutlass.Numeric],
) -> None:
    caps = _hw.caps(inputs[0].device)
    plan_key = (
        aten_name,
        id(fn),
        compute,
        tuple((t.dtype, tuple(t.shape), t.stride()) for t in inputs),
        out.dtype,
        out.stride(),
        len(consts),
        inputs[0].device.index,
        caps.cc,
        caps.smem_per_block_optin,
        tuple(_pointer(t) % 128 for t in inputs),
        _pointer(out) % 128,
    )
    plan = cached_plan(
        _PLAN,
        plan_key,
        lambda: _compile_plan(fn, aten_name, inputs, out, consts, compute),
    )
    output = out.view(torch.int8) if out.dtype == torch.bool else out
    runtime_inputs = [
        _L.read_only(t.as_strided(*plan.config.input_views[k]))
        for k, t in enumerate(inputs)
    ]
    runtime_out = output.as_strided(*plan.config.output_view)
    runtime_consts = list(consts) + [compute(1)] * (_CONST_SLOTS - len(consts))
    plan.fn(runtime_inputs, runtime_out, *runtime_consts, _L.stream())

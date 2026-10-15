# CuTeDSL pointwise kernels matching the native CUDA fast paths. Unsupported
# layouts and dtype combinations are declined by overrides.py and stay on ATen.

from __future__ import annotations

import math
import os
from typing import Any, NamedTuple, TYPE_CHECKING

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]  # noqa: TC002

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


_BLOCK_THREADS = 128
_ELEMENTS_PER_THREAD = 8
_TILES_PER_BLOCK = 6
_MIN_RUBIN_IO_BYTES = 16 * 1024 * 1024
_RUBIN_CC = (10, 7)
_SCALAR_SLOTS = 3
_CONST_SLOTS = 3
_PLAN: dict[tuple[Any, ...], _Plan] = {}
_KERNELS: dict[tuple[Any, ...], Any] = {}
_ENABLED = os.getenv("TORCH_NATIVE_CUTEDSL_POINTWISE_TUNING", "1") != "0"
_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


class _VectorConfig(NamedTuple):
    path: str
    vec: int
    elements_per_thread: int
    index_bits: int = 32


class _Tma2DConfig(NamedTuple):
    path: str
    transposed: tuple[bool, ...]
    box: int
    block: int
    input_views: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    output_view: tuple[tuple[int, ...], tuple[int, ...]]


class _Plan(NamedTuple):
    config: _VectorConfig | _Tma2DConfig
    fn: Any


def _is_wrapped_scalar(t: torch.Tensor) -> bool:
    return isinstance(t, torch.Tensor) and t.dim() == 0 and t.device.type == "cpu"


def _tensor_inputs(inputs: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    return [t for t in inputs if not _is_wrapped_scalar(t)]


def _scalar_mask(inputs: Sequence[torch.Tensor]) -> tuple[bool, ...]:
    return tuple(_is_wrapped_scalar(t) for t in inputs)


def _pointer(t: torch.Tensor) -> int:
    with torch._C.DisableTorchFunctionSubclass():
        return t.const_data_ptr()


def _vector_config(
    inputs: Sequence[torch.Tensor],
    shape: tuple[int, ...],
    output_address: int,
    out_dtype: torch.dtype | None = None,
) -> _VectorConfig | None:
    if not inputs or any(
        tuple(t.shape) != shape or not t.is_contiguous() for t in inputs
    ):
        return None
    dtype = inputs[0].dtype
    itemsize = dtype.itemsize
    out_dtype = dtype if out_dtype is None else out_dtype
    numel = math.prod(shape)
    cc = _hw.caps(inputs[0].device).cc
    io_bytes = numel * (len(inputs) * itemsize + out_dtype.itemsize)
    if cc == _RUBIN_CC and io_bytes >= _MIN_RUBIN_IO_BYTES:
        vec = min(8, 32 // itemsize)
        input_bytes = len(inputs) * itemsize
        thread_work = (
            1 if input_bytes >= 128 else 128 // (1 << (input_bytes - 1).bit_length())
        )
        thread_work = thread_work if thread_work >= 8 and vec == 8 else 8
    else:
        vec = min(8, 16 // itemsize)
        thread_work = 8
    alignment = vec * itemsize
    if numel == 0 or numel % vec or output_address % (vec * out_dtype.itemsize):
        return None
    nvec = numel // vec
    span = _BLOCK_THREADS * (thread_work // vec)
    blocks = (nvec + span - 1) // span
    limit = torch.iinfo(torch.int32).max
    if blocks > limit:
        return None
    if any(_pointer(t) % alignment for t in inputs):
        return None
    index_bits = 64 if numel - 1 > limit or blocks * span > limit else 32
    return _VectorConfig("vector", vec, thread_work // vec, index_bits)


def _tma_geometry(element_bytes: int) -> tuple[int, int] | None:
    if element_bytes == 1:
        return 64, 512
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
) -> _Tma2DConfig | None:
    if len(shape) < 2 or not inputs:
        return None
    dtype = inputs[0].dtype
    geometry = _tma_geometry(dtype.itemsize)
    if geometry is None or _hw.caps(inputs[0].device).cc < (9, 0):
        return None
    if (
        any(size <= 0 or size > torch.iinfo(torch.int32).max for size in shape)
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
                if not any(transposed):
                    continue
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
                return _Tma2DConfig(
                    "tma2d",
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


def _select_config(
    inputs: Sequence[torch.Tensor],
    shape: tuple[int, ...],
    output_address: int,
    out_dtype: torch.dtype,
    allow_tma: bool = True,
) -> _VectorConfig | _Tma2DConfig | None:
    if allow_tma:
        config = _tma2d_config(inputs, shape, output_address)
        if config is not None:
            return config
    return _vector_config(inputs, shape, output_address, out_dtype)


def can_run(
    inputs: Sequence[torch.Tensor],
    shape: Sequence[int],
    out_dtype: torch.dtype,
    target: torch.Tensor | None = None,
    allow_tma: bool = True,
) -> bool:
    if not _ENABLED or not inputs:
        return False
    tensors = _tensor_inputs(inputs)
    if not tensors or out_dtype not in (*_DTYPES, torch.bool):
        return False
    ref = tensors[0]
    if ref.device.type != "cuda" or _hw.caps(ref.device).cc != _RUBIN_CC:
        return False
    if ref.dtype not in _DTYPES or out_dtype not in (ref.dtype, torch.bool):
        return False
    if any(
        t.device != ref.device
        or t.dtype != ref.dtype
        or t.numel() == 0
        or t.storage_offset() != 0
        or t.is_neg()
        or t.is_conj()
        for t in tensors
    ):
        return False
    if any(_is_wrapped_scalar(t) and t.dtype.is_complex for t in inputs):
        return False
    if target is not None:
        if (
            target.device != ref.device
            or target.dtype != out_dtype
            or tuple(target.shape) != tuple(shape)
            or not target.is_contiguous()
            or target.storage_offset() != 0
            or target.is_neg()
            or target.is_conj()
            or torch._C._is_cow_tensor(target)  # pyrefly: ignore[missing-attribute]
        ):
            return False
        for t in tensors:
            if torch._C._overlaps(target, t) and not (
                t.is_contiguous()
                and t.stride() == target.stride()
                and _pointer(t) == _pointer(target)
            ):
                return False
    address = 0 if target is None else _pointer(target)
    return (
        _select_config(tensors, tuple(shape), address, out_dtype, allow_tma) is not None
    )


def try_run(
    fn: Any,
    key: tuple[Any, ...],
    inputs: Sequence[torch.Tensor],
    out: torch.Tensor,
    consts: Sequence[Any],
    compute: Any,
) -> bool:
    if len(consts) > _CONST_SLOTS or sum(_scalar_mask(inputs)) > _SCALAR_SLOTS:
        return False
    if not can_run(inputs, out.shape, out.dtype, out):
        return False
    run(fn, key, inputs, out, consts, compute)
    return True


def _input_indices(
    scalar_mask: tuple[bool, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    tensor_indices = []
    scalar_indices = []
    tensor_index = 0
    scalar_index = 0
    for is_scalar in scalar_mask:
        if is_scalar:
            tensor_indices.append(-1)
            scalar_indices.append(scalar_index)
            scalar_index += 1
        else:
            tensor_indices.append(tensor_index)
            scalar_indices.append(-1)
            tensor_index += 1
    return tuple(tensor_indices), tuple(scalar_indices)


class _ElementwiseVector:
    def __init__(
        self,
        fn: Any,
        scalar_mask: tuple[bool, ...],
        nconsts: int,
        compute: Any,
        out_type: Any,
        config: _VectorConfig,
    ) -> None:
        self.fn = fn
        self.scalar_mask = scalar_mask
        self.tensor_indices, self.scalar_indices = _input_indices(scalar_mask)
        self.nargs = len(scalar_mask)
        self.ninputs = len(scalar_mask) - sum(scalar_mask)
        self.nconsts = nconsts
        self.compute = compute
        self.out_type = out_type
        self.vec = config.vec
        self.ept = config.elements_per_thread
        self.index_type = cutlass.Int64 if config.index_bits == 64 else cutlass.Int32

    @cute.jit
    def __call__(
        self,
        mIns: list,
        mOut: cute.Tensor,
        scalar0,
        scalar1,
        scalar2,
        const0,
        const1,
        const2,
        stream: cuda.CUstream,
    ) -> None:
        nvec = self.index_type(mOut.shape[0])
        self.kernel(
            mIns,
            mOut,
            scalar0,
            scalar1,
            scalar2,
            const0,
            const1,
            const2,
            nvec,
        ).launch(
            grid=[cute.ceil_div(nvec, const_expr(_BLOCK_THREADS * self.ept)), 1, 1],
            block=[_BLOCK_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mIns: list,
        mOut: cute.Tensor,
        scalar0,
        scalar1,
        scalar2,
        const0,
        const1,
        const2,
        nvec,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        scalar_ins = (scalar0, scalar1, scalar2)
        base = self.index_type(bidx) * const_expr(_BLOCK_THREADS * self.ept) + tidx
        for p in cutlass.range_constexpr(self.ept):
            index = base + p * _BLOCK_THREADS
            if index < nvec:
                regs = []
                for k in cutlass.range_constexpr(self.ninputs):
                    reg = cute.make_rmem_tensor_like(mIns[k][index, None])
                    cute.autovec_copy(mIns[k][index, None], reg)
                    regs.append(reg)
                result = cute.make_rmem_tensor_like(mOut[index, None])
                for e in cutlass.range_constexpr(self.vec):
                    values = []
                    for arg in cutlass.range_constexpr(self.nargs):
                        if const_expr(self.scalar_mask[arg]):
                            values.append(scalar_ins[self.scalar_indices[arg]])
                        else:
                            values.append(
                                self.compute(regs[self.tensor_indices[arg]][e])
                            )
                    if const_expr(self.nconsts == 0):
                        value = self.fn(*values)
                    elif const_expr(self.nconsts == 1):
                        value = self.fn(*values, const0)
                    elif const_expr(self.nconsts == 2):
                        value = self.fn(*values, const0, const1)
                    else:
                        value = self.fn(*values, const0, const1, const2)
                    result[e] = self.out_type(value)
                cute.autovec_copy(result, mOut[index, None])


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
        scalar_mask: tuple[bool, ...],
        nconsts: int,
        compute: Any,
        out_type: Any,
        in_dtype: Any,
        config: _Tma2DConfig,
    ) -> None:
        self.fn = fn
        self.scalar_mask = scalar_mask
        self.tensor_indices, self.scalar_indices = _input_indices(scalar_mask)
        self.nargs = len(scalar_mask)
        self.ninputs = len(scalar_mask) - sum(scalar_mask)
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
        scalar0,
        scalar1,
        scalar2,
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
            scalar0,
            scalar1,
            scalar2,
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
        scalar0,
        scalar1,
        scalar2,
        const0,
        const1,
        const2,
        layouts: Any,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        fast_group, slow_tile, batch = cute.arch.block_idx()
        scalar_ins = (scalar0, scalar1, scalar2)
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
                    elif const_expr(self.in_dtype.width == 8):
                        fast = lane | ((wi & 1) << 5)
                        slow = wi >> 1
                    else:
                        local = wi * 32 + lane
                        fast = local & 15
                        slow = local >> 4
                    values = []
                    for arg in cutlass.range_constexpr(self.nargs):
                        if const_expr(self.scalar_mask[arg]):
                            values.append(scalar_ins[self.scalar_indices[arg]])
                        else:
                            k = self.tensor_indices[arg]
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


def _compile_plan(
    fn: Any,
    key: tuple[Any, ...],
    inputs: Sequence[torch.Tensor],
    out: torch.Tensor,
    scalar_mask: tuple[bool, ...],
    consts: Sequence[Any],
    compute: Any,
) -> _Plan:
    tensors = _tensor_inputs(inputs)
    shape = tuple(tensors[0].shape)
    config = _select_config(tensors, shape, _pointer(out), out.dtype)
    if config is None:
        raise AssertionError("pointwise override selected an unsupported launch")
    in_type = torch2cute[tensors[0].dtype]
    out_type = torch2cute[torch.int8 if out.dtype == torch.bool else out.dtype]
    if sum(scalar_mask) > _SCALAR_SLOTS or len(consts) > _CONST_SLOTS:
        raise AssertionError("pointwise scalar slot count exceeded")
    scalar_seeds = [compute(1)] * _SCALAR_SLOTS
    const_seeds = [compute(1)] * _CONST_SLOTS
    if isinstance(config, _VectorConfig):
        op = _ElementwiseVector(fn, scalar_mask, len(consts), compute, out_type, config)
        fakes = [
            _L.fake_compact(
                in_type,
                (cute.sym_int(width=config.index_bits), config.vec),
                stride_order=(1, 0),
                align=config.vec * tensors[0].element_size(),
            )
            for _ in tensors
        ]
        fake_out = _L.fake_compact(
            out_type,
            (cute.sym_int(width=config.index_bits), config.vec),
            stride_order=(1, 0),
            align=config.vec * out.element_size(),
        )
    else:
        op = _ElementwiseTma2D(
            fn, scalar_mask, len(consts), compute, out_type, in_type, config
        )
        fakes = [
            _fake_tensor(
                in_type,
                config.input_views[k],
                128 if config.transposed[k] else 16,
            )
            for k in range(len(tensors))
        ]
        fake_out = _fake_tensor(out_type, config.output_view, 16)
    caps = _hw.caps(tensors[0].device)
    kernel_key = (
        id(fn),
        scalar_mask,
        len(consts),
        compute,
        in_type,
        out_type,
        config,
        tensors[0].device.index,
        caps.cc,
        caps.smem_per_block_optin,
    )
    compiled = cached_plan(
        _KERNELS,
        kernel_key,
        lambda: _L.compile_kernel(
            op, fakes, fake_out, *scalar_seeds, *const_seeds, _L.stream()
        ),
        op=f"aten::{key[0]}",
    )
    return _Plan(config, compiled)


def run(
    fn: Any,
    key: tuple[Any, ...],
    inputs: Sequence[torch.Tensor],
    out: torch.Tensor,
    consts: Sequence[Any],
    compute: Any,
) -> None:
    scalar_mask = _scalar_mask(inputs)
    tensors = _tensor_inputs(inputs)
    caps = _hw.caps(tensors[0].device)
    plan_key = (
        key[0],
        id(fn),
        compute,
        tuple((t.dtype, tuple(t.shape), t.stride()) for t in tensors),
        out.dtype,
        out.stride(),
        len(consts),
        scalar_mask,
        tensors[0].device.index,
        caps.cc,
        caps.smem_per_block_optin,
        tuple(_pointer(t) % 128 for t in tensors),
        _pointer(out) % 128,
    )
    plan = cached_plan(
        _PLAN,
        plan_key,
        lambda: _compile_plan(fn, key, inputs, out, scalar_mask, consts, compute),
    )
    scalar_inputs = [
        compute(t.item()) for t, is_scalar in zip(inputs, scalar_mask) if is_scalar
    ]
    scalar_inputs.extend([compute(1)] * (_SCALAR_SLOTS - len(scalar_inputs)))
    runtime_consts = list(consts)
    runtime_consts.extend([compute(1)] * (_CONST_SLOTS - len(runtime_consts)))
    output = out.view(torch.int8) if out.dtype == torch.bool else out
    if plan.config.path == "vector":
        vec = plan.config.vec
        runtime_inputs = [_L.read_only(t.reshape(-1, vec)) for t in tensors]
        runtime_out = output.reshape(-1, vec)
    else:
        runtime_inputs = [
            _L.read_only(t.as_strided(*plan.config.input_views[k]))
            for k, t in enumerate(tensors)
        ]
        runtime_out = output.as_strided(*plan.config.output_view)
    plan.fn(
        runtime_inputs,
        runtime_out,
        *scalar_inputs,
        *runtime_consts,
        _L.stream(),
    )

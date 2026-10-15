# Generic elementwise (pointwise) CuteDSL kernels.
#
# One op-agnostic kernel family serves every row of the pointwise definition table.
# Each thread applies the row's @cute.jit `fn` (with baked scalar consts) to inputs
# converted to the compute dtype, storing each output cast to its out dtype. No
# cross-thread communication.
#
# THREE paths, chosen host-side:
#   FAST (vec): all operands contiguous, identical shape, numel % V == 0. The
#     arrays coalesce to a flat (numel/V, V) layout; each thread vector-loads a
#     V-wide fragment (V*dtype = 128 bits -> wide global load), computes, stores.
#     This is the bandwidth path and hits ~parity with aten.
#   ROWVEC: all operands share one ROW-DENSE-GAPPED geometry ([M,N]:(K,1) with K>N,
#     or an n-D last-dim slice; see _row_gap_view). Operands are viewed as
#     (rows, run/V, V): the within-row V is a contiguous 128-bit vector, only the
#     row base is gapped -- vectorizing the density aten's scalar offset-calculator
#     path ignores (~2x aten measured). Output is contiguous (matches aten).
#   GENERAL (strided): anything else (broadcast / ragged numel / non-float-out).
#     Operands are expanded to the broadcast shape and wrapped via from_dlpack,
#     which carries each operand's real layout (broadcast dims are stride-0); the
#     kernel indexes linearly and CuTe decodes the offset. Correct for all cases,
#     not vectorized -- which is why the override cond DECLINES irregular layouts
#     (transpose / channels-last / mixed gaps) to aten instead of serving them here.
#
# Addressing is canonical CuTe in both paths -- no hand-rolled offset math.

import math
import os
from typing import Any, NamedTuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import const_expr
from cutlass.cute.nvgpu import cpasync

import torch

from ...cutedsl import hw_caps as _hw, launch as _L
from ...cutedsl.dtypes import torch2cute
from ...cutedsl.plan_cache import cached_plan
from .indexing import can_use_32bit_index_math


# Exposed perf knobs (autotunable). run() fills any knob left None from _choose_config
# (the measured per-(numel, dtype) heuristic); the constants below name each knob's
# baseline value and candidate range. Pass explicit values to run() to override (the
# autotuner, or a per-machine retune).
#   _BLOCK    -- threads per block.
#   _EPT      -- elements per thread: each thread processes _EPT items (vec-rows on the
#               fast path, scalar elements on the strided path) via a block-strided,
#               CONSTEXPR-unrolled loop. _EPT is thus BOTH the work-coarsening and the
#               loop-unroll knob (the inner body is unrolled _EPT deep). >1 raises
#               ILP/occupancy for compute-bound ops (transcendentals: exp/log/erf);
#               =1 is the pure-bandwidth default. The grid shrinks by _EPT.
#   _VEC_BITS -- load/store vector width in bits (128 default; 64/256 candidates).
#   _LOAD     -- gmem->reg load path for the FAST (vec) route:
#               "direct"  -- gmem->rmem via autovec_copy (default; pure bandwidth).
#               "cpasync" -- cp.async gmem->smem ring (cpasync.CopyG2SOp), _PIPE_DEPTH
#                            deep, overlapping the next segment's load with this
#                            segment's compute. Wins on compute-bound ops (~1.05x on
#                            exp); flat vs depth (elementwise is largely bandwidth-
#                            bound). All CuteDSL primitives -- no hand-rolled PTX.
#               "tma"     -- tile-mode TMA (cp.async.bulk.tensor) gmem->smem via
#                            make_tiled_tma_atom + PipelineTmaAsync (single (block,V)
#                            tile/CTA, ept==1). Requires a >=128-bit inner box (V*dtype
#                            a multiple of 128b) or the atom is malformed
#                            (cudaErrorIllegalInstruction); _build_plan falls back to
#                            direct when that (or ept>1) doesn't hold. Shows no
#                            elementwise upside vs cpasync (bandwidth-bound, no tile
#                            reuse) -- kept as a selectable option for the autotuner
#                            (research code), not a recommended default.
#               The strided (broadcast/irregular) route is always "direct".
#   _PIPE_DEPTH -- ring depth for cpasync/tma (>=1; 1 == no overlap).
_BLOCK = 256
_EPT = 1
_VEC_BITS = 128
_LOAD = "direct"
_PIPE_DEPTH = 2
_PLAN = {}  # key -> _Plan (compiled kernel + shape-invariant launch decisions)
# Compiled-kernel dedup, one level BELOW _PLAN: distinct plan keys (different shapes /
# scalar values) that bake identically share one compiled kernel. The vec path's key
# carries no shape (dynamic nvec) and no path's key carries scalar values (runtime
# args), so kernel count is O(op x dtype-combo x knobs), not O(call diversity).
_KERNELS = {}


class _PointwiseConfig(NamedTuple):
    # The full pointwise knob set (specific to this kernel family: the reduction kernels
    # have their own, differently-shaped knobs -- block_x/grid_y/tpr/subrow_target/...).
    # run() knobs left None are filled from _choose_config; explicit values (autotuner /
    # per-machine retune) override per field.
    block: int = 256
    ept: int = 1
    vec_bits: int = 128
    load: str = "direct"
    pipe_depth: int = 1


# ept (thread coarsening) is the ONE first-order knob, as an ordered threshold TABLE
# (min DEVICE-FILL count, ept); first row whose threshold is met wins. Coarsening refills
# the memory pipe once the grid exceeds the device's resident-thread capacity (~1.3x at
# scale). Thresholds are the B200-tuned anchors, expressed as multiples of that device's
# lane capacity so they scale to any GPU (see _choose_config): B200 has ~303K lanes, so
# e2 at ~384K = ~1.27 waves, e4 at ~768K = ~2.53 waves. Normalizing numel by dtype width
# (vs 16b) bakes in "narrow dtypes pack more per wide load and starve sooner": bf16 hits
# the thresholds directly, fp32 at ~2x the element count, fp64 ~4x.
_EPT_SCHEDULE = ((768 * 1024, 4), (384 * 1024, 2), (0, 1))
# Wide dtypes (>=32b) at large sizes take a 256b vector (marginal but consistent edge);
# narrow dtypes already saturate the load at 128b. min numel is a device-fill multiple.
_VEC256_MIN = (32, 8 * 1024 * 1024)


def _choose_config(numel: int, compute_bits: int, hw=None, nin=1) -> "_PointwiseConfig":
    # Measured per-(numel, dtype) heuristic (B200; fp32 + bf16 threshold study), scaled to
    # the device. Pointwise has ONE effective dimension (flat numel), so the config is a
    # function of numel, dtype, and device fill capacity. block / load / pipe_depth are
    # second-order (+-1-4%) -> the config defaults. ept (table) and vec_bits (wide-dtype-
    # at-scale) move with the shape; both thresholds scale by hw.fill_scale so a larger
    # GPU coarsens at a proportionally larger numel and a smaller one sooner (fill_scale
    # is 1.0 on B200 -> the anchor numbers reproduce exactly).
    fill = 1.0 if hw is None else hw.sm_count * hw.max_threads_per_sm / (148 * 2048)
    norm = numel // max(compute_bits // 16, 1)  # dtype-normalized element count
    ept = next(e for thresh, e in _EPT_SCHEDULE if norm >= thresh * fill)
    # The 256b widening was measured on ops that READ memory, where the extra width buys
    # load throughput. A WRITE-ONLY op (nin == 0: fill_ and the constructors) has no loads
    # to widen and 256b nearly HALVES it -- measured on a 64M fp32 fill_: 3.74 TB/s at
    # 256b vs 6.87 TB/s at 128b, where aten is 6.83. So gate the widening on having inputs.
    wide = nin > 0 and compute_bits >= _VEC256_MIN[0] and numel >= _VEC256_MIN[1] * fill
    return _PointwiseConfig(ept=ept, vec_bits=256 if wide else 128)


def _vec_width(compute_bits: int, vec_bits: int = _VEC_BITS) -> int:
    # Elements per `vec_bits`-wide vector for the compute dtype (at 128b: fp32->4,
    # fp16/bf16->8, fp64->2). vec_bits is the exposed load/store-width knob (default
    # 128 = the wide LDG/STG target; 64/256 the other candidates). The fast path
    # requires numel divisible by this.
    return max(vec_bits // compute_bits, 1)


class _ElementwiseVec:
    # Vectorized flat path. Operands are (nvec, V) cute tensors; each thread owns
    # `ept` V-wide rows, strided by block*ept across the grid. block/ept/V and the
    # load path (direct/cpasync/tma) + pipe_depth are the exposed knobs. For cpasync/
    # tma the per-input V-row is staged through a pipe_depth-deep smem ring so the
    # next segment's async load overlaps this segment's compute; direct loads
    # gmem->rmem inline. All three share `_compute_store` (the fn + cast + store).
    #
    # Scalar args (add's alpha, softplus's beta, ...) are RUNTIME `consts` list args,
    # not baked constants: one compiled kernel serves every scalar value (measured
    # free -- the scalar broadcast rides the same registers). nconsts (the arity) is
    # compile-time; the values are not.
    def __init__(
        self,
        fn,
        nin,
        nout,
        nconsts,
        compute,
        out_types,
        V,
        block,
        ept,
        load,
        pipe_depth,
        in_dtype,
    ):
        self.fn = fn
        self.nin = nin
        self.nout = nout
        self.nconsts = nconsts
        self.compute = compute
        self.out_types = out_types
        self.V = V
        self.block = block
        self.ept = ept
        self.load = load
        self.pipe_depth = pipe_depth
        # Input element dtype (a concrete Numeric class). Needed for the tma/cpasync
        # smem allocations: under TMA mIns[k].element_type is a descriptor union, not
        # a plain Numeric, so we can't read it off the wrapped tensor there.
        self.in_dtype = in_dtype

    @cute.jit
    def __call__(self, mIns: list, mOuts: list, consts: list, stream: cuda.CUstream):
        nvec = mOuts[0].shape[0]
        tile = const_expr(self.block * self.ept)  # vec-rows retired per block
        # tile-mode TMA atoms (host-compile-time, inside the jit region), one per
        # input; only the tma load path uses them. ept==1 for tma (host guard), so the
        # tile is (block, V).
        tma_atoms = None
        ins = mIns
        if const_expr(self.load == "tma"):
            lay = cute.make_ordered_layout(
                (self.block * self.ept, self.V), order=(1, 0)
            )
            tma_atoms = []
            tviews = []
            for k in cutlass.range_constexpr(self.nin):
                atom, v = cpasync.make_tiled_tma_atom(
                    cpasync.CopyBulkTensorTileG2SOp(),
                    mIns[k],
                    lay,
                    (self.block * self.ept, self.V),
                )
                tma_atoms.append(atom)
                tviews.append(v)
            ins = tviews
        self.kernel(ins, mOuts, consts, nvec, tma_atoms).launch(
            grid=[cute.ceil_div(nvec, tile), 1, 1],
            block=[self.block, 1, 1],
            stream=stream,
        )

    @cute.jit
    def _compute_store(self, regs, mOuts, consts, i):
        # Shared fn + per-element compute + cast + vector store for one V-row `i`.
        # regs: list[nin] of (V,) register fragments (the loaded inputs).
        V = const_expr(self.V)
        outs = [cute.make_rmem_tensor_like(mOuts[j][i, None]) for j in range(self.nout)]
        for e in cutlass.range_constexpr(V):
            vals = tuple(self.compute(regs[k][e]) for k in range(const_expr(self.nin)))
            res = self.fn(*vals, *consts)
            if const_expr(self.nout == 1):
                outs[0][e] = self.out_types[0](res)
            else:
                for j in cutlass.range_constexpr(self.nout):
                    outs[j][e] = self.out_types[j](res[j])
        for j in cutlass.range_constexpr(self.nout):
            cute.autovec_copy(outs[j], mOuts[j][i, None])

    @cute.kernel
    def kernel(
        self, mIns: list, mOuts: list, consts: list, nvec: cutlass.Int32, tma_atoms
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        block = const_expr(self.block)
        base = bidx * const_expr(block * self.ept) + tidx
        if const_expr(self.load == "direct"):
            # gmem->rmem inline, per-thread block-strided ept-unrolled loop.
            for p in cutlass.range_constexpr(self.ept):
                i = base + p * block
                if i < nvec:
                    regs = []
                    for k in cutlass.range_constexpr(self.nin):
                        r = cute.make_rmem_tensor_like(mIns[k][i, None])
                        cute.autovec_copy(mIns[k][i, None], r)
                        regs.append(r)
                    self._compute_store(regs, mOuts, consts, i)
        elif const_expr(self.load == "cpasync"):
            self._kernel_cpasync(mIns, mOuts, consts, nvec, base, block)
        else:  # "tma"
            self._kernel_tma(mIns, mOuts, consts, nvec, tma_atoms, base, block)

    @cute.jit
    def _kernel_cpasync(self, mIns, mOuts, consts, nvec, base, block):
        # cp.async gmem->smem ring, pipe_depth deep. Segment p's V-row for each input
        # is staged into sBuf[input][p % D]; the next segment's async copy overlaps
        # this segment's compute. Uses cpasync.CopyG2SOp + commit/wait_group only.
        D = const_expr(self.pipe_depth)
        V = const_expr(self.V)
        ept = const_expr(self.ept)
        dt = self.in_dtype
        smem = cutlass.utils.SmemAllocator()
        sBuf = [
            smem.allocate_tensor(
                dt,
                cute.make_ordered_layout((D, self.block, V), order=(2, 1, 0)),
                byte_alignment=16,
            )
            for k in range(const_expr(self.nin))
        ]
        atoms = [
            cute.make_copy_atom(
                cpasync.CopyG2SOp(), dt, num_bits_per_copy=const_expr(V * dt.width)
            )
            for k in range(const_expr(self.nin))
        ]
        tidx, _, _ = cute.arch.thread_idx()

        # Prologue: issue the first min(D, ept) segment loads (inlined -- no closure;
        # CuteDSL rejects closures that capture kernel-local values).
        for p in cutlass.range_constexpr(D):
            if p < ept:
                ip = base + p * block
                for k in cutlass.range_constexpr(self.nin):
                    if ip < nvec:
                        cute.copy(
                            atoms[k], mIns[k][ip, None], sBuf[k][p % D, tidx, None]
                        )
                cute.arch.cp_async_commit_group()
        for s in cutlass.range_constexpr(ept):
            # keep min(D, ept-s)-1 groups in flight so seg s (the oldest) is ready.
            cute.arch.cp_async_wait_group(const_expr(min(D, ept - s) - 1))
            cute.arch.barrier()
            i = base + s * block
            if i < nvec:
                regs = []
                for k in cutlass.range_constexpr(self.nin):
                    r = cute.make_rmem_tensor_like(sBuf[k][s % D, tidx, None])
                    cute.autovec_copy(sBuf[k][s % D, tidx, None], r)
                    regs.append(r)
                self._compute_store(regs, mOuts, consts, i)
            # Issue seg s+D into the freed ring slot ((s+D)%D == s%D).
            if const_expr(s + D < ept):
                ni = base + const_expr(s + D) * block
                for k in cutlass.range_constexpr(self.nin):
                    if ni < nvec:
                        cute.copy(
                            atoms[k],
                            mIns[k][ni, None],
                            sBuf[k][(s + D) % D, tidx, None],
                        )
                cute.arch.cp_async_commit_group()

    @cute.jit
    def _kernel_tma(self, mIns, mOuts, consts, nvec, tma_atoms, base, block):
        # Tile-mode TMA gmem->smem via PipelineTmaAsync (correct mbarrier/phase). One
        # bulk load of the whole (block, V) tile per input (ept==1 for tma), then each
        # thread computes its row from smem. All CuteDSL primitives (make_tiled_tma_atom
        # + PipelineTmaAsync + tma_partition + cute.copy) -- no hand-rolled mbarrier.
        from cutlass import pipeline

        V = const_expr(self.V)
        ept = const_expr(self.ept)
        tm = const_expr(self.block * self.ept)
        dt = self.in_dtype
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        sBuf = [
            smem.allocate_tensor(
                dt, cute.make_ordered_layout((tm, V), order=(1, 0)), byte_alignment=16
            )
            for k in range(const_expr(self.nin))
        ]
        mbar = smem.allocate_array(cutlass.Int64, num_elems=2)
        # Bytes the TMA transfers into smem across all inputs = the barrier's tx_count.
        tx = const_expr(self.nin * tm * V * dt.width // 8)
        pipe = pipeline.PipelineTmaAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, const_expr(self.block)
            ),
            tx_count=tx,
            barrier_storage=mbar,
            tidx=tidx,
        )
        ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        if cute.arch.warp_idx() == 0:
            pipe.producer_acquire(ps)
            bar = pipe.producer_get_barrier(ps)
            for k in cutlass.range_constexpr(self.nin):
                g = cute.local_tile(mIns[k], (tm, V), (bidx, 0))
                s, gg = cpasync.tma_partition(
                    tma_atoms[k],
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sBuf[k], 0, 2),
                    cute.group_modes(g, 0, 2),
                )
                cute.copy(tma_atoms[k], gg, s, tma_bar_ptr=bar)
            pipe.producer_commit(ps)
        pipe.consumer_wait(cs)
        for p in cutlass.range_constexpr(ept):
            i = base + p * block  # global vec-row
            r_local = tidx + p * block  # row within this block's (tm, V) tile
            if i < nvec:
                regs = []
                for k in cutlass.range_constexpr(self.nin):
                    r = cute.make_rmem_tensor_like(sBuf[k][r_local, None])
                    cute.autovec_copy(sBuf[k][r_local, None], r)
                    regs.append(r)
                self._compute_store(regs, mOuts, consts, i)
        pipe.consumer_release(cs)


class _ElementwiseStrided:
    # General path: each thread retires `ept` output elements (block-strided), linear
    # index into each operand's (possibly broadcast / strided) layout -- CuTe decodes
    # the offset. block/ept are the exposed knobs.
    #
    # with_index: pass the thread's FLAT OUTPUT INDEX to `fn` as its first argument, the
    # analogue of aten's gpu_kernel_with_index. This is what the RANGE factories need
    # (arange/linspace compute their value from the index alone), and it is exposed only
    # on this path because the vec/rowvec routes hand the fn a vector fragment rather
    # than one element, so there is no single index to give it.
    def __init__(
        self, fn, nin, nout, nconsts, compute, out_types, block, ept, with_index=False
    ):
        self.fn = fn
        self.nin = nin
        self.nout = nout
        self.nconsts = nconsts
        self.compute = compute
        self.out_types = out_types
        self.block = block
        self.ept = ept
        self.with_index = with_index

    @cute.jit
    def __call__(self, mIns: list, mOuts: list, consts: list, stream: cuda.CUstream):
        n = cute.size(mOuts[0])
        tile = const_expr(self.block * self.ept)
        self.kernel(mIns, mOuts, consts, n).launch(
            grid=[cute.ceil_div(n, tile), 1, 1],
            block=[self.block, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mIns: list, mOuts: list, consts: list, n: cutlass.Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        block = const_expr(self.block)
        base = bidx * const_expr(block * self.ept) + tidx
        for p in cutlass.range_constexpr(self.ept):
            i = base + p * block
            if i < n:
                vals = tuple(
                    self.compute(mIns[k][i]) for k in range(const_expr(self.nin))
                )
                # Int64 index: aten's gpu_kernel_with_index hands the lambda an int64_t,
                # and arange's size can exceed Int32 (the step multiply happens in the
                # accumulate type, so the index must not wrap first).
                if const_expr(self.with_index):
                    outs = self.fn(cutlass.Int64(i), *vals, *consts)
                else:
                    outs = self.fn(*vals, *consts)
                if const_expr(self.nout == 1):
                    mOuts[0][i] = self.out_types[0](outs)
                else:
                    for j in cutlass.range_constexpr(self.nout):
                        mOuts[j][i] = self.out_types[j](outs[j])


class _ElementwiseRowVec:
    # Vectorized ROW-DENSE-GAPPED path. Inputs are (rows, ncv, V) cute tensors carrying
    # the gap in the row stride (ncv = run//V vec-chunks per row); the output is
    # CONTIGUOUS, viewed the same (rows, ncv, V) way. Each thread owns one (row, chunk)
    # V-fragment: the within-row V load is a contiguous 128-bit vector (coalesced), only
    # the row base is gapped -- the density aten's scalar offset path ignores (~2x aten).
    # Reuses _ElementwiseVec._compute_store's per-element fn+cast+store over the V lane.
    def __init__(self, fn, nin, nout, nconsts, compute, out_types, V, ncv, block):
        self.fn = fn
        self.nin = nin
        self.nout = nout
        self.nconsts = nconsts
        self.compute = compute
        self.out_types = out_types
        self.V = V
        self.ncv = ncv  # vec-chunks per row (run // V)
        self.block = block

    @cute.jit
    def __call__(self, mIns: list, mOuts: list, consts: list, stream: cuda.CUstream):
        rows = mOuts[0].shape[0]
        nvec = rows * const_expr(self.ncv)  # total (row, chunk) fragments
        self.kernel(mIns, mOuts, consts, nvec).launch(
            grid=[cute.ceil_div(nvec, self.block), 1, 1],
            block=[self.block, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mIns: list, mOuts: list, consts: list, nvec: cutlass.Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * const_expr(self.block) + tidx
        ncv = const_expr(self.ncv)
        V = const_expr(self.V)
        if i < nvec:
            m = i // ncv  # row
            c = i % ncv  # vec-chunk within row
            regs = []
            for k in cutlass.range_constexpr(self.nin):
                r = cute.make_rmem_tensor_like(mIns[k][m, c, None])
                cute.autovec_copy(mIns[k][m, c, None], r)
                regs.append(r)
            outs = [
                cute.make_rmem_tensor_like(mOuts[j][m, c, None])
                for j in range(self.nout)
            ]
            for e in cutlass.range_constexpr(V):
                vals = tuple(
                    self.compute(regs[k][e]) for k in range(const_expr(self.nin))
                )
                res = self.fn(*vals, *consts)
                if const_expr(self.nout == 1):
                    outs[0][e] = self.out_types[0](res)
                else:
                    for j in cutlass.range_constexpr(self.nout):
                        outs[j][e] = self.out_types[j](res[j])
            for j in cutlass.range_constexpr(self.nout):
                cute.autovec_copy(outs[j], mOuts[j][m, c, None])


# Output dtypes the vec path can vector-store. The kernel computes in `compute` and
# casts per element (out_types[j](res)) before the wide copy, so the OUT dtype need not
# equal compute -- it only must be a real float (bf16/fp16 out with fp32 compute is the
# common case: aten promotes bf16 unary/binary math to fp32 then stores bf16). The
# integer widths (int32/int64) ride the vec path too: the int->int arithmetic ops
# (add/mul/neg/...) have input == compute == output at one width, so the wide
# (numel/V, V) load/store is well-formed (verified). Still EXCLUDED: bool (comparison
# out) and the frexp int32 exponent -- a 1-byte or width-mismatched output can't take
# the vectorized store, so those ops fall to the general (strided) path.
_VEC_OUT_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int32,
    torch.int64,
    # Full integer matrix (conversion overrides; int8 also backs the bool view).
    torch.int8,
    torch.int16,
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
)


def _vec_ok(inputs, shape, out_dtypes, compute_torch, V, out_ref=None):
    # Fast path requires: every input already has the output shape (no broadcast), is
    # contiguous, the element count is a multiple of V (exact (numel/V, V) reshape), AND
    # every output is a vector-storable float (see _VEC_OUT_DTYPES).
    #
    # For a NULLARY op (nin == 0) the `all(... for t in inputs)` tests below are vacuously
    # true, so the shape/contiguity/alignment obligation falls entirely on the OUTPUT --
    # check out_ref explicitly or an unaligned in-place target would take the vec path and
    # fault in from_dlpack.
    if any(d not in _VEC_OUT_DTYPES for d in out_dtypes):
        return False
    numel = math.prod(shape)
    if numel == 0 or numel % V != 0:
        return False
    probes = list(inputs) if inputs else ([out_ref] if out_ref is not None else [])
    if not probes:
        return False
    if not all(_is_16b_aligned(t) for t in probes):
        return False
    return all(tuple(t.shape) == tuple(shape) and t.is_contiguous() for t in probes)


def _is_16b_aligned(t) -> bool:
    # The vec/rowvec wraps pass assumed_align=16 so the DSL may emit 128-bit loads;
    # from_dlpack VALIDATES that promise and raises "Tensor data pointer is not aligned
    # to 16 bytes" otherwise. A fresh torch allocation is >=256 B aligned, but a VIEW
    # into one is not: base[1:5] on int32 starts 4 B in. That raise is a hard crash on
    # an ordinary call (it hit mul/add/maximum on a sliced narrow-dtype operand), so
    # gate the fast path on the real pointer and let unaligned operands take the
    # strided path instead.
    #
    # const_data_ptr, NOT data_ptr: this runs from a cond on every input, and data_ptr()
    # MATERIALIZES a copy-on-write tensor (breaking the COW-preservation contract the
    # read-only export exists to honour). The const accessor reads the same address
    # without taking write ownership.
    return t.const_data_ptr() % 16 == 0


def _row_gap_view(t):
    # Classify a ROW-DENSE, ROW-GAPPED tensor -> (rows, run, row_step) in ELEMENTS, or
    # None. "run" is the length of the maximal dense innermost run (stride-1 suffix whose
    # dims are mutually contiguous); "rows" = numel/run; "row_step" is the uniform stride
    # between runs. Requires: (1) a real gap (run < numel, else it's just contiguous ->
    # the flat vec path), and (2) the dims ABOVE the run collapse to ONE uniform step
    # (mutually contiguous), so the tensor is exactly (rows, run) with row stride
    # row_step. Transpose / channels-last (no stride-1 dense innermost) and multi-gap
    # layouts (outer not uniform) return None -> the general strided path.
    shape, stride = tuple(t.shape), t.stride()
    if not shape or stride[-1] != 1:
        return None
    run, expect, k = 1, 1, 0  # grow the dense innermost run
    for i in range(len(shape) - 1, -1, -1):
        if stride[i] == expect:
            run *= shape[i]
            expect *= shape[i]
            k += 1
        else:
            break
    if k == len(shape):
        return None  # fully dense -> not gapped (flat vec path handles it)
    outer_shape, outer_stride = shape[: len(shape) - k], stride[: len(shape) - k]
    for i in range(len(outer_shape) - 1):  # outer dims must be mutually contiguous
        if outer_shape[i + 1] * outer_stride[i + 1] != outer_stride[i]:
            return None
    return t.numel() // run, run, outer_stride[-1]


# --- OPERAND VIEWS and their compile-time DESCRIPTORS. Each path reshapes or restrides the torch
# tensor one way and describes that same layout to the compile; keeping the pair adjacent is what
# stops the two from drifting (they used to be one wrap call, so they could not).


def _view_vec(t, V):
    # (numel/V, V): row i is one thread's V-wide fragment.
    return t.reshape(-1, V)


def _fake_vec(t, V, dyn):
    # 16-byte alignment: torch allocations are >=256 B aligned and V*itemsize == 128 bits, so the
    # wide load is safe. `dyn` leaves the vector COUNT symbolic, so one kernel serves every numel.
    lead = _L.sym() if dyn else t.numel() // V
    return _L.fake_compact(
        torch2cute[t.dtype], (lead, V), stride_order=(1, 0), align=16
    )


def _view_rowvec(t, rows, run, row_step, V):
    # A dense run per row, rows `row_step` apart: the innermost V is a contiguous vector, only the
    # row base carries the gap.
    return t.as_strided((rows, run // V, V), (row_step, V, 1))


def _fake_rowvec(t, rows, run, row_step, V):
    # Extents and pitch are STATIC here: the plan key carries (rows, run, row_step), so a different
    # gap geometry is a different kernel either way.
    return cute.runtime.make_fake_tensor(
        torch2cute[t.dtype],
        (rows, run // V, V),
        stride=(row_step, V, 1),
        assumed_align=16,
    )


def _fake_strided(t, shape):
    # The catch-all route BAKES each operand's layout, which is why the key carries them.
    v = t.expand(shape)
    return cute.runtime.make_fake_tensor(
        torch2cute[t.dtype], tuple(v.shape), stride=v.stride()
    )


def _rowvec_ok(inputs, shape, out_dtypes, compute_torch, V):
    # Row-vec path: every input is the same row-dense-gapped geometry, run % V == 0, and
    # (as for the flat vec path) every output is a vector-storable dtype. Returns the
    # shared (rows, run, row_step) or None. Output is allocated CONTIGUOUS (matches aten,
    # which drops the gap on the result), so only inputs need the gapped view.
    #
    # A NULLARY op has no inputs to carry a gap geometry, and this path's premise is
    # "gapped inputs, contiguous output" -- there is nothing to vectorize within. Decline
    # explicitly rather than relying on the empty-views fallthrough, which would also wrap
    # the output at assumed_align=16 that an unaligned target cannot honour.
    if not inputs:
        return None
    if any(d not in _VEC_OUT_DTYPES for d in out_dtypes):
        return None
    if any(tuple(t.shape) != tuple(shape) for t in inputs):
        return None  # no broadcast on this path (operands must share the exact shape)
    if not all(_is_16b_aligned(t) for t in inputs):
        return None  # rowvec also wraps at assumed_align=16 (see _is_16b_aligned)
    views = [_row_gap_view(t) for t in inputs]
    if any(v is None for v in views) or len(set(views)) != 1:
        return None  # all inputs must share ONE row-gapped geometry
    rows, run, row_step = views[0]
    return views[0] if run % V == 0 else None


class _Plan(NamedTuple):
    path: str  # "vec" | "rowvec" | "strided"
    shape: tuple  # broadcast output shape
    V: int  # vector width (vec / rowvec paths)
    out_dtypes: tuple  # per-output torch dtype (its length is the output count)
    fn: object  # the compiled kernel
    rowgap: tuple = ()  # (rows, run, row_step) for the rowvec path; () otherwise


def _build_plan(
    fn,
    op_name,
    nin,
    nout,
    consts,
    compute,
    compute_torch,
    inputs,
    out_dtypes,
    block,
    ept,
    vec_bits,
    load,
    pipe_depth,
    out_ref=None,
    with_index=False,
):
    # ALL shape-invariant work for this operand signature: broadcast shape, path
    # selection, op construction, and the kernel compile (against the live tensors as
    # seeds -- their layout matches every later call with this key). Run once per
    # key; the result is memoized so repeat calls only alloc + wrap + launch.
    # block/ept/vec_bits/load/pipe_depth are the exposed knobs, baked into the op +
    # compiled kernel. load/pipe_depth apply only to the vec route; the strided route
    # is always direct (broadcast/irregular layouts can't use the vec smem staging).
    # NULLARY ops (nin == 0: fill_ and the constructors built on it) have no input to
    # infer from -- the caller-supplied OUTPUT carries the shape, device and layout that
    # inputs normally provide. `ref` is that source of truth: operand 0 when there is
    # one, else the seed output. broadcast_shapes() of nothing is () (a 0-d scalar), so
    # the shape must come from ref for nin == 0.
    ref = inputs[0] if inputs else out_ref
    if ref is None:
        # nin == 0 (fill_ / the constructors) always passes the target: it is the only
        # source of shape, device and dtype. run() enforces this at the call site.
        raise AssertionError("nullary pointwise plan needs a target tensor")
    shape = (
        tuple(torch.broadcast_shapes(*(t.shape for t in inputs)))
        if inputs
        else tuple(ref.shape)
    )
    # Fill any knob the caller left None from the measured per-(numel, dtype) heuristic,
    # scaled to the device. numel is the flat element count -- pointwise's single
    # effective dimension.
    cfg = _choose_config(math.prod(shape), compute.width, _hw.caps(ref.device), nin)
    block = cfg.block if block is None else block
    ept = cfg.ept if ept is None else ept
    vec_bits = cfg.vec_bits if vec_bits is None else vec_bits
    load = cfg.load if load is None else load
    pipe_depth = cfg.pipe_depth if pipe_depth is None else pipe_depth
    out_types = [torch2cute[d] for d in out_dtypes]
    V = _vec_width(compute.width, vec_bits)
    # TMA tile-mode constraints (fall back to direct rather than emit a malformed
    # atom -> cudaErrorIllegalInstruction):
    #   - ept must be 1: TMA stages ONE (block, V) tile per CTA; the multi-tile
    #     (block*ept) staging is not wired. cpasync handles the ept>1 pipelined case.
    #   - the tile's innermost (contiguous) box dimension is V elements; TMA requires
    #     it be a multiple of 16 bytes, i.e. V*dtype a multiple of 128 bits. vec_bits
    #     < 128 (e.g. 64 -> V=2 fp32 -> 8-byte inner box) violates this.
    if load == "tma" and (ept != 1 or (V * compute.width) % 128 != 0):
        load = "direct"
    # cp.async per-copy width (CopyG2SOp num_bits_per_copy = V*dtype) is limited to
    # 32/64/128 bits by the hardware; a wider vec (e.g. 256b -> V=8 fp32) can't ride
    # one cp.async, so fall back to direct rather than fail IR verification.
    if load == "cpasync" and V * compute.width > 128:
        load = "direct"
    dev = ref.device
    # The strided route BAKES its operands' layouts, and for a nullary op the output is the
    # only operand -- so seed with the REAL target rather than a fresh contiguous tensor,
    # or a strided/transposed target compiles a contiguous kernel and from_dlpack rejects
    # it at launch ("Mismatched mOuts[0].strides[0]"). Ops WITH inputs keep the fresh seed:
    # their output is always contiguous (allocated here or cond-verified dense).
    seed_outs = (
        [out_ref]
        if not inputs and out_ref is not None
        else [torch.empty(shape, device=dev, dtype=d) for d in out_dtypes]
    )
    rowgap = ()
    in_dtypes = tuple(t.dtype for t in inputs)
    # The kernel-cache key holds ONLY what the compiled kernel BAKES (see _KERNELS).
    # Scalar values are runtime args on every path, so they never appear. The vec path
    # uses the dynamic-nvec wrap, so its key has no shape at all -- one kernel per
    # (fn, dtypes, knobs) serves every numel. rowvec/strided bake layouts (autovec_copy
    # needs static fragments; strided decodes a baked layout), so their keys carry the
    # geometry -- same per-layout compile count as before, but scalar-value recompiles
    # are gone there too.
    common = (
        id(fn),
        nin,
        nout,
        len(consts),
        compute,
        tuple(out_dtypes),
        in_dtypes,
        with_index,
    )
    # An index-consuming fn only works on the strided route (the vectorized routes give the
    # fn a whole V-wide fragment, not one element, so there is no single index to pass), so
    # skip the vec/rowvec tests entirely and let the else-branch below take it.
    vec1 = not with_index and _vec_ok(inputs, shape, out_dtypes, compute_torch, V, ref)
    if not can_use_32bit_index_math(
        (*inputs, *seed_outs), shape, block * ept * (V if vec1 else 1)
    ):
        raise ValueError("pointwise operands exceed signed 32-bit indexing")
    if vec1 and load != "tma":
        # tma is excluded from the dynamic wrap: make_tiled_tma_atom builds a
        # descriptor from the (static) tile extents; the rare tma plan falls to the
        # static branch below.
        op = _ElementwiseVec(
            fn,
            nin,
            nout,
            len(consts),
            compute,
            out_types,
            V,
            block,
            ept,
            load,
            pipe_depth,
            torch2cute[ref.dtype],
        )
        cin = [_fake_vec(t, V, True) for t in inputs]
        cout = [_fake_vec(o, V, True) for o in seed_outs]
        path = "vec"
        kkey = common + ("vec", V, block, ept, load, pipe_depth)
    elif vec1:
        op = _ElementwiseVec(
            fn,
            nin,
            nout,
            len(consts),
            compute,
            out_types,
            V,
            block,
            ept,
            load,
            pipe_depth,
            torch2cute[ref.dtype],
        )
        cin = [_fake_vec(t, V, False) for t in inputs]
        cout = [_fake_vec(o, V, False) for o in seed_outs]
        path = "vec_static"
        kkey = common + ("vec_static", shape, V, block, ept, load, pipe_depth)
    elif (
        not with_index
        and (rg := _rowvec_ok(inputs, shape, out_dtypes, compute_torch, V)) is not None
    ):
        # Row-dense-gapped inputs -> vectorize within rows (contiguous output). The
        # output is dense, so its run == its full row-length; only inputs carry the gap.
        rows, run, row_step = rg
        op = _ElementwiseRowVec(
            fn, nin, nout, len(consts), compute, out_types, V, run // V, block
        )
        cin = [_fake_rowvec(t, rows, run, row_step, V) for t in inputs]
        cout = [_fake_rowvec(o, rows, run, run, V) for o in seed_outs]
        path, rowgap = "rowvec", rg
        kkey = common + ("rowvec", rg, V, block)
    else:
        op = _ElementwiseStrided(
            fn, nin, nout, len(consts), compute, out_types, block, ept, with_index
        )
        cin = [_fake_strided(t, shape) for t in inputs]
        cout = [_fake_strided(o, tuple(o.shape)) for o in seed_outs]
        path = "strided"
        # This route BAKES each operand's layout, so every baked layout must be in the
        # kernel key -- including the OUTPUTS'. For ops with inputs the outputs are always
        # contiguous, so `shape` pins them; a nullary op's output is the only operand and
        # can be strided/transposed, so key it explicitly or a transposed target's kernel
        # gets handed a differently-strided one and from_dlpack rejects it at launch.
        lays = tuple((tuple(t.shape), t.stride()) for t in inputs)
        out_lays = tuple((tuple(o.shape), o.stride()) for o in seed_outs)
        kkey = common + ("strided", shape, lays, out_lays, block, ept)
    # Compile against PLACEHOLDER scalar values, not the live ones. The scalars are
    # genuine runtime arguments (they arrive as %arg f32/i32 in the IR), but the DSL
    # mangles every non-IR argument's VALUE into the generated MLIR symbol name, so
    # compiling with the live value both (a) makes the symbol vary with data that the
    # kernel does not bake and (b) emits an UNPARSABLE symbol once repr() switches to
    # exponent form: repr(1e16) == "1e+16", and the mangler's character filter does not
    # strip "+", yielding `..._Float321e+16_...` -> "expected '('". Seeding with a
    # canonical value keeps one symbol per kkey; the real values are passed at launch.
    seed_consts = [c.__class__(1) for c in consts]
    compiled = cached_plan(
        _KERNELS,
        kkey,
        lambda: _L.compile_kernel(op, cin, cout, seed_consts, _L.stream()),
        op=op_name,
    )
    return _Plan(path, shape, V, tuple(out_dtypes), compiled, rowgap)


def run(
    fn,
    key,
    nin,
    nout,
    consts,
    compute,
    compute_torch,
    inputs,
    out_dtypes,
    block=None,
    ept=None,
    vec_bits=None,
    load=None,
    pipe_depth=None,
    out=None,
    with_index=False,
):
    # inputs: torch tensors (any broadcastable shapes / strides). Returns nout torch
    # tensors of the broadcast shape. The plan (path / shape / compiled kernel) is
    # memoized per `key`; a cache hit does only the irreducible per-call work --
    # allocate outputs, wrap the live operands, launch. Any knob left None is filled by
    # _choose_config (the measured per-(numel, dtype) heuristic); the autotuner passes
    # explicit overrides (and includes them in `key`).
    # out: caller-provided output tensor(s) for the in-place / .out variants -- when
    # given, the kernel writes into these instead of allocating (the caller guarantees
    # they are the broadcast shape, right dtype, and contiguous; the override cond gates
    # that). None -> allocate fresh outputs (the functional path).
    if out is not None and nout == 1 and not with_index:
        knobs = (block, ept, vec_bits, load, pipe_depth)
        if all(value is None for value in knobs) and _try_run_tuned(
            fn, key, inputs, out[0], consts, compute
        ):
            return tuple(out)
    plan = cached_plan(
        _PLAN,
        key,
        lambda: _build_plan(
            fn,
            # key[0] is the aten op symbol (e.g. "add.Tensor"); the KERNEL cache under
            # _build_plan instruments with it, so a tlparse artifact fires only when a
            # genuinely new kernel is built (plan misses that reuse a kernel don't).
            f"aten::{key[0]}",
            nin,
            nout,
            consts,
            compute,
            compute_torch,
            inputs,
            out_dtypes,
            block,
            ept,
            vec_bits,
            load,
            pipe_depth,
            # NULLARY (nin == 0, e.g. fill_): with no input, the caller-supplied output is
            # the only source of shape/device/layout. Such a call always passes `out`.
            out[0] if not inputs and out else None,
            with_index,
        ),
    )
    if inputs:
        dev = inputs[0].device
    elif out:
        dev = out[0].device
    else:
        # nin == 0 (fill_ / the constructors): the target is the only source of device.
        raise AssertionError("nullary pointwise run needs a target tensor")
    outs = (
        list(out)
        if out is not None
        else [torch.empty(plan.shape, device=dev, dtype=d) for d in plan.out_dtypes]
    )
    if plan.path in ("vec", "vec_static"):
        cin = [_L.read_only(_view_vec(t, plan.V)) for t in inputs]
        cout = [_view_vec(o, plan.V) for o in outs]
    elif plan.path == "rowvec":
        rows, run, row_step = plan.rowgap
        cin = [
            _L.read_only(_view_rowvec(t, rows, run, row_step, plan.V)) for t in inputs
        ]
        # outputs are contiguous (row_step == run), whether freshly allocated or a
        # caller-provided .out/in-place tensor the cond verified dense.
        cout = [_view_rowvec(o, rows, run, run, plan.V) for o in outs]
    else:
        cin = [_L.read_only(t.expand(plan.shape)) for t in inputs]
        cout = list(outs)
    # Scalars are RUNTIME kernel args (never baked): the live per-call values go
    # straight to the launch, so a new alpha/lambd/... is never a recompile.
    plan.fn(cin, cout, list(consts), _L.stream())
    return tuple(outs)


_BLOCK_THREADS = 128
_ELEMENTS_PER_THREAD = 8
_TILES_PER_BLOCK = 6
_MIN_RUBIN_IO_BYTES = 16 * 1024 * 1024
_RUBIN_CC = (10, 7)
_CONST_SLOTS = 3
_TUNED_PLAN = {}
_TUNING_ENABLED = os.getenv("TORCH_NATIVE_CUTEDSL_POINTWISE_TUNING", "1") != "0"
_SUPPORTED_TUNED_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


class _TunedVectorConfig(NamedTuple):
    path: str
    vec: int
    elements_per_thread: int


class _TunedTmaConfig(NamedTuple):
    path: str
    transposed: tuple[bool, ...]
    box: int
    block: int
    input_views: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    output_view: tuple[tuple[int, ...], tuple[int, ...]]


class _TunedPlan(NamedTuple):
    config: _TunedVectorConfig | _TunedTmaConfig
    fn: object


def _pointer(t):
    with torch._C.DisableTorchFunctionSubclass():
        return t.const_data_ptr()


def _vector_config(inputs, shape, output_address, out_dtype=None):
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
    return _TunedVectorConfig("vector", vec, thread_work // vec)


def _tma_geometry(element_bytes):
    if element_bytes == 1:
        return 64, 512
    if element_bytes in (2, 4):
        return 32, 128
    if element_bytes == 8:
        return 16, 32
    return None


def _physical_dim_order(strides):
    if not strides or any(stride <= 0 for stride in strides):
        return None
    order = tuple(sorted(range(len(strides)), key=strides.__getitem__))
    if strides[order[0]] != 1:
        return None
    if any(strides[order[i]] <= strides[order[i - 1]] for i in range(1, len(order))):
        return None
    return order


def _dense_run(shape, strides, dims):
    return all(
        shape[dims[i - 1]] * strides[dims[i - 1]] == strides[dims[i]]
        for i in range(1, len(dims))
    )


def _contiguous_strides(shape):
    strides = [0] * len(shape)
    stride = 1
    for dim in range(len(shape) - 1, -1, -1):
        strides[dim] = stride
        stride *= shape[dim]
    return tuple(strides)


def _tma2d_config(inputs, shape, output_address):
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
                return _TunedTmaConfig(
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


def _select_config(inputs, shape, output_address, out_dtype, allow_tma=True):
    if allow_tma:
        config = _tma2d_config(inputs, shape, output_address)
        if config is not None:
            return config
    return _vector_config(inputs, shape, output_address, out_dtype)


def can_run_tuned(inputs, shape, out_dtype, target=None, allow_tma=True):
    if not _TUNING_ENABLED or not inputs:
        return False
    tensors = inputs
    if not tensors or out_dtype not in (*_SUPPORTED_TUNED_DTYPES, torch.bool):
        return False
    if not can_use_32bit_index_math(
        (*tensors, target) if target is not None else tensors, shape
    ):
        return False
    ref = tensors[0]
    if ref.device.type != "cuda" or _hw.caps(ref.device).cc != _RUBIN_CC:
        return False
    if ref.dtype not in _SUPPORTED_TUNED_DTYPES or out_dtype not in (
        ref.dtype,
        torch.bool,
    ):
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


def _tma_smem_layout(dtype, box, transposed):
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
    def __init__(self, fn, ninputs, nconsts, compute, out_type, in_dtype, config):
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
                    elif const_expr(self.in_dtype.width == 8):
                        fast = lane | ((wi & 1) << 5)
                        slow = wi >> 1
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


def _fake_tensor(dtype, view, align):
    shape, stride = view
    return cute.runtime.make_fake_tensor(
        dtype, tuple(shape), stride=tuple(stride), assumed_align=align
    )


def _compile_tuned_plan(fn, key, inputs, out, consts, compute):
    config = _select_config(inputs, tuple(out.shape), _pointer(out), out.dtype)
    if config is None:
        raise AssertionError("pointwise override selected an unsupported launch")
    in_type = torch2cute[inputs[0].dtype]
    out_type = cutlass.Int8 if out.dtype == torch.bool else torch2cute[out.dtype]
    if isinstance(config, _TunedVectorConfig):
        op = _ElementwiseVec(
            fn,
            len(inputs),
            1,
            len(consts),
            compute,
            (out_type,),
            config.vec,
            _BLOCK_THREADS,
            config.elements_per_thread,
            "direct",
            1,
            in_type,
        )
        fakes = [
            _L.fake_compact(
                in_type,
                (cute.sym_int(width=32), config.vec),
                stride_order=(1, 0),
                align=config.vec * inputs[0].element_size(),
            )
            for _ in inputs
        ]
        fake_out = _L.fake_compact(
            out_type,
            (cute.sym_int(width=32), config.vec),
            stride_order=(1, 0),
            align=config.vec * out.element_size(),
        )
        compile_args = (fakes, [fake_out], [compute(1) for _ in consts], _L.stream())
    else:
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
        "tuned",
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
        op=f"aten::{key[0]}",
    )
    return _TunedPlan(config, compiled)


def _try_run_tuned(fn, key, inputs, out, consts, compute):
    if len(consts) > _CONST_SLOTS or not can_run_tuned(
        inputs, out.shape, out.dtype, out
    ):
        return False
    caps = _hw.caps(inputs[0].device)
    plan_key = (
        key[0],
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
        _TUNED_PLAN,
        plan_key,
        lambda: _compile_tuned_plan(fn, key, inputs, out, consts, compute),
    )
    output = out.view(torch.int8) if out.dtype == torch.bool else out
    if isinstance(plan.config, _TunedVectorConfig):
        vec = plan.config.vec
        runtime_inputs = [_L.read_only(t.reshape(-1, vec)) for t in inputs]
        runtime_out = output.reshape(-1, vec)
        plan.fn(runtime_inputs, [runtime_out], list(consts), _L.stream())
    else:
        runtime_inputs = [
            _L.read_only(t.as_strided(*plan.config.input_views[k]))
            for k, t in enumerate(inputs)
        ]
        runtime_out = output.as_strided(*plan.config.output_view)
        runtime_consts = list(consts) + [compute(1)] * (_CONST_SLOTS - len(consts))
        plan.fn(runtime_inputs, runtime_out, *runtime_consts, _L.stream())
    return True

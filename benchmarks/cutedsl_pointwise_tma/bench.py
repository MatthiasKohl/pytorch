from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Callable

import cutlass
import cutlass.cute as cute
import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parent


def load_snapshot(variant: str = "baseline", stages: int = 6, box: int = 32) -> Any:
    import torch._native.ops.pointwise

    for name, filename in (("_layout", "_layout.py"), ("_perf_kernel", "kernel.py")):
        suffix = name if name == "_layout" else f"{name}_{variant}_{stages}_{box}"
        fullname = f"torch._native.ops.pointwise.{suffix}"
        source_path = ROOT / "published" / filename
        if name == "_perf_kernel" and (variant != "baseline" or box != 32):
            source = source_path.read_text()
            if variant == "linear":
                source = source.replace("self.transposed = config.transposed", "self.transposed = config.transposed\n        self.linear_stores = True not in config.transposed")
                source = source.replace("if const_expr(self.in_type.width == 32):", "if const_expr(self.linear_stores):\n                        fast = lane\n                        slow = wi\n                    elif const_expr(self.in_type.width == 32):")
            if variant == "vecstore":
                start = source.index("                for i in cutlass.range_constexpr(_ELEMENTS_PER_THREAD):")
                end = source.index("\n\n\nclass _ElementwiseScalarVector", start)
                source = source[:start] + (ROOT / "vector_epilogue.txt").read_text() + source[end:]
            if variant == "parallel":
                start = source.index("        if warp_idx == 0:\n")
                end = source.index("        cute.arch.barrier()\n", start)
                source = source[:start] + (ROOT / "parallel_issue.txt").read_text() + source[end:]
            if variant.startswith("rowmajor"):
                start = source.index("    inner_bytes = box * dtype.width // 8")
                source = source[:start] + "    if not transposed:\n        return cute.make_composed_layout(cute.make_swizzle(0, _SWIZZLE_BASE_LOG2, _SWIZZLE_SHIFT), 0, cute.make_layout((box, box, _TILES_PER_BLOCK), stride=(box, 1, box * box)))\n" + source[start:]
                if variant == "rowmajor_linear":
                    source = source.replace("self.transposed = config.transposed", "self.transposed = config.transposed\n        self.linear_stores = True not in config.transposed")
                    source = source.replace("if const_expr(self.in_type.width == 32):", "if const_expr(self.linear_stores):\n                        fast = lane\n                        slow = wi\n                    elif const_expr(self.in_type.width == 32):")
                if variant == "rowmajor_vecstore":
                    start = source.index("                for i in cutlass.range_constexpr(_ELEMENTS_PER_THREAD):")
                    end = source.index("\n\n\nclass _ElementwiseScalarVector", start)
                    source = source[:start] + (ROOT / "vector_epilogue.txt").read_text() + source[end:]
            if variant.startswith("rect"):
                columns = int(variant[4:])
                head, tail = source.split("class _ElementwiseScalarVector", 1)
                head = head.replace("(box, box, _TILES_PER_BLOCK)", f"(32, {columns}, _TILES_PER_BLOCK)")
                head = head.replace("(self.box, self.box)", f"(32, {columns})")
                head = head.replace("self.box * _TILES_PER_BLOCK", f"{columns} * _TILES_PER_BLOCK")
                head = head.replace("mOut.shape[2] // const_expr(self.box)", f"mOut.shape[2] // {columns}")
                head = head.replace("self.ntma * self.box * self.box", f"self.ntma * 32 * {columns}")
                head = head.replace("warp_iteration = (tidx // _WARP_SIZE) * _ELEMENTS_PER_THREAD", "warp_iteration = (tidx // _WARP_SIZE) * 8")
                head = head.replace("(fast_tile + stage) * const_expr(self.box)", f"(fast_tile + stage) * {columns}")
                head = head.replace("wi = warp_iteration + i", f"wi = warp_iteration + i // {columns // 32}")
                head = head.replace("fast = lane", f"fast = lane + (i % {columns // 32}) * _WARP_SIZE")
                source = head + "class _ElementwiseScalarVector" + tail
            if box != 32:
                source = source.replace("warp_iteration = (tidx // _WARP_SIZE) * _ELEMENTS_PER_THREAD", "warp_iteration = (tidx // _WARP_SIZE) * (self.box // 4)")
                source = source.replace("wi = warp_iteration + i", "wi = warp_iteration + i // (self.box // _WARP_SIZE)")
                source = source.replace("fast = lane", "fast = lane + (i % (self.box // _WARP_SIZE)) * _WARP_SIZE")
            source_path = ROOT / f"kernel_{variant}_s{stages}_box{box}.py"
            source_path.write_text(source)
        spec = importlib.util.spec_from_file_location(fullname, source_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = module
        spec.loader.exec_module(module)
        module._TILES_PER_BLOCK = stages
        module._TMA_BOX_SIDE = box
        module._ELEMENTS_PER_THREAD = box * box // 128
        if name == "_perf_kernel" and variant.startswith("rect"):
            module._ELEMENTS_PER_THREAD = 32 * int(variant[4:]) // 128
    return module


@cute.jit
def identity(x: Any) -> Any:
    return x


@cute.jit
def plus_one(x: Any) -> Any:
    return x + cutlass.Float32(1)


@cute.jit
def add_two(x: Any, y: Any) -> Any:
    return x + y


@triton.jit
def non_tma_matrix(
    X, Y, O,
    N: tl.constexpr, SX0: tl.constexpr, SX1: tl.constexpr,
    SY0: tl.constexpr, SY1: tl.constexpr, SO0: tl.constexpr, SO1: tl.constexpr,
    OP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
):
    i = tl.program_id(0) * BM + tl.arange(0, BM)
    j = tl.program_id(1) * BN + tl.arange(0, BN)
    value = tl.load(X + i[:, None] * SX0 + j[None, :] * SX1)
    if OP == 1:
        value += 1.0
    elif OP == 2:
        value += tl.load(Y + i[:, None] * SY0 + j[None, :] * SY1)
    elif OP == 3:
        value += tl.load(Y)
    tl.store(O + i[:, None] * SO0 + j[None, :] * SO1, value)


@triton.jit
def non_tma_flat(X, Y, O, N: tl.constexpr, OP: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + i, i < N)
    if OP == 1:
        value += 1.0
    elif OP == 2:
        value += tl.load(Y + i, i < N)
    elif OP == 3:
        value += tl.load(Y)
    tl.store(O + i, value, i < N)


def make_case(name: str, n: int, count: int) -> list[tuple[list[torch.Tensor], torch.Tensor]]:
    results = []
    for _ in range(count):
        x = torch.randn(n, n, device="cuda", dtype=torch.float32)
        if name == "transpose_copy":
            ins, out = [x], torch.empty_strided((n, n), (1, n), device="cuda")
        elif name == "transpose_add_one":
            ins, out = [x], torch.empty_strided((n, n), (1, n), device="cuda")
        elif name == "dense_add_host":
            ins, out = [x], torch.empty_like(x)
        elif name == "dense_add_cuda":
            ins, out = [x, torch.tensor(1.0, device="cuda")], torch.empty_like(x)
        elif name == "dense_add_tensor":
            ins, out = [x, torch.randn_like(x)], torch.empty_like(x)
        elif name == "same_transposed_add":
            ins, out = [x.t(), torch.randn_like(x).t()], torch.empty_like(x.t())
        elif name == "transposed_add_cuda":
            ins, out = [x.t(), torch.tensor(1.0, device="cuda")], torch.empty_like(x.t())
        elif name == "mixed_add":
            ins, out = [x, torch.randn_like(x).t()], torch.empty_like(x)
        elif name == "mixed_add_transposed_out":
            ins, out = [x.t(), torch.randn_like(x)], torch.empty_like(x.t())
        else:
            raise ValueError(name)
        results.append((ins, out))
    return results


def operation(name: str) -> tuple[int, Callable[..., Any]]:
    if name == "transpose_copy":
        return 0, identity
    if name in ("dense_add_host", "transpose_add_one"):
        return 1, plus_one
    if name in ("dense_add_cuda", "transposed_add_cuda"):
        return 3, add_two
    return 2, add_two


def make_tma(kernel: Any, name: str, ins: list[torch.Tensor], out: torch.Tensor) -> Callable[[], None]:
    # Force dense TMA only for diagnosis; normal dense add dispatch stays on ATen.
    from torch._native.ops.pointwise._layout import _OperandLayout, _tma2d_config
    from torch._native.cutedsl import launch

    layout = tuple(
        _OperandLayout(tuple(t.shape) if t.numel() != 1 else (),
                       t.stride() if t.numel() != 1 else (), t.element_size(),
                       t.data_ptr() % 128, t.is_contiguous())
        for t in ins
    )
    cfg = _tma2d_config(layout, tuple(ins[0].shape), 256 * 1024, out.stride())
    if cfg is None:
        raise ValueError(f"no TMA config for {name}")
    _, fn = operation(name)
    plan = kernel._compile_plan(fn, name, ins, out, (), cutlass.Float32, cfg)
    runtime_ins = [launch.read_only(t.as_strided(*view)) for t, view in zip(ins, cfg.input_views)]
    runtime_out = out.as_strided(*cfg.output_view)
    return lambda: plan.fn(runtime_ins, runtime_out, cutlass.Float32(1), cutlass.Float32(1), cutlass.Float32(1), launch.stream())


def make_aten(name: str, ins: list[torch.Tensor], out: torch.Tensor) -> Callable[[], None]:
    if name == "transpose_copy":
        return lambda: out.copy_(ins[0])
    if name in ("dense_add_host", "transpose_add_one"):
        return lambda: torch.add(ins[0], 1.0, out=out)
    return lambda: torch.add(ins[0], ins[1], out=out)


def make_cute_transpose(name: str, ins: list[torch.Tensor], out: torch.Tensor, tile: int, threads: int, asynchronous: bool, classic: bool = False) -> Callable[[], None]:
    from cute_transpose import ClassicTranspose, Transpose
    from torch._native.cutedsl import launch

    cls = ClassicTranspose if classic else Transpose
    op = cls(ins[0].size(0), tile, threads, asynchronous, name == "transpose_add_one")
    fake = launch.fake_compact(cutlass.Float32, (ins[0].numel(),), align=16)
    compiled = launch.compile_kernel(op, fake, fake, launch.stream())
    x = ins[0].view(-1)
    target = out.as_strided((out.numel(),), (1,))
    return lambda: compiled(x, target, launch.stream())


def make_cute_vector(kernel: Any, name: str, ins: list[torch.Tensor], out: torch.Tensor) -> Callable[[], None]:
    from torch._native.ops.pointwise._layout import _TmaConfig
    from torch._native.cutedsl import launch

    n = out.numel()
    view = ((n,), (1,))
    count = sum(t.numel() != 1 for t in ins)
    ept = 32 // count
    cfg = _TmaConfig(tuple(False if t.numel() != 1 else None for t in ins), 8, 128,
                     tuple(view if t.numel() != 1 else ((1,), (1,)) for t in ins), view, ept)
    _, fn = operation(name)
    plan = kernel._compile_plan(fn, name, ins, out, (), cutlass.Float32, cfg)
    inputs = [launch.read_only(t.as_strided(*v)) for t, v in zip(ins, cfg.input_views)]
    target = out.as_strided((out.numel(),), (1,))
    return lambda: plan.fn(inputs, target, cutlass.Float32(1), cutlass.Float32(1), cutlass.Float32(1), launch.stream())


def make_triton(name: str, ins: list[torch.Tensor], out: torch.Tensor, bm: int, bn: int, warps: int) -> Callable[[], None]:
    op, _ = operation(name)
    x = ins[0]
    y = ins[1] if len(ins) > 1 else ins[0]
    if (x.is_contiguous() and out.is_contiguous() and (y.numel() == 1 or y.is_contiguous())) or name in ("same_transposed_add", "transposed_add_cuda"):
        block = bm * bn
        return lambda: non_tma_flat[(triton.cdiv(x.numel(), block),)](x, y, out, x.numel(), op, block, num_warps=warps)
    sy = y.stride() if y.ndim == 2 else (0, 0)
    return lambda: non_tma_matrix[(triton.cdiv(x.size(0), bm), triton.cdiv(x.size(1), bn))](
        x, y, out, x.size(0), *x.stride(), *sy, *out.stride(), op, bm, bn, num_warps=warps)


def graph_for(calls: list[Callable[[], None]], repeat: int) -> torch.cuda.CUDAGraph:
    for call in calls:
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(repeat):
            calls[i % len(calls)]()
    return graph


def timing(graph: torch.cuda.CUDAGraph, repeat: int, trials: int = 9) -> dict[str, float]:
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    values = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(trials):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000 / repeat)
    return {"us": statistics.median(values), "min_us": min(values), "max_us": max(values)}


def check(name: str, ins: list[torch.Tensor], out: torch.Tensor) -> None:
    if name == "transpose_copy":
        expected = ins[0]
    elif name in ("dense_add_host", "transpose_add_one"):
        expected = ins[0] + 1
    else:
        expected = ins[0] + ins[1]
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


CASES = ("transpose_copy", "dense_add_host", "dense_add_cuda", "dense_add_tensor", "mixed_add", "mixed_add_transposed_out", "transpose_add_one", "same_transposed_add", "transposed_add_cuda")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--buffers", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=64)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--profile", choices=("tma", "aten", "triton", "cute_ldg", "cute_ldgsts", "cute_vector"))
    parser.add_argument("--config", type=int, nargs=3, default=(32, 128, 4))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--variant", default="baseline", choices=("baseline", "linear", "vecstore", "parallel", "rect64", "rect128", "rect256", "rowmajor", "rowmajor_linear", "rowmajor_vecstore"))
    parser.add_argument("--stages", type=int, default=6)
    parser.add_argument("--box", type=int, default=32)
    parser.add_argument("--cute-tile", type=int, default=64)
    parser.add_argument("--cute-threads", type=int, default=128)
    parser.add_argument("--cute-classic", action="store_true")
    parser.add_argument("--implementations", nargs="+", default=("tma", "aten", "triton"))
    args = parser.parse_args()
    torch.backends.python_native.cutedsl.disable()
    from torch import _native
    _native.set_aot_enabled(False)
    torch.manual_seed(123)
    kernel = load_snapshot(args.variant, args.stages, args.box)
    rows = []
    for name in args.cases:
        buffers = make_case(name, args.n, args.buffers)
        configs = [tuple(args.config)]
        if args.tune:
            configs = [(m, n, w) for m, n in ((32,32),(32,64),(32,128),(32,256),(64,32),(64,64),(64,128),(128,32),(128,64),(128,128)) for w in (4,8)]
        tuned = []
        for cfg in configs:
            if args.profile and args.profile != "triton":
                tuned.append((0.0, cfg))
                continue
            calls = [make_triton(name, ins, out, *cfg) for ins, out in buffers]
            graph = graph_for(calls, args.repeat)
            check(name, *buffers[0])
            result = timing(graph, args.repeat, 5 if args.tune else 9)
            tuned.append((result["us"], cfg))
        _, config = min(tuned)
        factories = {
            "tma": lambda ins, out: make_tma(kernel, name, ins, out),
            "aten": lambda ins, out: make_aten(name, ins, out),
            "triton": lambda ins, out: make_triton(name, ins, out, *config),
            "cute_ldg": lambda ins, out: make_cute_transpose(name, ins, out, args.cute_tile, args.cute_threads, False, args.cute_classic),
            "cute_ldgsts": lambda ins, out: make_cute_transpose(name, ins, out, args.cute_tile, args.cute_threads, True, args.cute_classic),
            "cute_vector": lambda ins, out: make_cute_vector(kernel, name, ins, out),
        }
        if args.profile:
            call = factories[args.profile](*buffers[0])
            for _ in range(3):
                call()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            call()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            return
        for impl in args.implementations:
            factory = factories[impl]
            calls = [factory(ins, out) for ins, out in buffers]
            graph = graph_for(calls, args.repeat)
            check(name, *buffers[0])
            result = timing(graph, args.repeat)
            traffic = buffers[0][0][0].numel() * 4 * (3 if name in ("dense_add_tensor", "mixed_add", "mixed_add_transposed_out") else 2)
            row = {"case": name, "impl": impl, "variant": args.variant, "stages": args.stages, "box": args.box, "n": args.n, "buffers": args.buffers, "config": config if impl == "triton" else None, **result, "TB_s": traffic / result["us"] / 1e6}
            rows.append(row)
            print(json.dumps(row), flush=True)
        if args.output:
            args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()

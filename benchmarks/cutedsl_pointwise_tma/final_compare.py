from __future__ import annotations

import json
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Callable

import torch

import bench


def main() -> None:
    buffers_count = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    repeat = 64
    root = Path(__file__).resolve().parent
    torch.backends.python_native.cutedsl.disable()
    from torch import _native
    _native.set_aot_enabled(False)
    torch.manual_seed(123)
    baseline = bench.load_snapshot()
    shared_fix = bench.load_snapshot("rowmajor", 2)
    optimized = bench.load_snapshot("rowmajor_vecstore", 2)
    copy_optimized = bench.load_snapshot("parallel", 4)
    # Each module's config type is metadata-compatible with the shared selector.
    cases = ("transpose_copy", "dense_add_host", "dense_add_cuda", "dense_add_tensor", "same_transposed_add", "transposed_add_cuda", "mixed_add", "mixed_add_transposed_out")
    rows = []
    for case in cases:
        buffers = bench.make_case(case, 4096, buffers_count)
        factories: dict[str, Callable[..., Any]] = {
            "original_tma": lambda ins, out: bench.make_tma(baseline, case, ins, out),
            "shared_fix_tma": lambda ins, out: bench.make_tma(shared_fix, case, ins, out),
            "optimized_tma": lambda ins, out: bench.make_tma(copy_optimized if case == "transpose_copy" else optimized, case, ins, out),
            "aten": lambda ins, out: bench.make_aten(case, ins, out),
            "non_tma_triton": lambda ins, out: bench.make_triton(case, ins, out, *((32, 64, 4) if case == "transpose_copy" else (64, 64, 4) if case in ("mixed_add", "mixed_add_transposed_out") else (128, 32, 8))),
        }
        if case == "transpose_copy":
            factories["non_tma_cute_ldg"] = lambda ins, out: bench.make_cute_transpose(case, ins, out, 64, 128, False, True)
            factories["non_tma_cute_ldgsts"] = lambda ins, out: bench.make_cute_transpose(case, ins, out, 128, 128, True, True)
        elif case not in ("mixed_add", "mixed_add_transposed_out"):
            factories["non_tma_cute_vector"] = lambda ins, out: bench.make_cute_vector(baseline, case, ins, out)
        graphs = {}
        for name, factory in factories.items():
            calls = [factory(ins, out) for ins, out in buffers]
            graph = bench.graph_for(calls, repeat)
            # Poison outputs before replay to reject empty captures or stale results.
            for _, out in buffers:
                out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            for ins, out in buffers:
                bench.check(case, ins, out)
            graphs[name] = graph
        samples: dict[str, list[float]] = {name: [] for name in graphs}
        order = list(graphs)
        rng = random.Random(42)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for trial in range(25):
            rng.shuffle(order)
            for name in order:
                graph = graphs[name]
                for _ in range(3):
                    graph.replay()
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) * 1000 / repeat)
        for name, values in samples.items():
            values.sort()
            row = {"case": case, "impl": name, "buffers": buffers_count, "us": statistics.median(values), "p10_us": values[2], "p90_us": values[22], "min_us": values[0], "max_us": values[-1], "samples": values}
            rows.append(row)
            print(json.dumps({k:v for k,v in row.items() if k != "samples"}), flush=True)
        (root / f"final-buffers{buffers_count}.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()

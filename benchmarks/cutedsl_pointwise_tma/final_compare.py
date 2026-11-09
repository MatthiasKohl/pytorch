from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import random
import statistics
import subprocess
from typing import Any, Callable

import torch

import bench


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("buffers", type=int, nargs="?", default=4)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--localized", action="store_true")
    parser.add_argument("--expected-memory-clock", type=int)
    parser.add_argument("--repeat", type=int, default=64)
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--cases", nargs="+", choices=bench.CASES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--candidates", type=Path, help="Per-case tuning candidates; uses the same timing and correctness checks")
    args = parser.parse_args()
    candidates = json.loads(args.candidates.read_text()) if args.candidates else None
    if args.n < 512 or args.n % 512 != 0 or args.buffers < 1 or args.repeat < args.buffers or args.trials < 3:
        parser.error("n must be a multiple of 512; buffers, repeat and trials must cover all buffers and samples")
    buffers_count, repeat = args.buffers, args.repeat
    root = Path(__file__).resolve().parent
    suffix = (f"-n{args.n}" if args.n != 4096 else "") + ("-localized" if args.localized else "")
    output = args.output or bench.OUTPUT_ROOT / f"final-buffers{buffers_count}{suffix}.json"
    if output.resolve().is_relative_to(root):
        parser.error("Write raw benchmark results outside the benchmark package")
    output.parent.mkdir(parents=True, exist_ok=True)

    def memory_clock() -> int:
        value = int(subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=clocks.current.memory", "--format=csv,noheader,nounits"], text=True).strip())
        if args.expected_memory_clock is not None and value != args.expected_memory_clock:
            raise RuntimeError(f"Expected memory clock {args.expected_memory_clock} MHz, observed {value}")
        return value

    clock_before = memory_clock()
    torch.backends.python_native.cutedsl.disable()
    from torch import _native
    _native.set_aot_enabled(False)
    torch.cuda.set_device(0)
    torch.manual_seed(123)
    contexts, pools, streams = (), [], []
    if args.localized:
        from torch.cuda.green_contexts import GreenContext, get_num_locality_domains

        if get_num_locality_domains(0) != 2:
            raise RuntimeError("The localized benchmark requires two locality domains")
        contexts = GreenContext.split(locality_domain_ids=(0, 1), device_id=0)
        pools = [torch.cuda.MemPool(torch.cuda.memory.LocalizedAllocator(domain, device=0).allocator()) for domain in range(2)]
        streams = [context.Stream() for context in contexts]
        if [context.locality_domain_id for context in contexts] != [0, 1]:
            raise RuntimeError("Green-context domain assignment differs from the requested split")
    props = torch.cuda.get_device_properties(0)
    metadata = {
        "device": props.name, "cc": [props.major, props.minor], "device_sms": props.multi_processor_count,
        "n": args.n, "input_bytes": args.n * args.n * 4, "buffers": buffers_count,
        "localized": args.localized, "green_context_sms": [context.sm_count for context in contexts],
        "memory_mhz_before": clock_before, "repeat": repeat, "trials": args.trials,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "cutlass": importlib.metadata.version("nvidia-cutlass-dsl"), "triton": importlib.metadata.version("triton"),
        "candidates": candidates,
    }
    print(json.dumps({"metadata": metadata}), flush=True)
    baseline = bench.load_snapshot()
    shared_fix = bench.load_snapshot("rowmajor", 2)
    optimized = bench.load_snapshot("rowmajor_vecstore", 2)
    copy_optimized = bench.load_snapshot("parallel", 4)
    # Each module's config type is metadata-compatible with the shared selector.
    cases = args.cases or ("transpose_copy", "dense_add_host", "dense_add_cuda", "dense_add_tensor", "same_transposed_add", "transposed_add_cuda", "mixed_add", "mixed_add_transposed_out")
    rows = []
    for case in cases:
        memory_clock()
        if args.localized:
            from cuda.bindings import runtime
            from torch.cuda._utils import _check_cuda_bindings

            # Allocate each rectangular half in its matching domain's pool.
            groups = []
            for _ in range(buffers_count):
                group = []
                for domain, pool in enumerate(pools):
                    with torch.cuda.use_mem_pool(pool, device=0):
                        ins, out = bench.make_case(case, args.n, 1, (args.n // 2, args.n))[0]
                    for tensor in (*ins, out):
                        attrs = _check_cuda_bindings(runtime.cudaPointerGetAttributes(tensor.data_ptr()))
                        if int(attrs.localityDomainOrdinal) != domain:
                            raise RuntimeError("Tensor allocation is on the wrong locality domain")
                    group.append((ins, out))
                groups.append(group)
        else:
            groups = [[pair] for pair in bench.make_case(case, args.n, buffers_count)]
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
        if candidates is not None:
            factories = {}
            for candidate in candidates[case]:
                kind, params = candidate["kind"], candidate.get("params", [])
                name = candidate["name"]
                if kind == "tma":
                    module = bench.load_snapshot(*params)
                    factories[name] = lambda ins, out, module=module: bench.make_tma(module, case, ins, out)
                elif kind == "vector":
                    factories[name] = lambda ins, out, params=params: bench.make_cute_vector(baseline, case, ins, out, *params)
                elif kind == "transpose":
                    factories[name] = lambda ins, out, params=params: bench.make_cute_transpose(case, ins, out, *params)
                elif kind == "triton":
                    factories[name] = lambda ins, out, params=params: bench.make_triton(case, ins, out, *params)
                elif kind == "aten":
                    factories[name] = lambda ins, out: bench.make_aten(case, ins, out)
                else:
                    raise ValueError(f"Unknown candidate kind: {kind}")
        graphs = {}
        keepalive = {}
        for name, factory in factories.items():
            print(json.dumps({"prepare": name, "case": case}), flush=True)
            group_calls = [[factory(ins, out) for ins, out in group] for group in groups]
            if args.localized:
                # execute_on_streams forks and joins both halves into the caller.
                calls = [lambda work=work: torch.cuda.execute_on_streams(streams, lambda fn: fn(), work) for work in group_calls]
            else:
                calls = [work[0] for work in group_calls]
            keepalive[name] = calls
            graph = bench.graph_for(calls, repeat)
            # Poison outputs before replay to reject empty captures or stale results.
            for group in groups:
                for _, out in group:
                    out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            for group in groups:
                for ins, out in group:
                    bench.check(case, ins, out)
            graphs[name] = graph
        samples: dict[str, list[float]] = {name: [] for name in graphs}
        order = list(graphs)
        rng = random.Random(42)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for trial in range(args.trials):
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
            row = {"case": case, "impl": name, "n": args.n, "input_bytes": args.n * args.n * 4, "localized": args.localized, "buffers": buffers_count, "us": statistics.median(values), "p10_us": values[round((len(values) - 1) * 0.1)], "p90_us": values[round((len(values) - 1) * 0.9)], "min_us": values[0], "max_us": values[-1], "samples": values}
            rows.append(row)
            print(json.dumps({k:v for k,v in row.items() if k != "samples"}), flush=True)
        output.write_text(json.dumps(rows, indent=2) + "\n")
        metadata["memory_mhz_after"] = memory_clock()
        output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()

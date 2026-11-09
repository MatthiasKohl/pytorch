# Pointwise benchmark hand-off

GB200 measurements are complete. See
[the GB200/Rubin comparison](reference/blackwell/RESULTS.md) for the current
recommendation, minima at both footprints, controls and limitations. The
[Rubin reference](reference/rubin/footprints-4752.md) retains its original
median-based report; the new comparison uses timing minima as requested.

Use non-TMA vectors for dense adds. For transpose copy, the practical GB200
TMA choice is a 64x64 tile, one pipeline stage and 256 threads:
`bench.load_snapshot("baseline", 1, 64, 256)`. It improves the original TMA at
both sizes, beats the tested non-TMA reference at 64 MiB and is effectively
tied at 1 GiB. The 32x256 rectangular experiment is slightly faster at 64 MiB
but slower at 1 GiB. All intermediate configurations and timings are preserved.

Rubin's measured 32x32/six-stage/128-thread TMA remains a good copy choice.
The four-tile distributed-producer version is slightly better at 1 GiB;
reconstruct it with `bench.load_snapshot("parallel", 4)` and
`parallel_issue.txt`. The new GB200 geometry has not been measured on Rubin.
There is no validated single geometry that is best across both architectures.
The immediate remaining experiment is to rerun the new finalists on Rubin.

This is a transfer-only benchmark package. No optimization has been applied
to `torch/_native`, and no feature PR has been updated. Copy and contiguous-add
TMA benchmarks force the executor; they do not describe normal dispatch.

## Environment and reproduction

Use this feature checkout with a working editable PyTorch build, CuTeDSL,
Triton and CUDA bindings. Nsight Compute is optional for diagnostics. The
references use CuTeDSL 4.8.0 and Triton 3.8.0. Inspect setup scripts before
sourcing them: they may install packages or rebuild PyTorch. No build or
installation was needed for the GB200 runs.

Activate the environment, or set `POINTWISE_PERF_VENV` to its location. Set
`POINTWISE_PERF_CACHE_ROOT` to a permitted temporary cache directory, then run
from the repository root:

```bash
source benchmarks/cutedsl_pointwise_tma/env.sh
python benchmarks/cutedsl_pointwise_tma/final_compare.py 4 --n 4096 --localized --cases transpose_copy dense_add_host dense_add_cuda dense_add_tensor mixed_add --candidates benchmarks/cutedsl_pointwise_tma/reference/blackwell/finalists-candidates.json
python benchmarks/cutedsl_pointwise_tma/final_compare.py 4 --n 16384 --localized --cases transpose_copy dense_add_host dense_add_cuda dense_add_tensor mixed_add --candidates benchmarks/cutedsl_pointwise_tma/reference/blackwell/finalists-candidates.json
```

GB200 memory clocks were left unchanged; 3996 MHz was observed. No lock is
required for this setup. For the Rubin reference, explicitly lock memory
clocks to 4752,4752 MHz and pass `--expected-memory-clock 4752`. Do not assume
Rubin's clock setting is supported on Blackwell. SM clocks were not modified.
Only device index 0 is used.

4096x4096 FP32 is 64 MiB per matrix; 16384x16384 is 1 GiB. The localized mode
allocates two rectangular halves in separate domain-local pools and executes
them on matching green-context streams. Each timed operation includes both
halves; a 1 GiB input has two 512 MiB halves. Allocation domains are checked.
The streams fork and join inside graph capture, and tensors and contexts stay
alive until work completes. The localized partition may expose fewer SMs
than an unpartitioned run.

Final runs use four rotating buffer sets, 64 operations per graph, and 25
interleaved samples with output-poisoning and exact correctness checks. Use
the minimum graph-averaged latency for the peak comparison; medians and ranges
are saved too. For matched unpartitioned controls, omit `--localized` and use
`control-candidates.json`. Positional `1` selects a fixed-buffer experiment.

## Preserved experiments

`reference/blackwell/results.json` contains 18 complete runs, 1,034 timing rows,
and every candidate definition. It includes the fixed Rubin finalists,
pipeline/shared-layout sweeps, vector tuning, LDG/LDGSTS copy comparisons,
93 copy-specific geometry/traversal configurations, full confirmation and
unpartitioned controls. To repeat a screening set, copy its run's
`metadata.candidates` object to a temporary JSON and pass it to `--candidates`.

`load_snapshot` accepts `(variant, stages, box, threads, order)`. The shared
selector derives tensor views; `make_tma` explicitly applies the experimental
box and thread count. Larger boxes adjust the lane mapping accordingly.
`make_cute_vector` accepts `(kernel, case, inputs, output, vec, ept, block)`.
Without extra arguments, both helpers preserve the old fixed configurations.

The frozen executor is in `published/`, from public commit
`57476b7efc075dbf974e37876eb749f0020e8b81`. Do not format or edit those files:
variant generation uses exact source-text replacements. `bench.py`,
`parallel_issue.txt` and `vector_epilogue.txt` reconstruct the prototypes.
The old `minimal-fp32-tma.patch` remains unapplied; it is not the selected tuning.

`bench.py --profile` supports `--box`, `--tma-threads` and `--order` for copy
diagnostics. `profile_suite.py` is still an unpartitioned single-kernel tool,
not a localized pair benchmark. Use Nsight's `--clock-control none` and keep
profiling replay times separate from latency comparisons.

Raw results, generated kernels and profiler output go to
`POINTWISE_PERF_OUTPUT_DIR`, outside this package. Shared references contain
curated timings only. Do not transfer raw logs, binary profiles, internal
profiler summaries, machine identifiers or private build metadata. The
original benchmark commit must be cleaned before public publication; deleting
files in a later commit does not remove earlier copies from history.

Keep benchmark files separate from the feature PR stack. No commits or PR
updates were made during the GB200 experiments.

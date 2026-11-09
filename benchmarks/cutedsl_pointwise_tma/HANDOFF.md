# Blackwell hand-off

Compare FP32 transpose copy and dense add on Blackwell, using GPU index 0 only.
Run both 4096x4096 (64 MiB per matrix) and 16384x16384 (1 GiB per matrix), with
explicit memory-clock control and localization. This is a transfer-only
benchmark package; no optimization has been applied to `torch/_native`.

Use this same feature checkout with a working editable PyTorch build, CuTeDSL,
Triton, CUDA bindings, and Nsight Compute (`ncu`) for profiling. The reference
uses CuTeDSL 4.8.0 and Triton 3.8.0. Inspect the environment before building or
installing anything. Read setup scripts before sourcing them: they may also
install packages or rebuild PyTorch.

Activate the existing venv, or set `POINTWISE_PERF_VENV` to its location. Set
`POINTWISE_PERF_CACHE_ROOT` to a temporary cache directory permitted by the
local instructions. From the repository root, lock the assigned GPU's memory
clocks before running. The Rubin reference uses 4752,4752 MHz. Query Blackwell's
supported clocks and select its supported target instead of assuming 4752 MHz
is valid there; pass that target to every command's clock check:

```bash
source benchmarks/cutedsl_pointwise_tma/env.sh
nvidia-smi -i 0 --query-supported-clocks=memory --format=csv
export POINTWISE_MEMORY_MHZ=4752  # Rubin; use the supported Blackwell target.
sudo -n nvidia-smi -i 0 --lock-memory-clocks="$POINTWISE_MEMORY_MHZ,$POINTWISE_MEMORY_MHZ"
python benchmarks/cutedsl_pointwise_tma/final_compare.py 4 --n 4096 --localized --expected-memory-clock "$POINTWISE_MEMORY_MHZ"
python benchmarks/cutedsl_pointwise_tma/final_compare.py 4 --n 16384 --localized --expected-memory-clock "$POINTWISE_MEMORY_MHZ"
```

The localized mode allocates two half-sized problems in separate domain-local
memory pools and runs each on its matching green-context stream. The two halves
together contain the stated matrix size; a 1 GiB input has two 512 MiB halves.
Allocation domains are checked. Timing forks and joins both streams, with all
tensors and contexts retained until work completes. The SM resources exposed
by the localized partition may differ from the unpartitioned control.

These commands validate outputs after poisoning them, and compare original TMA,
the shared-layout/pipeline fix, the vector-store prototype, ATen, and non-TMA
kernels. They record clock checks, device/software metadata, and 25 interleaved
samples of 64-call CUDA graphs over four buffer sets. Raw results, generated
kernels and profiler output go to `POINTWISE_PERF_OUTPUT_DIR`, which defaults
under the configured cache root, outside this benchmark directory. Use
`--output` to select another external path. Repeat without `--localized`
for the unpartitioned control, and use positional `1` for a fixed-buffer check.
Compare with the new `reference/rubin/footprints-4752.json` and its companion
report. The older `final-buffers*.json` were not localized or explicitly
clock-locked and are historical only. The frozen baseline is
in `published/`; do not format or edit it because variant generation uses exact
source-text replacements. It comes from submitted commit
`57476b7efc075dbf974e37876eb749f0020e8b81`. Shared native utilities are imported
from this checkout (Rubin base: `9712d89f589e2bab95c8dd9cadf0ced95d5a0232`).

Dense TMA has two-way shared-memory bank conflicts; a row-major shared tile
removes them. Pipeline depth and vector stores need separate evaluation at both
footprints. In particular, do not assume the globally shallower two-tile
pipeline is suitable for transpose copy, or that non-TMA always wins once
localization is enabled. See the new report for the measured results.

New localized 1 GiB medians (microseconds): transpose is 139.42 for original
TMA versus 146.72 for the existing non-TMA reference; the global two-tile change
regresses it to 159.02. Host-scalar add improves from 218.65 to 143.14 with the
combined prototype, versus 122.79 for the CuTe vector path. Tensor add improves
from 360.48 to 190.75, versus 182.05 for the vector path. The new comparisons
cover both footprints, use checked 4752 MHz clocks, and include a fresh-process
1 GiB confirmation. Non-TMA configurations are fixed finalists, not exhaustive
retuning for localized halves.

First reproduce these fixed configurations, then retune for Blackwell using
`sweep.py` modes `tma`, `rowmajor`, `classic`, `cute`, `parallel`, and `rect`,
and `bench.py --tune`. The CuTe vector baseline hardcodes Rubin's vec=8 and
EPT=32 (unary)/16 (binary); also test Blackwell-appropriate vector/EPT choices.
`final_compare.py` uses fixed finalists, so retuned winners need a separate run.

The four-tile/distributed-producer copy is preserved by
`bench.load_snapshot("parallel", 4)` and `parallel_issue.txt`; it is the
`optimized_tma` entry for transpose copy. Other variants are reconstructed by
`bench.py` and `sweep.py`. Candidate timings are retained in the reference JSON.
Not every raw tuning log from the original node survived the transfer.

Keep identical output strides, GPU-only graph timing, and output-poisoning
checks. Copy and contiguous-add TMA measurements force the executor; they do
not establish that normal dispatch selects TMA. Profile shared excess
wavefronts, instruction/store counts, and barrier stalls separately from timing.
`profile_suite.py` remains an unlocalized single-kernel diagnostic; it does not
measure the localized pair. Preserve memory-clock locks using Nsight's
`--clock-control none`, and do not use counter-replay times as pair latency.
The combined vector epilogue is FP32-only experimental code. Do not apply
`minimal-fp32-tma.patch` or update the PRs until results are reviewed.

For transfer, check in this entire directory on a separate benchmark branch.
Generated files are ignored. Do not check in `agent_space`, caches, logs, or
binary profiles. Shared references contain curated timings only, with private
device/build metadata removed. Keep raw diagnostics outside this directory.
Benchmark changes and new references should stay separate from the feature PR
stack. The original benchmark commit must be cleaned up before any public
publication; deleting files in a later commit does not remove earlier copies.

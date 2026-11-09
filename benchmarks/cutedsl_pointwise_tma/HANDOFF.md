# Blackwell hand-off

Continue the 4096x4096 FP32 transpose-copy and dense-add comparison on Blackwell,
using GPU index 0 only. This is a transfer-only benchmark package, not an update
to the feature PRs. No optimization has been applied to `torch/_native`.

Use this same feature checkout with a working editable PyTorch build, CuTeDSL,
Triton, CUDA bindings, and Nsight Compute (`ncu`) for profiling. The Rubin run
used CuTeDSL 4.8.0, Triton 3.8.0, Python 3.12, and CUDA 13.5; record the Blackwell
versions. Inspect the node's environment before building or installing anything.
Do not source `/tmp/mjoux/setup.sh` blindly: it also installs and builds.

From the repository root, with the existing venv (override `POINTWISE_PERF_VENV`
if its location differs):

```bash
source benchmarks/cutedsl_pointwise_tma/env.sh
python benchmarks/cutedsl_pointwise_tma/final_compare.py 4
python benchmarks/cutedsl_pointwise_tma/final_compare.py 1
python benchmarks/cutedsl_pointwise_tma/profile_suite.py
```

These commands validate outputs and compare original TMA, the shared-layout fix,
the optimized TMA prototype, ATen, and tuned non-TMA kernels. Results are written
beside the scripts; compare them with `reference/rubin/`. The frozen baseline is
in `published/`; do not format or edit it because variant generation uses exact
source-text replacements. It comes from submitted commit
`57476b7efc075dbf974e37876eb749f0020e8b81`. Shared native utilities are imported
from this checkout (Rubin base: `9712d89f589e2bab95c8dd9cadf0ced95d5a0232`).

Rubin medians in microseconds (original / non-TMA / optimized TMA): transpose
12.41 / 11.45 / 12.04; host-scalar add 16.54 / 11.49 / 11.67; tensor add
24.18 / 15.84 / 15.98. Dense TMA had two-way shared-memory bank conflicts.
A row-major shared tile removes them; two pipeline tiles and vector stores
close most of the gap. LDGSTS did not beat the best LDG transpose tested.

First reproduce these fixed configurations, then retune for Blackwell using
`sweep.py` modes `tma`, `rowmajor`, `classic`, `cute`, `parallel`, and `rect`,
and `bench.py --tune`. The CuTe vector baseline hardcodes Rubin's vec=8 and
EPT=32 (unary)/16 (binary); also test Blackwell-appropriate vector/EPT choices.
`final_compare.py` uses fixed finalists, so retuned winners need a separate run.

Keep identical output strides, GPU-only graph timing, and output-poisoning
checks. Copy and contiguous-add TMA measurements force the executor; they do
not establish that normal dispatch selects TMA. Profile shared excess
wavefronts, instruction/store counts, and barrier stalls separately from timing.
The combined vector epilogue is FP32-only experimental code. Do not apply
`minimal-fp32-tma.patch` or update the PRs until results are reviewed.

For transfer, check in this entire directory on a separate benchmark branch.
Generated files are ignored. Do not check in `agent_space`, caches, logs, or
binary profiles. Nothing has been staged or committed in this session.

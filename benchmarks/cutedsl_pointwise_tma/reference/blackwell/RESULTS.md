# GB200 and Rubin pointwise comparison

Use a non-TMA vector path for dense elementwise work and reserve TMA tuning for actual transpose/layout conversion. A copy-specific TMA configuration now beats the tested non-TMA references at 64 MiB on GB200 and is effectively tied at 1 GiB. Rubin already favors TMA copy in the preserved measurements. No single fixed tile, block size and pipeline depth has been validated as best on both architectures.

## Measurement conditions

FP32; 4096x4096 (64 MiB per matrix) and 16384x16384 (1 GiB per matrix); four rotating buffer sets. Localized runs split the full problem into two separately allocated rectangular halves, each in its matching locality-domain memory pool and green-context stream. Both halves are included in the timed operation. Allocation domains are checked. Unpartitioned controls are also included.

GB200 clocks were left unchanged; observed memory clock was 3996 MHz. The Rubin reference used its requested 4752 MHz memory-clock lock. Only the assigned device was used. Compilation, allocation, initialization and validation are outside timing. Every graph is replayed after poisoning all outputs with NaN, then checked exactly against ATen.

Final comparisons use 64 operations per CUDA graph, 25 interleaved samples per implementation and three warmup replays per sample. Tables use the minimum graph-averaged latency, as requested; medians and sample ranges are retained in JSON. Screening uses 32 operations and nine samples and is kept separate from final confirmation. Tiny differences below about 1% should be treated as near ties.

All timing tables below exclude profiling replay times. Reported bandwidth counts logical reads and writes; it is not a hardware-counter measurement. A unary/copy operation transfers twice the stated input footprint, and tensor add transfers three times that footprint.

## Copy tuning

The practical GB200 choice across both sizes is the original TMA executor with a 64x64 tile, one pipeline stage and 256 threads. It uses scalar global stores; the existing producer loop is sufficient. Reconstruct it with `bench.load_snapshot("baseline", 1, 64, 256)` and `bench.make_tma(...)`. The benchmark now propagates experimental tile/block sizes into the launch config; changing module constants alone did not do that.

| Architecture | Input | Original TMA (us) | Recommended TMA (us) | Best tested non-TMA (us) |
|---|---|---:|---:|---:|
| Rubin | 64 MiB | 10.654 | 10.654 | 11.032 |
| Rubin | 1 GiB | 139.296 | 139.296 | 146.591 |
| GB200 | 64 MiB | 21.613 | 19.248 | 20.218 |
| GB200 | 1 GiB | 336.427 | 311.487 | 312.622 |

The recommended GB200 choice reduces original TMA latency by 10.9% at 64 MiB and 7.4% at 1 GiB. It is 4.8% faster than the tuned non-TMA reference at 64 MiB; its 0.4% advantage at 1 GiB is a near tie. It reaches about 6.9 TB/s of effective copy payload bandwidth at 1 GiB. All copy winners in these final comparisons are localized.

Absolute peak GB200 minima were 19.081 us for a 32x256 rectangular tile at 64 MiB and 311.463 us for the distributed-producer 64x64/one-stage/256-thread variant at 1 GiB. The rectangle slows to 323.221 us at 1 GiB. The simpler recommended 64x64 variant stays within 0.9% of the best tested TMA at both sizes. With one stage, only the first producer warp is needed, so distributing producers adds no useful parallelism.

For Rubin, the table deliberately keeps the measured original 32x32/six-stage/128-thread executor at both sizes. The preserved four-tile distributed-producer version improves 1 GiB slightly, to 137.887 us, while its 64 MiB minimum is 10.777 us. Both are valid candidates; neither the new GB200 geometry nor its extra CTA-order experiments have been run on Rubin. Do not assume the new geometry is portable without that test.

CTA traversal changes, 128x128 tiles, 512-thread blocks and vector-store epilogues did not improve both GB200 sizes over the selected simple configuration. All 93 copy-specific configurations passed checks at small size and both requested footprints. The JSON preserves their parameters and timing outcomes.

## Dense adds and mixed layouts

The following minima are from full final comparisons and controls, not short tuning sweeps. TMA and non-TMA minima can come from different tested configurations. The non-TMA column includes ATen.

| Architecture | Input | Workload | Best TMA (us) | Best non-TMA (us) |
|---|---|---|---:|---:|
| Rubin | 64 MiB | dense_add_host | 11.039 | 9.368 |
| Rubin | 64 MiB | dense_add_cuda | 11.167 | 9.375 |
| Rubin | 64 MiB | dense_add_tensor | 13.856 | 13.086 |
| Rubin | 64 MiB | mixed_add | 14.239 | 16.038 |
| Rubin | 1 GiB | dense_add_host | 143.008 | 122.655 |
| Rubin | 1 GiB | dense_add_cuda | 143.769 | 122.534 |
| Rubin | 1 GiB | dense_add_tensor | 190.624 | 181.926 |
| Rubin | 1 GiB | mixed_add | 204.313 | 219.167 |
| GB200 | 64 MiB | dense_add_host | 19.241 | 18.839 |
| GB200 | 64 MiB | dense_add_cuda | 19.236 | 18.864 |
| GB200 | 64 MiB | dense_add_tensor | 27.545 | 27.687 |
| GB200 | 64 MiB | mixed_add | 28.840 | 30.052 |
| GB200 | 1 GiB | dense_add_host | 299.573 | 300.112 |
| GB200 | 1 GiB | dense_add_cuda | 300.366 | 300.176 |
| GB200 | 1 GiB | dense_add_tensor | 443.770 | 436.479 |
| GB200 | 1 GiB | mixed_add | 462.568 | 453.621 |

Rubin clearly favors the vector path for all three dense-add cases. On GB200 the optimized TMA add occasionally has a slightly lower minimum, but the differences are below 1%; a tensor-add case favors the vector path more clearly. There is no consistent dense-add TMA advantage across architectures. Retain vector kernels for these layouts.

For GB200, vec=2 or 4, EPT=8, block=256 is a good vector choice in this sweep. Rubin used vec=8, EPT=32 for unary/scalar add and EPT=16 for tensor add, block=128. The preserved Rubin vector configuration is also competitive on GB200, but it is not always the fastest. These are FP32 benchmark configurations, not a new general dtype heuristic.

Mixed add is the genuinely conflicting-layout case: one row-major operand, one transposed operand and a row-major output. Rubin favors the shared-layout fix with two stages at both footprints. GB200 favors the row-major shared layout with one stage at 64 MiB, but the tuned Triton reference leads TMA by about 2% at 1 GiB. The original transposed-output mirror, same-transposed inputs and transposed CUDA-scalar add were also validated in the fixed-configuration runs. Do not generalize transpose-copy tuning to every mixed pointwise layout.

## Profiling

Targeted profiles completed for the original TMA copy, the 64x64/one-stage/256-thread copy and the tuned Triton reference. These were unpartitioned single-launch diagnostics, not localized pair measurements; their replay times are excluded from the tables. All raw profiler files remain outside this package.

The larger square reduces the number of TMA tiles and associated barriers per matrix by four relative to 32x32. The one-stage configuration also removes the six-stage block pipeline. These are source-level changes; the new executor keeps the same arithmetic and scalar global-store strategy.

## Reproduction and saved artifacts

Activate the working environment, set `POINTWISE_PERF_CACHE_ROOT` to an allowed temporary cache root, and optionally set `POINTWISE_PERF_OUTPUT_DIR` outside this package. No build or package installation was needed.

```bash
source benchmarks/cutedsl_pointwise_tma/env.sh
python benchmarks/cutedsl_pointwise_tma/final_compare.py 4 --n 4096 --localized --cases transpose_copy dense_add_host dense_add_cuda dense_add_tensor mixed_add --candidates benchmarks/cutedsl_pointwise_tma/reference/blackwell/finalists-candidates.json
python benchmarks/cutedsl_pointwise_tma/final_compare.py 4 --n 16384 --localized --cases transpose_copy dense_add_host dense_add_cuda dense_add_tensor mixed_add --candidates benchmarks/cutedsl_pointwise_tma/reference/blackwell/finalists-candidates.json
```

For matched unpartitioned controls, omit `--localized` and use `control-candidates.json`. `results.json` contains 18 complete runs and 1,034 timing rows, all candidate definitions, and hashes for the final measured source. It retains median, minimum, p10/p90 and maximum for every candidate; complete sample arrays and raw machine metadata remain outside the shared package. To reconstruct any screening set, copy that run's `metadata.candidates` object to a temporary JSON file and pass it to `--candidates`.

The frozen executor is public commit `57476b7efc075dbf974e37876eb749f0020e8b81`; do not edit `published/`. CuTeDSL 4.8.0 and Triton 3.8.0 were used. `parallel_issue.txt`, `vector_epilogue.txt` and the source transformations in `bench.py` preserve the intermediate implementations. Raw logs, generated source copies, profiler exports and private build/device metadata are excluded from this directory.

The copy and contiguous-add TMA tests force the executor; these measurements do not change normal dispatch. No production kernel, feature commit or PR was updated. The old `minimal-fp32-tma.patch` remains unapplied and is not the selected tuning.

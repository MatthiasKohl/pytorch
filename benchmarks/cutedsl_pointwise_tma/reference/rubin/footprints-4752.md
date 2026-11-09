# Controlled Rubin footprint comparison

These results supersede the earlier unlocalized measurements for judging the required localized setup. All measurements here are new runs on the replacement Rubin node; no timings were carried over from the interrupted node.

Conditions: GPU 0 only; memory clocks explicitly locked at 4752,4752 MHz; SM clocks not changed; FP32; four rotating buffer sets; 64 logical operations per CUDA graph; 25 interleaved timing samples per implementation. Compilation, allocation, input initialization, and correctness checks are outside timing. Every graph was validated after poisoning all outputs with NaN.

4096x4096 is 64 MiB per matrix. 16384x16384 is 1 GiB per matrix. In localized mode each is split into two separate rectangular allocations of half that size, using domain-local memory pools and matching green-context streams. Both halves are included in each timed operation. Allocation domains are queried and checked. The pair is joined before the timing event completes; tensors and contexts remain alive throughout. The localized partition can expose fewer SMs than the unpartitioned control; all localized implementations use the same partition.

A unary operation/copy transfers 2 GiB per 1 GiB input (one read and one write). Tensor add transfers 3 GiB (two reads and one write). Reported bandwidth is this logical payload divided by elapsed time, not a hardware-counter DRAM measurement.

The executor snapshot is frozen at public commit 57476b7efc075dbf974e37876eb749f0020e8b81. CuTeDSL 4.8.0 and Triton 3.8.0 were used. The accompanying JSON retains measured-source hashes, clock checks, medians and sample ranges. Machine-specific hardware/build metadata and profiler data are kept outside this package. Subsequent output-path cleanup does not change the measured computational kernels.

Configurations were held fixed across sizes to test the proposed changes; the non-TMA finalists were not exhaustively retuned for localized rectangular halves. Reproduce these controls before retuning on Blackwell.

Variants: original_tma has six pipeline tiles; shared_fix_tma uses a row-major normal-input shared tile and two pipeline tiles; optimized_tma additionally uses four-element vector stores. For transpose_copy only, optimized_tma instead uses the distributed-producer/four-pipeline-tile experiment. The non-TMA reference below is the fastest of the included Triton and CuTeDSL prototypes for that workload, excluding ATen; the JSON identifies the implementation.

## 64mib-localized

| Workload | Original TMA (us) | Shared fix + two tiles (us) | Other TMA prototype (us) | Non-TMA reference (us) | ATen (us) |
|---|---:|---:|---:|---:|---:|
| transpose_copy | 10.784 | 11.552 | 10.908 | 11.055 | 12.192 |
| dense_add_host | 16.928 | 11.680 | 11.168 | 9.504 | 9.504 |
| dense_add_cuda | 16.800 | 11.690 | 11.169 | 9.392 | 50.336 |
| dense_add_tensor | 24.865 | 13.986 | 13.858 | 13.098 | 13.217 |
| same_transposed_add | 24.864 | 13.984 | 13.730 | 13.216 | 13.217 |
| transposed_add_cuda | 16.802 | 11.680 | 11.169 | 9.504 | 50.464 |
| mixed_add | 17.319 | 14.368 | 15.008 | 16.160 | 65.568 |
| mixed_add_transposed_out | 17.312 | 14.368 | 14.880 | 15.904 | 65.568 |

| Workload | Non-TMA implementation |
|---|---|
| transpose_copy | non_tma_triton |
| dense_add_host | non_tma_cute_vector |
| dense_add_cuda | non_tma_cute_vector |
| dense_add_tensor | non_tma_cute_vector |
| same_transposed_add | non_tma_triton |
| transposed_add_cuda | non_tma_cute_vector |
| mixed_add | non_tma_triton |
| mixed_add_transposed_out | non_tma_triton |

## 1gib-localized

| Workload | Original TMA (us) | Shared fix + two tiles (us) | Other TMA prototype (us) | Non-TMA reference (us) | ATen (us) |
|---|---:|---:|---:|---:|---:|
| transpose_copy | 139.424 | 159.023 | 138.015 | 146.723 | 181.416 |
| dense_add_host | 218.655 | 152.865 | 143.136 | 122.785 | 122.786 |
| dense_add_cuda | 215.585 | 152.368 | 143.779 | 122.784 | 781.728 |
| dense_add_tensor | 360.480 | 193.439 | 190.751 | 182.048 | 182.305 |
| same_transposed_add | 360.480 | 193.440 | 190.752 | 182.048 | 182.183 |
| transposed_add_cuda | 215.584 | 152.248 | 143.648 | 122.912 | 782.880 |
| mixed_add | 235.808 | 204.447 | 211.231 | 219.296 | 993.824 |
| mixed_add_transposed_out | 235.808 | 204.575 | 211.360 | 205.215 | 994.208 |

| Workload | Non-TMA implementation |
|---|---|
| transpose_copy | non_tma_triton |
| dense_add_host | non_tma_cute_vector |
| dense_add_cuda | non_tma_cute_vector |
| dense_add_tensor | non_tma_cute_vector |
| same_transposed_add | non_tma_cute_vector |
| transposed_add_cuda | non_tma_cute_vector |
| mixed_add | non_tma_triton |
| mixed_add_transposed_out | non_tma_triton |

## Conclusions

- The add fixes remain useful at 1 GiB. The combined prototype reduces host-scalar add time by 34.5% and tensor add time by 47.1% within the same localized setup.
- Do not apply the global two-tile pipeline change blindly: localized 1 GiB transpose copy regresses from 139.42 to 159.02 us (14.1% longer). The row-major change does not affect its transposed shared input, so this isolates the pipeline-depth change. The four-tile distributed-producer prototype is about 138.02 us.
- Localization changes the ordering seen in the original unlocalized report. In the included configurations, original TMA transpose copy now beats the non-TMA reference at both sizes. This is not a claim about every possible retuned non-TMA kernel.
- Dense vector kernels still lead the optimized TMA add, especially unary add: 143.14 versus 122.79 us at 1 GiB. For tensor add the gap is smaller: 190.75 versus 182.05 us.
- Localized vector tensor add reaches 17.69 TB/s of effective payload bandwidth at 1 GiB. All versions use the same pools, partition and timing methodology.
- For mixed add, the smaller shared-layout/two-tile change is preferable to the vector-store experiment: 204.45 versus 211.23 us at 1 GiB.
- A fresh-process confirmation reran transpose, host-scalar add, CUDA-scalar add, tensor add and mixed add. Across all 30 reported configurations, its median differed by at most 0.12% from the first large localized run. All output checks passed.

## Unpartitioned controls

| Workload | 64 MiB original / combined TMA (us) | 1 GiB original / combined TMA (us) |
|---|---:|---:|
| transpose_copy | 12.444 / 12.028 | 168.564 / 166.089 |
| dense_add_host | 16.508 / 11.653 | 206.287 / 153.597 |
| dense_add_cuda | 16.281 / 11.796 | 203.800 / 153.612 |
| dense_add_tensor | 24.144 / 15.972 | 340.409 / 228.427 |
| mixed_add | 17.510 / 16.438 | 230.925 / 227.981 |

The original bank-conflicted add can get slower after localization. The localized partition also has fewer SMs; a bandwidth benefit is not guaranteed for a shared-memory/instruction-limited kernel. Keep the within-mode original-versus-prototype comparison separate from the localization comparison.

The existing profile_suite.py is an unlocalized single-kernel diagnostic, not a localized pair profile. No new counter profiles were collected for this footprint follow-up. The earlier bank-conflict diagnosis is retained as context, not presented as new profiling of this node.

No production kernel or feature PR was changed. minimal-fp32-tma.patch remains an unapplied candidate and contains the globally shallower pipeline that regresses transpose; do not apply it unchanged. Use HANDOFF.md for reproduction and Blackwell setup.

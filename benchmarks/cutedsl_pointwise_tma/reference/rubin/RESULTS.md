# Historical Rubin timing comparison

This historical run did not explicitly lock memory clocks or enable
localization. Use `footprints-4752.md` and `footprints-4752.json` for the newer
controlled 64 MiB / 1 GiB comparisons.

The baseline executor is frozen at public commit
`57476b7efc075dbf974e37876eb749f0020e8b81`. Inputs and outputs are FP32,
4096x4096, with preallocated outputs and four rotating buffer sets. Timings
exclude allocation and compilation and use 25 interleaved samples of 64-call
CUDA graphs. All graph outputs were checked after poisoning them with NaN.

## Rotating-buffer timings

| Workload | Original TMA (us) | Best non-TMA (us) | Original / non-TMA | Best TMA prototype (us) | ATen (us) |
|---|---:|---:|---:|---:|---:|
| transpose_copy | 12.405 | 11.447 | 1.084x | 12.037 | 12.733 |
| dense_add_host | 16.537 | 11.494 | 1.439x | 11.668 | 11.913 |
| dense_add_cuda | 16.307 | 11.437 | 1.426x | 11.801 | 50.921 |
| dense_add_tensor | 24.180 | 15.844 | 1.526x | 15.979 | 15.985 |
| same_transposed_add | 24.178 | 15.835 | 1.527x | 15.995 | 15.965 |
| transposed_add_cuda | 16.317 | 11.430 | 1.428x | 11.690 | 50.850 |
| mixed_add | 17.532 | 17.063 | 1.028x | 16.188 | 65.654 |
| mixed_add_transposed_out | 17.512 | 16.858 | 1.039x | 16.203 | 65.683 |

`optimized_tma` uses the vector-store/two-tile prototype for add, and the
four-tile/distributed-producer prototype for transpose copy. These are
experimental alternatives; the newer controlled report describes their
tradeoffs. Some copy and contiguous-add comparisons force the TMA executor
rather than exercise its ordinary dispatch route.

The companion historical JSON files retain per-implementation timing samples.
Profiler output and machine-specific provenance are not part of this package.
Source generators, the frozen baseline, and replacement snippets preserve the
prototypes without generated kernel copies. See `../../HANDOFF.md` for current
reproduction instructions.

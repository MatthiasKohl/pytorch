# Pointwise TMA performance investigation

Measured on GPU index 0, NVIDIA VR NVL72, compute capability 10.7, driver 620.69, CUDA 13.5. No B200 was available in this container. The GPU has 212 SMs and 126 MiB of L2.

The TMA baseline is the submitted PR snapshot at pytorch/pytorch commit 57476b7efc075dbf974e37876eb749f0020e8b81, loaded from published/kernel.py and published/_layout.py without changing tracked files. Other shared utilities come from this checkout. Kernels use FP32 input/output throughout.

## Method

4096 x 4096 matrices (64 MiB each). Explicit preallocated outputs; compilation and allocation excluded. CUDA graphs contain 64 sequential kernel calls, rotating over four independent input/output sets. Every replay was checked after poisoning every output with NaN. Timings are medians of 25 interleaved, deterministically randomized samples per implementation, with three warmup graph replays before each sample. final-buffers4.json contains all samples and percentiles. final-buffers1.json repeats the experiment using one fixed input/output set.

Transpose copy reads a row-major matrix and writes the same logical elements into an output with strides (1, 4096). This performs the same physical data rearrangement as materializing a transpose. The copy executor uses identity arithmetic. Host-scalar add uses a specialized unary x+1 executor. CUDA-scalar add loads an actual one-element device tensor. Tensor add uses two matrix operands.

Copy and ordinary contiguous matrix/host-scalar calls are executor-level diagnostic comparisons, not a claim that existing native dispatch routes them through TMA. Contiguous CUDA-scalar calls normally use the vector path. Same-transposed-layout addition and mixed-layout addition exercise layouts admitted by TMA. All compared implementations write identical output layouts.

## Rotating-buffer results

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

The fastest non-TMA kernel is the specialized Triton LDG implementation. A separately written CuTeDSL LDG/shared-memory transpose also beats the original TMA kernel: 11.754 us. The best tested LDGSTS transpose measured 13.400 us. Both 32-bit and 128-bit LDGSTS variants were tested; neither beat the best LDG kernel. Nsight confirms actual LDGSTS instructions in the asynchronous prototype.

## Profiling findings

1. Dense TMA shared reads have deterministic two-way bank conflicts. For x+1, SourceCounters reports 1,048,576 shared wavefronts versus an ideal 524,288. For x+y, it reports 2,097,152 versus an ideal 1,048,576. The normal-input layout tiles an 8-row by 4-column atom with rows extended before columns; its physical order does not fit the epilogue lane mapping. A genuinely row-major shared tile removes the excessive wavefronts.
2. The original TMA epilogue issues 524,288 warp-level global-store instructions. Vector stores reduce this to 131,072. Combined with a shallower two-tile pipeline and the shared-layout change, dense-add throughput approaches the non-TMA vector baseline. The smallest useful change (shared layout plus pipeline depth) already gets x+1 to about 12.08 us and x+y to about 16.11 us; the vector epilogue brings these to about 11.67 and 15.98 us.
3. TMA startup and barrier costs also matter. Dense x+1 issues about 8.23 million warp instructions in the original kernel versus 1.74 million for the non-TMA Triton vector kernel. The optimized TMA prototype uses about 4.10 million. Its shared-memory footprint falls from about 25.65 KB to 9.23 KB per block, including driver overhead. Profiled barrier-stall cycles per issued instruction fall from about 12.29 to 2.24.
4. Pure transpose does not have the deterministic normal-input shared-layout conflict. The non-TMA transpose uses 131,072 global-store instructions versus TMA's 524,288, and fewer total instructions (about 3.83 million versus 7.84 million). Distributing TMA issue work among warps with four tiles per block reduces the measured original 12.405 us to 12.037 us; the best non-TMA result remains 11.447 us. A rectangular 32x128 one-transfer prototype is similar. Larger square tiles, simply removing the XOR lane mapping, and vectorizing stores with the old shared layout did not improve the result.
5. Mixed-layout add remains a good TMA use case: the small shared-layout/pipeline fix gives about 16.19 us versus 17.06 us for the tuned non-TMA prototype and 65.65 us for ATen. Reversing operand/output orientation gives similar behavior. Output stride orientation itself does not create the large performance gap.

Nsight timings include profiler overhead and are not used as the latency results above. All counter reports and exported .ncu-repz files are retained. Cache and clock control were disabled for the comparative profiles; use measured event timings to compare latency. The raw bank-conflict event counts can include additional replay/hardware effects, so the exact excessive-wavefront counts above come from SourceCounters' instruction-based ideal-versus-actual calculation.

## Prototype scope

All changes are in ignored scratch space; no branch, index, commits, PRs, build caches, or installed packages were changed. Larger tiles, rectangular transfers, store mappings, pipeline depths, and distributed producers were swept. The best combined vector epilogue is specialized to aligned FP32 outputs and is not a general replacement ready for all existing dtypes/layouts. A smaller shared-layout fix is available as minimal-fp32-tma.patch for review; it has not been applied. B200 measurements and broader regression testing are still needed before choosing production heuristics.

## Reproduce

```bash
source agent_space/pointwise_tma_perf/env.sh
python agent_space/pointwise_tma_perf/final_compare.py 4
python agent_space/pointwise_tma_perf/final_compare.py 1
python agent_space/pointwise_tma_perf/sweep.py tma
python agent_space/pointwise_tma_perf/sweep.py cute
python agent_space/pointwise_tma_perf/sweep.py classic
python agent_space/pointwise_tma_perf/sweep.py rowmajor
python agent_space/pointwise_tma_perf/profile_suite.py
```

profile_suite.py skips completed profiles. All cache paths are set under /tmp/mjoux. bench.py contains the non-TMA Triton kernels; cute_transpose.py contains the LDG/LDGSTS prototypes. Generated kernel_rowmajor*.py files contain the concrete TMA experiments. final_compare.py selects each reported prototype explicitly.

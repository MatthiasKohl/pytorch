#pragma once

// This file provides two functions to help write GPU elementwise kernels:
//
//   gpu_kernel(TensorIterator iter, <lambda>)
//   gpu_kernel_with_scalars(TensorIterator iter, <lambda>)
//
// The gpu_kernel_with_scalars generates specializations that support a
// single scalar CPU argument, such as from `cuda_tensor + 5`. The CPU scalar
// is lifted to a kernel parameter instead of copying to device memory.
// This should be  used in conjunction with TensorIterator::allow_cpu_scalars_,
// which is the default for TensorIterator::binary_op. Otherwise, all inputs
// and the output must be on the GPU.
//
// For example, to write a reciprocal kernel for GPU float Tensors:
//
//   gpu_kernel(iter, []GPU_LAMBDA(float a) {
//    return 1.0f / a;
//   });
//
// To write a multiplication kernel for GPU float Tensors where one argument
// may be a CPU scalar:
//
//   gpu_kernel_with_scalars(iter, []GPU_LAMBDA(float a, float b) {
//     return a * b;
//   });
//
// See BinaryOpsKernel.cu for the complete implementation
//

#include <array>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <tuple>
#include <type_traits>

#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/DynamicCast.h>
#include <c10/core/ScalarType.h>
#include <c10/macros/Macros.h>
#include <c10/util/SmallVector.h>
#include <c10/util/TypeCast.h>
#include <c10/util/env.h>

#if !defined(USE_ROCM)
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#endif

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080
#include <cuda/ptx>
#define AT_CUDA_TMA_ELEMENTWISE_ENABLED 1
#else
#define AT_CUDA_TMA_ELEMENTWISE_ENABLED 0
#endif

#ifdef __NVCC__
#define ASSERT_HOST_DEVICE_LAMBDA(type)                       \
  static_assert(                                              \
      __nv_is_extended_host_device_lambda_closure_type(type), \
      #type " must be a __host__ __device__ lambda")
#else
#define ASSERT_HOST_DEVICE_LAMBDA(type)
#endif

namespace at::native {

#ifdef USE_ROCM
// Custom configuration for vectorized elementwise kernel
// with template instantiation.
namespace vectorized_templated_config {
constexpr int num_threads() {
  return 512;
}

constexpr int elems_per_thread() {
  return 32;
}

constexpr int block_work_size() {
  return elems_per_thread() * num_threads();
}
} // namespace vectorized_templated_config
#endif

template <typename args_t, size_t... Is>
constexpr auto sum_of_sizes(args_t args, std::index_sequence<Is...>) {
    if constexpr (sizeof...(Is) == 0) {
      return 0;
    } else {
      return (sizeof(std::tuple_element_t<Is, args_t>) + ...);
    }
}

template <typename args_t, size_t... Is>
constexpr size_t max_of_sizes(args_t, std::index_sequence<Is...>) {
  size_t max_size = 0;
  ((max_size = std::max(max_size, sizeof(std::tuple_element_t<Is, args_t>))),
   ...);
  return max_size;
}

#ifdef USE_ROCM
template <int io_sizes>
constexpr auto elems_per_thread(){
  if constexpr (io_sizes == 1) {
    return 16;
  } else if constexpr (io_sizes < 4) {
    return 8;
  } else {
    return 4;
  }
}
#else
template <int io_sizes>
constexpr auto elems_per_thread(){
  if constexpr (io_sizes == 1) {
    return 16;
  } else {
    return 8;
  }
}

// Rubin (SM 10.7) supports 1024 resident threads per SM. Target 128 bytes of
// input per thread to keep roughly 128 KiB in flight per SM.
template <size_t input_size>
constexpr auto elems_per_thread_128b(size_t output_size) {
  if constexpr (input_size == 0) {
    // note: without inputs, we simply double the number of elements per thread
    // for SM 10.7 as there are half the number of threads per SM compared
    // to previous generations.
    return output_size == 1 ? 32 : 16;
  } else if constexpr (input_size >= 128) {
    return 1;
  } else {
    constexpr auto rounded_input_size = std::bit_ceil(input_size);
    return static_cast<int>(128 / rounded_input_size);
  }
}
#endif


//thread work size of 8 regresses the perf of elementwise kernel on cuda
//this doesn't change ROCm behavior as thread_work_size is already 4 on ROCm
constexpr int elementwise_thread_work_size() {return 4;}
constexpr int elementwise_block_work_size() {
  return elementwise_thread_work_size() * num_threads();
}

#ifdef USE_ROCM
template <typename args_t, size_t... Is>
constexpr auto input_size(args_t args, std::index_sequence<Is...>) {
  if constexpr (sizeof...(Is) == 0) {
    return 0;
  } else {
    return sizeof(std::tuple_element_t<0, args_t>);
  }
}

template <int vec_size, int io_size>
constexpr auto calc_optimal_vec_size() {
  static_assert(vec_size != 0);
  static_assert(io_size != 0);
  if constexpr (io_size == 1 && vec_size >= 16) {
    return 16;
  } else if constexpr (io_size <= 2 && vec_size >= 8) {
    return 8;
  } else if constexpr (io_size <= 4 && vec_size >= 4) {
    return 4;
  } else if constexpr (vec_size >= 4) {
    return 4;
  } else if constexpr (vec_size >= 2) {
    return 2;
  } else {
    return 1;
  }
}
#endif

template <typename func_t>
constexpr auto calc_io_size(){
  using traits = function_traits<func_t>;
  using args_t = typename traits::ArgsTuple;
#ifdef USE_ROCM
  constexpr auto input_size = at::native::input_size(args_t{}, std::make_index_sequence<std::tuple_size_v<args_t>>{});
  constexpr auto output_size = sizeof(typename traits::result_type);
  return (input_size > 0) ? ((input_size < output_size) ? input_size : output_size) : output_size;
#else
  constexpr auto input_size = at::native::sum_of_sizes(args_t{}, std::make_index_sequence<std::tuple_size_v<args_t>>{});
  constexpr auto output_size = sizeof(typename traits::result_type);
  return input_size + output_size;
#endif
}

#ifndef USE_ROCM
// To save on binary size of libtorch_cuda.so, we split the vectorized_elementwise_kernel
// into three: one targeting 128 bytes per thread (and only vec_size=8), one for
// vec_size=8, and one for vec_size=[2, 4]. vec_size=8 is only used on sm_90
// and sm_10x, and the 128-byte-per-thread configuration is only used on sm_10x.
template <int vec_size, int tws, typename func_t, typename array_t>
__forceinline__ __device__ void vectorized_elementwise_kernel_impl(
    int N,
    func_t f,
    array_t data) {
  using traits = function_traits<func_t>;
  static_assert(tws > 0, "Thread work size must be positive");
  constexpr int bws = tws * num_threads();
  int remaining = N - bws * blockIdx.x;

  // The compiler generally preserves source order, so keep the hot path first
  // to improve instruction cache use.
  if (remaining >= bws) {
    elementwise_kernel_helper(
        f, memory::policies::vectorized<vec_size, array_t, tws>(data));
  } else {
    auto input_calc = TrivialOffsetCalculator<traits::arity>();
    auto output_calc = TrivialOffsetCalculator<1>();
    auto loader = memory::LoadWithoutCast();
    auto storer = memory::StoreWithoutCast();
    auto policy = memory::policies::unroll<
        array_t,
        decltype(input_calc),
        decltype(output_calc),
        memory::LoadWithoutCast,
        memory::StoreWithoutCast,
        tws>(data, remaining, input_calc, output_calc, loader, storer);
    elementwise_kernel_helper(f, policy);
  }
}

template <
    int vec_size,
    typename func_t,
    typename array_t,
    bool use_128b_tws = false>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void vectorized_elementwise_kernel(int N, func_t f, array_t data) {
  if constexpr (vec_size == 8 && use_128b_tws) {
#if __CUDA_ARCH__ / 100 == 10
    using output_t = typename function_traits<func_t>::result_type;
    constexpr auto input_size = calc_io_size<func_t>() - sizeof(output_t);
    constexpr auto tws_128b = elems_per_thread_128b<input_size>(sizeof(output_t));
    vectorized_elementwise_kernel_impl<vec_size, tws_128b>(N, f, data);
#else
    CUDA_KERNEL_ASSERT(
        false &&
        "Fatal! vectorized_elementwise_kernel<8,...> with 128-byte-per-thread "
        "work size supports only sm_10x. Please report an issue on GitHub.");
#endif
  } else if constexpr (vec_size == 8) {
#if __CUDA_ARCH__ / 100 == 9 || __CUDA_ARCH__ / 100 == 10
    constexpr auto io_size = calc_io_size<func_t>();
    constexpr auto tws = elems_per_thread<io_size>();
    vectorized_elementwise_kernel_impl<vec_size, tws>(N, f, data);
#else
    CUDA_KERNEL_ASSERT(
      false && "Fatal! vectorized_elementwise_kernel<8,...> supports only "
      "sm_90 and sm_10x. Please report an issue on GitHub.");
#endif // __CUDA_ARCH__ / 100 == 9 || __CUDA_ARCH__ / 100 == 10
  } else {
    constexpr auto io_size = calc_io_size<func_t>();
    constexpr auto tws = elems_per_thread<io_size>();
    vectorized_elementwise_kernel_impl<vec_size, tws>(N, f, data);
  }
}

#else // USE_ROCM
template <int vec_size, int tws, typename func_t, typename array_t>
__forceinline__ __device__ void vectorized_elementwise_kernel_impl(int N, func_t f, array_t data) {
  using traits = function_traits<func_t>;
  constexpr auto io_size = calc_io_size<func_t>();
  constexpr int bws = tws * num_threads();
  int remaining = N - bws * blockIdx.x;

  if (remaining < bws) { // if this block handles the reminder,
                                       // just do a naive unrolled loop
    auto input_calc = TrivialOffsetCalculator<traits::arity>();
    auto output_calc = TrivialOffsetCalculator<1>();
    auto loader = memory::LoadWithoutCast();
    auto storer = memory::StoreWithoutCast();
    auto policy = memory::policies::unroll<
        array_t,
        decltype(input_calc),
        decltype(output_calc),
        memory::LoadWithoutCast,
        memory::StoreWithoutCast,
        tws>(
        data, remaining, input_calc, output_calc, loader, storer);
    elementwise_kernel_helper(f, policy);
  } else { // if this block has a full `block_work_size` data to handle, use
           // vectorized memory access
    constexpr auto optimal_vec_size = calc_optimal_vec_size<vec_size, io_size>();
    elementwise_kernel_helper(
        f, memory::policies::vectorized<optimal_vec_size, array_t, tws>(data));
  }
}

template <int vec_size, typename func_t, typename array_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void vectorized_elementwise_kernel(int N, func_t f, array_t data) {
  constexpr auto io_size = calc_io_size<func_t>();
  constexpr auto tws_942 = 16;
  constexpr auto tws = elems_per_thread<io_size>();
  if (__builtin_amdgcn_processor_is("gfx942"))
    return vectorized_elementwise_kernel_impl<vec_size, tws_942>(N, f, data);

  vectorized_elementwise_kernel_impl<vec_size, tws>(N, f, data);
}
#endif // USE_ROCM

template <
    typename func_t,
    typename array_t,
    int elems_per_thread,
    typename inp_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel(
    int N,
    func_t f,
    array_t data,
    inp_calc_t ic,
    out_calc_t oc,
    loader_t l,
    storer_t s) {
  int remaining = N - elems_per_thread * num_threads() * blockIdx.x;
  auto policy = memory::policies::
      unroll<array_t, inp_calc_t, out_calc_t, loader_t, storer_t, elems_per_thread>(
          data, remaining, ic, oc, l, s);
  elementwise_kernel_helper(f, policy);
}

// this function assume trivial 1d and no dynamic casting
template <typename func_t, typename array_t>
static inline void launch_vectorized_kernel(
    int64_t N,
    const func_t& f,
    array_t data) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  using traits = function_traits<func_t>;
  constexpr auto io_size = calc_io_size<func_t>();
  auto stream = at::cuda::getCurrentCUDAStream();
#ifdef USE_ROCM
  int vec_size = memory::can_vectorize_up_to<func_t>(data);
  c10::DeviceIndex curDevice = -1;
  AT_CUDA_CHECK(c10::cuda::GetDevice(&curDevice));
  // Similar check in vectorized_elementwise_kernel() as well. Both should be in sync.
  int tws = at::detail::getCUDAHooks().isGPUArch({"gfx942"}, curDevice) ? 16 : elems_per_thread<io_size>();
#else
  using cpp_type = typename function_traits<func_t>::result_type;
  const uint16_t max_vec_size = memory::can_vectorize_up_to<func_t>(data);
  uint16_t vec_size = 16 / static_cast<uint16_t>(sizeof(cpp_type));
  cudaDeviceProp* p = at::cuda::getDeviceProperties(stream.device().index());
  // Empirical benchmarking set this threshold; retain the existing path for
  // smaller workloads to avoid performance regressions
  // (the larger thread work size decreases thread level parallelism,
  // which is important for small footprints).
  constexpr int64_t min_sm107_io_size = 16 * 1024 * 1024;
  const bool use_sm107_optimizations =
      p->major == 10 && p->minor == 7 && N * io_size >= min_sm107_io_size;
  if (use_sm107_optimizations) {
    using args_t = typename function_traits<func_t>::ArgsTuple;
    constexpr auto max_input_size = at::native::max_of_sizes(
        args_t{}, std::make_index_sequence<std::tuple_size_v<args_t>>{});
    vec_size = 32 / static_cast<uint16_t>(std::max(sizeof(cpp_type), max_input_size));
  }
  vec_size = std::min<uint16_t>(vec_size, max_vec_size);
  // due to excessive binary size the `vectorized_elementwise_kernel` of
  // the size 8 is compiled for sm_90 and sm_10x only.
  // TODO: Lift this limitation when CUDA 12.x support is fully dropped
  if (p->major != 9 && p->major != 10) {
    vec_size = std::min<uint16_t>(vec_size, 4);
  }
#if !defined(CUDA_VERSION) || CUDA_VERSION < 12080
  if constexpr (sizeof(cpp_type) < 2) {
    vec_size = std::min<uint16_t>(vec_size, 4);
  }
#endif
  int tws = elems_per_thread<io_size>();
  constexpr auto input_size = io_size - sizeof(cpp_type);
  constexpr auto tws_128b = elems_per_thread_128b<input_size>(sizeof(cpp_type));
  if (tws_128b >= 8 && use_sm107_optimizations && vec_size == 8) {
    tws = tws_128b;
  }
#endif
  int bws = tws * num_threads();
  int64_t grid = (N + bws - 1) / bws;
  switch (vec_size) {
#ifdef USE_ROCM
    case 16:
      vectorized_elementwise_kernel<16, func_t, array_t>
          <<<grid, num_threads(), 0, stream>>>(N, f, data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
#endif
    case 8:
#ifdef USE_ROCM
      vectorized_elementwise_kernel<8, func_t, array_t>
          <<<grid, num_threads(), 0, stream>>>(N, f, data);
#else
      if (use_sm107_optimizations) {
        vectorized_elementwise_kernel<8, func_t, array_t, tws_128b >= 8>
            <<<grid, num_threads(), 0, stream>>>(N, f, data);
      } else {
        vectorized_elementwise_kernel<8, func_t, array_t>
            <<<grid, num_threads(), 0, stream>>>(N, f, data);
      }
#endif
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 4:
      vectorized_elementwise_kernel<4, func_t, array_t>
          <<<grid, num_threads(), 0, stream>>>(N, f, data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 2:
      vectorized_elementwise_kernel<2, func_t, array_t>
          <<<grid, num_threads(), 0, stream>>>(N, f, data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 1: {
      auto input_calc = TrivialOffsetCalculator<traits::arity>();
      auto output_calc = TrivialOffsetCalculator<1>();
      auto loader = memory::LoadWithoutCast();
      auto storer = memory::StoreWithoutCast();
      int64_t grid_unrolled = (N + elementwise_block_work_size() - 1) / elementwise_block_work_size();
      unrolled_elementwise_kernel<func_t, array_t, elementwise_thread_work_size()>
          <<<grid_unrolled, num_threads(), 0, stream>>>(
              N, f, data, input_calc, output_calc, loader, storer);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(false, "Unexpected vectorization size");
  }
}

#ifdef USE_ROCM
template <
    int vec_size,
    typename func_t,
    typename array_t,
    typename inp_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t,
    typename OutputType,
    typename... InputTypes>
C10_LAUNCH_BOUNDS_1(vectorized_templated_config::num_threads())
__global__ void vectorized_templated_elementwise_kernel(
    int N,
    func_t f,
    array_t data,
    inp_calc_t inp_calc,
    out_calc_t out_calc,
    loader_t loader,
    storer_t storer) {
  int remaining = N -
      vectorized_templated_config::block_work_size() *
          (gridDim.x - blockIdx.x - 1);
  constexpr bool reverted_idx = true;

  if (remaining <
      vectorized_templated_config::block_work_size()) { // if this block handles
                                                        // the reminder,
    // just do a naive unrolled loop
    auto policy = memory::policies::unroll_base<
        vectorized_templated_config::num_threads(),
        array_t,
        inp_calc_t,
        out_calc_t,
        loader_t,
        storer_t,
        vectorized_templated_config::elems_per_thread()>(
        data, remaining, inp_calc, out_calc, loader, storer);
    elementwise_kernel_helper<reverted_idx>(f, policy);
  } else { // if this block has a full `block_work_size` data to handle, use
           // vectorized memory access
    auto policy = memory::policies::vectorized_templated<
        vec_size,
        array_t,
        vectorized_templated_config::elems_per_thread(),
        vectorized_templated_config::num_threads(),
        OutputType,
        InputTypes...>(data);
    elementwise_kernel_helper<reverted_idx>(f, policy);
  }
}

// This function assume trivial 1d and supports template specialization
// to avoid dynamic casting.
// Input vectorization size is based on runtime information, i.e.
// the actual data types of the input and output tensor and cannot
// be determined using the functor type, as in regular non-templated
// vectorized kernels. The caller is in charge of selecting the correct input
// vectorization length.
template <
    typename func_t,
    typename array_t,
    typename inp_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t,
    typename OutputType,
    typename... InputTypes>
static inline void launch_vectorized_templated_kernel(
    int64_t N,
    const func_t& f,
    array_t data,
    inp_calc_t ic,
    out_calc_t oc,
    loader_t l,
    storer_t s) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  int64_t grid = (N + vectorized_templated_config::block_work_size() - 1) /
      vectorized_templated_config::block_work_size();
  auto stream = at::cuda::getCurrentCUDAStream();
  int vec_size = memory::can_vectorize_up_to<func_t>(data);
  switch (vec_size) {
    case 8:
      vectorized_templated_elementwise_kernel<
          8,
          func_t,
          array_t,
          inp_calc_t,
          out_calc_t,
          loader_t,
          storer_t,
          OutputType,
          InputTypes...>
          <<<grid, vectorized_templated_config::num_threads(), 0, stream>>>(
              N, f, data, ic, oc, l, s);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 4:
      vectorized_templated_elementwise_kernel<
          4,
          func_t,
          array_t,
          inp_calc_t,
          out_calc_t,
          loader_t,
          storer_t,
          OutputType,
          InputTypes...>
          <<<grid, vectorized_templated_config::num_threads(), 0, stream>>>(
              N, f, data, ic, oc, l, s);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 2:
      vectorized_templated_elementwise_kernel<
          2,
          func_t,
          array_t,
          inp_calc_t,
          out_calc_t,
          loader_t,
          storer_t,
          OutputType,
          InputTypes...>
          <<<grid, vectorized_templated_config::num_threads(), 0, stream>>>(
              N, f, data, ic, oc, l, s);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    default:
      // vector size 1 is not handled as part of vectorize_templated kernel
      TORCH_INTERNAL_ASSERT(false, "Unexpected vectorization size");
  }
}
#endif

template <
    typename func_t,
    typename array_t,
    typename inp_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t>
static inline void launch_unrolled_kernel(
    int64_t N,
    const func_t& f,
    array_t data,
    inp_calc_t ic,
    out_calc_t oc,
    loader_t l,
    storer_t s) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());

  int64_t grid = (N + elementwise_block_work_size() - 1) / elementwise_block_work_size();
  auto stream = at::cuda::getCurrentCUDAStream();
  unrolled_elementwise_kernel<func_t, array_t, elementwise_thread_work_size()>
      <<<grid, num_threads(), 0, stream>>>(N, f, data, ic, oc, l, s);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int nt, int vt, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel(int N, func_t f) {
  int tid = threadIdx.x;
  int nv = nt * vt;
  int idx = nv * blockIdx.x + tid;
#pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      f(idx);
      idx += nt;
    }
  }
}

template <int nt, int vt, typename func_t>
static void launch_legacy_kernel(int64_t N, const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel<nt, vt, func_t><<<grid, block, 0, stream>>>(N, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <
    typename func_t,
    typename index_t,
    typename traits = function_traits<func_t>>
C10_HOST_DEVICE typename traits::result_type invoke(
    const func_t& f,
    char* const C10_RESTRICT data[],
    const index_t strides[],
    int i);

template <
    typename func_t,
    typename index_t,
    typename traits = function_traits<func_t>>
C10_HOST_DEVICE typename traits::result_type invoke(
    const func_t& f,
    char* const C10_RESTRICT data[],
    const index_t strides[],
    const ScalarType dtypes[],
    int i);

#if AT_CUDA_TMA_ELEMENTWISE_ENABLED
namespace tma_elementwise {

constexpr int kMaxThreadsPerBlock = 512;
constexpr int kElementsPerThread = 8;
constexpr int kTilesPerBlock = 6;
constexpr int kVirtualRank = 2;
constexpr int kSmemAlignment = 1024;

template <int ninputs>
struct TmaElementwiseMeta {
  int32_t tiles_per_dim[kVirtualRank];
  int32_t virtual_shape[kVirtualRank];
  int32_t output_coord_stride_bytes[kVirtualRank];
  int32_t output_batch_stride_bytes;
  int32_t batch_count;
  int32_t input_transposed[ninputs];
  int32_t input_coord_stride_bytes[ninputs * kVirtualRank];
  int32_t smem_offsets[ninputs];
  int32_t swizzle_bytes[ninputs];
  int32_t swizzle_masks[ninputs];
  int32_t box_shift[kVirtualRank];
  int32_t tile_smem_offset;
  int32_t stage_smem_bytes;
  int32_t total_tma_bytes;
  int32_t smem_bytes;
};

template <int ninputs>
struct TmaElementwiseMaps {
  CUtensorMap maps[ninputs];
};

static inline int64_t align_up(int64_t value, int64_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

static inline bool is_aligned(const void* ptr, int64_t alignment) {
  return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

static inline bool tensor_map_data_type(
    int64_t element_size,
    CUtensorMapDataType* data_type) {
  switch (element_size) {
    case 1:
      *data_type = CU_TENSOR_MAP_DATA_TYPE_UINT8;
      return true;
    case 2:
      *data_type = CU_TENSOR_MAP_DATA_TYPE_UINT16;
      return true;
    case 4:
      *data_type = CU_TENSOR_MAP_DATA_TYPE_UINT32;
      return true;
    case 8:
      *data_type = CU_TENSOR_MAP_DATA_TYPE_UINT64;
      return true;
    default:
      return false;
  }
}

static inline CUtensorMapSwizzle tensor_map_swizzle(int32_t swizzle_bytes) {
  switch (swizzle_bytes) {
    case 32:
      return CU_TENSOR_MAP_SWIZZLE_32B;
    case 64:
      return CU_TENSOR_MAP_SWIZZLE_64B;
    case 128:
      return CU_TENSOR_MAP_SWIZZLE_128B;
    default:
      return CU_TENSOR_MAP_SWIZZLE_NONE;
  }
}

template <typename T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename traits, size_t input>
constexpr int64_t arg_element_size() {
  return sizeof(remove_cvref_t<typename traits::template arg<input>::type>);
}

template <typename traits, size_t ninputs>
struct max_arg_element_size_impl {
  static constexpr int64_t previous =
      max_arg_element_size_impl<traits, ninputs - 1>::value;
  static constexpr int64_t current =
      arg_element_size<traits, ninputs - 1>();
  static constexpr int64_t value = previous > current ? previous : current;
};

template <typename traits>
struct max_arg_element_size_impl<traits, 0> {
  static constexpr int64_t value = 0;
};

static inline bool find_physical_dim_order(
    c10::ArrayRef<int64_t> strides,
    int64_t element_size,
    c10::SmallVector<int32_t, 8>& dim_order) {
  dim_order.resize(strides.size());
  for (const auto dim : c10::irange(strides.size())) {
    if (strides[dim] <= 0) {
      return false;
    }
    dim_order[dim] = static_cast<int32_t>(dim);
  }
  std::sort(
      dim_order.begin(),
      dim_order.end(),
      [&](int32_t a, int32_t b) { return strides[a] < strides[b]; });
  if (strides[dim_order[0]] != element_size) {
    return false;
  }
  for (const auto dim : c10::irange(1, dim_order.size())) {
    if (strides[dim_order[dim]] <= strides[dim_order[dim - 1]]) {
      return false;
    }
  }
  return true;
}

static inline bool is_dense_run(
    c10::ArrayRef<int64_t> shape,
    c10::ArrayRef<int64_t> strides,
    c10::ArrayRef<int32_t> dims) {
  for (const auto dim : c10::irange(1, dims.size())) {
    int32_t previous = dims[dim - 1];
    if (shape[previous] * strides[previous] != strides[dims[dim]]) {
      return false;
    }
  }
  return true;
}

static inline int64_t shape_product(
    c10::ArrayRef<int64_t> shape,
    c10::ArrayRef<int32_t> dims) {
  int64_t product = 1;
  for (int32_t dim : dims) {
    if (shape[dim] > std::numeric_limits<int32_t>::max() / product) {
      return 0;
    }
    product *= shape[dim];
  }
  return product;
}

__device__ inline uint32_t swizzled_shared_offset(
    uint32_t byte_offset,
    uint32_t swizzle_mask) {
  return byte_offset ^ ((byte_offset & swizzle_mask) >> 3);
}

template <int64_t element_size>
__device__ inline void decode_virtual_coords(
    int warp_iteration,
    int lane,
    int32_t* local_coords) {
  if constexpr (element_size == 4) {
    local_coords[0] = lane;
    local_coords[1] = warp_iteration ^ (lane >> 3);
  } else if constexpr (element_size == 2) {
    local_coords[0] = lane;
    local_coords[1] = warp_iteration ^ ((lane >> 4) << 1);
  } else if constexpr (element_size == 1) {
    local_coords[0] = lane | ((warp_iteration & 1) << 5);
    local_coords[1] = warp_iteration >> 1;
  } else {
    static_assert(element_size == 8);
    const int local_idx = warp_iteration * 32 + lane;
    local_coords[0] = local_idx & 15;
    local_coords[1] = local_idx >> 4;
  }
}

template <int ninputs>
__device__ inline uint32_t input_shared_offset(
    const TmaElementwiseMeta<ninputs>& meta,
    int input,
    const int32_t* local_coords) {
  const int32_t* coord_stride =
      meta.input_coord_stride_bytes + input * kVirtualRank;
  const uint32_t byte_offset =
      static_cast<uint32_t>(local_coords[0]) *
          static_cast<uint32_t>(coord_stride[0]) +
      static_cast<uint32_t>(local_coords[1]) *
          static_cast<uint32_t>(coord_stride[1]);
  return swizzled_shared_offset(
      byte_offset, static_cast<uint32_t>(meta.swizzle_masks[input]));
}

template <typename func_t, typename array_t, int ninputs>
C10_LAUNCH_BOUNDS_1(kMaxThreadsPerBlock)
__global__ void tma_elementwise_kernel(
    func_t f,
    array_t data,
    __grid_constant__ const TmaElementwiseMaps<ninputs> maps,
    __grid_constant__ const TmaElementwiseMeta<ninputs> meta) {
#if __CUDA_ARCH__ >= 900
  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  constexpr int64_t max_input_element_size =
      max_arg_element_size_impl<traits, ninputs>::value;
  extern __shared__ char smem_base[];
  char* smem = smem_base;
  uint32_t smem32 = __cvta_generic_to_shared(smem);
  smem32 = (smem32 + kSmemAlignment - 1) & ~(kSmemAlignment - 1);
  asm("" : "+r"(smem32));
  smem = static_cast<char*>(__cvta_shared_to_generic(smem32));
  auto* barriers = reinterpret_cast<uint64_t*>(smem);

  const int32_t block_tile =
      static_cast<int32_t>(blockIdx.x) * kTilesPerBlock;
  const int32_t block_tile0 = block_tile << meta.box_shift[0];
  const int32_t block_tile1 =
      static_cast<int32_t>(blockIdx.y) << meta.box_shift[1];
  const int32_t batch = static_cast<int32_t>(blockIdx.z);
  const int32_t remaining_stages = meta.tiles_per_dim[0] - block_tile;
  const int32_t valid_stages =
      remaining_stages < kTilesPerBlock
      ? remaining_stages
      : kTilesPerBlock;
  const bool full_stages =
      remaining_stages >= kTilesPerBlock;

  auto issue_tma = [&](int stage, int32_t tile0) {
    char* stage_smem = smem + meta.tile_smem_offset +
        stage * meta.stage_smem_bytes;
    for (int input = 0; input < ninputs; input++) {
      int32_t tile_coords[kVirtualRank];
      if (meta.input_transposed[input]) {
        tile_coords[0] = block_tile1;
        tile_coords[1] =
            tile0 + batch * meta.virtual_shape[0];
      } else {
        tile_coords[0] = tile0;
        tile_coords[1] =
            block_tile1 + batch * meta.virtual_shape[1];
      }
      ::cuda::ptx::cp_async_bulk_tensor(
          ::cuda::ptx::space_shared,
          ::cuda::ptx::space_global,
          stage_smem + meta.smem_offsets[input],
          &maps.maps[input],
          tile_coords,
          &barriers[stage]);
    }
    ::cuda::ptx::mbarrier_arrive_expect_tx(
        ::cuda::ptx::sem_release,
        ::cuda::ptx::scope_cta,
        ::cuda::ptx::space_shared,
        &barriers[stage],
        static_cast<uint32_t>(meta.total_tma_bytes));
  };

  if (threadIdx.x == 0) {
    ::cuda::ptx::mbarrier_init(&barriers[0], 1);
    ::cuda::ptx::fence_proxy_async(::cuda::ptx::space_shared);
    issue_tma(0, block_tile0);
    if (full_stages) {
#pragma unroll
      for (int stage = 1; stage < kTilesPerBlock; stage++) {
        ::cuda::ptx::mbarrier_init(&barriers[stage], 1);
        ::cuda::ptx::fence_proxy_async(::cuda::ptx::space_shared);
        issue_tma(stage, block_tile0 + (stage << meta.box_shift[0]));
      }
    } else {
#pragma unroll
      for (int stage = 1; stage < kTilesPerBlock; stage++) {
        if (stage < valid_stages) {
          ::cuda::ptx::mbarrier_init(&barriers[stage], 1);
          ::cuda::ptx::fence_proxy_async(::cuda::ptx::space_shared);
          issue_tma(stage, block_tile0 + (stage << meta.box_shift[0]));
        }
      }
    }
  }
  __syncthreads();

  const int64_t base_output_offset =
      static_cast<int64_t>(batch) * meta.output_batch_stride_bytes +
      static_cast<int64_t>(block_tile0) *
          meta.output_coord_stride_bytes[0] +
      static_cast<int64_t>(block_tile1) *
          meta.output_coord_stride_bytes[1];
  char* output_block = data[0] + base_output_offset;
  std::array<int, ninputs> zero_strides{};
  auto consume_stage = [&](int stage) {
    while (!::cuda::ptx::mbarrier_try_wait_parity(
        ::cuda::ptx::sem_acquire,
        ::cuda::ptx::scope_cta,
        &barriers[stage],
        0)) {
    }

    char* stage_smem = smem + meta.tile_smem_offset +
        stage * meta.stage_smem_bytes;
    char* output_stage = output_block +
        static_cast<int64_t>(stage << meta.box_shift[0]) *
            meta.output_coord_stride_bytes[0];
    int warp_iteration =
        (static_cast<int>(threadIdx.x) >> 5) * kElementsPerThread;
    const int lane = static_cast<int>(threadIdx.x) & 31;
#pragma unroll
    for (int i = 0; i < kElementsPerThread; i++) {
      int32_t local_coords[kVirtualRank];
      decode_virtual_coords<max_input_element_size>(
          warp_iteration, lane, local_coords);
      std::array<char*, ninputs> input_ptrs;
#pragma unroll
      for (int input = 0; input < ninputs; input++) {
        uint32_t byte_offset =
            input_shared_offset<ninputs>(meta, input, local_coords);
        input_ptrs[input] =
            stage_smem + meta.smem_offsets[input] + byte_offset;
      }
      arg0_t result =
          invoke(f, input_ptrs.data(), zero_strides.data(), 0);
      const uint32_t output_offset =
          static_cast<uint32_t>(local_coords[0]) *
              static_cast<uint32_t>(meta.output_coord_stride_bytes[0]) +
          static_cast<uint32_t>(local_coords[1]) *
              static_cast<uint32_t>(meta.output_coord_stride_bytes[1]);
      *reinterpret_cast<arg0_t*>(output_stage + output_offset) =
          result;
      warp_iteration++;
    }
  };

  if (full_stages) {
#pragma unroll
    for (int stage = 0; stage < kTilesPerBlock; stage++) {
      consume_stage(stage);
    }
  } else {
#pragma unroll
    for (int stage = 0; stage < kTilesPerBlock; stage++) {
      if (stage < valid_stages) {
        consume_stage(stage);
      }
    }
  }
#endif
}

template <int ninputs, int64_t max_input_element_size>
static bool build_virtual_tma_metadata(
    TensorIteratorBase& iter,
    TmaElementwiseMeta<ninputs>& meta,
    TmaElementwiseMaps<ninputs>& maps,
    int& block_threads) {
  auto shape = iter.shape();
  const int ndim = iter.ndim();
  if (ndim < kVirtualRank) {
    return false;
  }
  for (const auto dim : c10::irange(ndim)) {
    if (shape[dim] <= 0 ||
        shape[dim] > std::numeric_limits<int32_t>::max()) {
      return false;
    }
  }

  c10::SmallVector<int32_t, 8> output_order;
  auto output_strides = iter.strides(0);
  int64_t output_element_size = iter.element_size(0);
  if (!find_physical_dim_order(
          output_strides, output_element_size, output_order)) {
    return false;
  }
  c10::SmallVector<int32_t, 8> output_position(ndim);
  for (const auto position : c10::irange(ndim)) {
    output_position[output_order[position]] =
        static_cast<int32_t>(position);
  }

  std::array<c10::SmallVector<int32_t, 8>, ninputs + 1> arg_orders;
  for (int arg = 0; arg < iter.ntensors(); arg++) {
    if (!is_aligned(iter.data_ptr(arg), 16)) {
      return false;
    }
    int64_t element_size = iter.element_size(arg);
    if (element_size != 1 && element_size != 2 && element_size != 4 &&
        element_size != 8) {
      return false;
    }
    auto strides = iter.strides(arg);
    c10::SmallVector<int32_t, 8> dim_order;
    if (!find_physical_dim_order(strides, element_size, dim_order)) {
      return false;
    }
    arg_orders[arg].resize(ndim);
    for (const auto position : c10::irange(ndim)) {
      arg_orders[arg][position] =
          output_position[dim_order[position]];
    }
  }

  constexpr int64_t geometry_element_size = max_input_element_size;

  int32_t box_shift;
  if (geometry_element_size == 1) {
    box_shift = 6;
    block_threads = 512;
  } else if (
      geometry_element_size == 2 || geometry_element_size == 4) {
    box_shift = 5;
    block_threads = 128;
  } else if (geometry_element_size == 8) {
    box_shift = 4;
    block_threads = 32;
  } else {
    return false;
  }
  const int32_t box = 1 << box_shift;
  if (static_cast<int64_t>(box) * box !=
      static_cast<int64_t>(block_threads) * kElementsPerThread) {
    return false;
  }

  for (int a_end = 1; a_end < ndim; a_end++) {
    for (int b_end = a_end + 1; b_end <= ndim; b_end++) {
      c10::ArrayRef<int32_t> a_dims(output_order.data(), a_end);
      c10::ArrayRef<int32_t> b_dims(
          output_order.data() + a_end, b_end - a_end);
      c10::ArrayRef<int32_t> c_dims(
          output_order.data() + b_end, ndim - b_end);
      std::array<bool, ninputs + 1> transposed{};
      bool order_matches = true;

      for (int arg = 0; arg < iter.ntensors(); arg++) {
        bool identity = true;
        bool swapped = true;
        for (const auto position : c10::irange(ndim)) {
          identity &=
              arg_orders[arg][position] == static_cast<int32_t>(position);
          int32_t expected;
          if (position < b_end - a_end) {
            expected = a_end + position;
          } else if (position < b_end) {
            expected = position - (b_end - a_end);
          } else {
            expected = position;
          }
          swapped &= arg_orders[arg][position] == expected;
        }
        if (arg == 0 ? !identity : (!identity && !swapped)) {
          order_matches = false;
          break;
        }
        transposed[arg] = !identity;
      }
      if (!order_matches) {
        continue;
      }

      int64_t shape_a = shape_product(shape, a_dims);
      int64_t shape_b = shape_product(shape, b_dims);
      int64_t batch_count = shape_product(shape, c_dims);
      if (shape_a == 0 || shape_b == 0 || batch_count == 0 ||
          shape_a * batch_count >
              std::numeric_limits<int32_t>::max() ||
          shape_b * batch_count >
              std::numeric_limits<int32_t>::max() ||
          shape_a % box != 0 || shape_b % box != 0) {
        continue;
      }

      bool dense = true;
      for (int arg = 0; arg < iter.ntensors(); arg++) {
        auto strides = iter.strides(arg);
        c10::ArrayRef<int32_t> first =
            transposed[arg] ? b_dims : a_dims;
        c10::ArrayRef<int32_t> second =
            transposed[arg] ? a_dims : b_dims;
        c10::SmallVector<int32_t, 8> second_and_batch(
            second.begin(), second.end());
        second_and_batch.append(c_dims.begin(), c_dims.end());
        if (!is_dense_run(shape, strides, first) ||
            !is_dense_run(shape, strides, second_and_batch)) {
          dense = false;
          break;
        }
      }
      if (!dense) {
        continue;
      }

      TmaElementwiseMeta<ninputs> candidate_meta{};
      TmaElementwiseMaps<ninputs> candidate_maps{};
      candidate_meta.virtual_shape[0] = static_cast<int32_t>(shape_a);
      candidate_meta.virtual_shape[1] = static_cast<int32_t>(shape_b);
      candidate_meta.batch_count = static_cast<int32_t>(batch_count);
      candidate_meta.box_shift[0] = box_shift;
      candidate_meta.box_shift[1] = box_shift;
      candidate_meta.tiles_per_dim[0] =
          static_cast<int32_t>(shape_a / box);
      candidate_meta.tiles_per_dim[1] =
          static_cast<int32_t>(shape_b / box);

      int64_t grid_x =
          (candidate_meta.tiles_per_dim[0] + kTilesPerBlock - 1) /
          kTilesPerBlock;
      if (grid_x <= 0 ||
          grid_x > std::numeric_limits<int32_t>::max()) {
        continue;
      }

      int64_t output_stride_a = output_strides[a_dims.front()];
      int64_t output_stride_b = output_strides[b_dims.front()];
      int64_t output_batch_stride =
          c_dims.empty() ? 0 : output_strides[c_dims.front()];
      int64_t max_output_local_offset =
          (box - 1) * output_stride_a + (box - 1) * output_stride_b;
      if (output_stride_a > std::numeric_limits<int32_t>::max() ||
          output_stride_b > std::numeric_limits<int32_t>::max() ||
          output_batch_stride > std::numeric_limits<int32_t>::max() ||
          output_stride_b % 16 != 0 ||
          (output_batch_stride != 0 && output_batch_stride % 16 != 0) ||
          max_output_local_offset >
              std::numeric_limits<uint32_t>::max() ||
          shape_a * output_element_size % 16 != 0) {
        continue;
      }
      candidate_meta.output_coord_stride_bytes[0] =
          static_cast<int32_t>(output_stride_a);
      candidate_meta.output_coord_stride_bytes[1] =
          static_cast<int32_t>(output_stride_b);
      candidate_meta.output_batch_stride_bytes =
          static_cast<int32_t>(output_batch_stride);

      std::array<int32_t, ninputs> input_element_size{};
      std::array<int32_t, ninputs> input_row_stride_bytes{};
      bool descriptors_valid = true;
      int64_t smem_offset = 0;
      int64_t total_tma_bytes = 0;
      for (int input = 0; input < ninputs; input++) {
        int arg = input + 1;
        int64_t element_size = iter.element_size(arg);
        input_element_size[input] = static_cast<int32_t>(element_size);
        bool input_transposed = transposed[arg];
        candidate_meta.input_transposed[input] = input_transposed;
        int64_t fast_extent =
            (input_transposed ? shape_b : shape_a) * element_size;
        int64_t inner_bytes = static_cast<int64_t>(box) * element_size;
        if (fast_extent % 16 != 0 || inner_bytes > 128) {
          descriptors_valid = false;
          break;
        }
        if (!input_transposed) {
          candidate_meta.swizzle_bytes[input] = 0;
        } else if (inner_bytes <= 32) {
          candidate_meta.swizzle_bytes[input] = 32;
        } else if (inner_bytes <= 64) {
          candidate_meta.swizzle_bytes[input] = 64;
        } else {
          candidate_meta.swizzle_bytes[input] = 128;
        }
        if (candidate_meta.swizzle_bytes[input] != 0 &&
            !is_aligned(iter.data_ptr(arg), 128)) {
          descriptors_valid = false;
          break;
        }
        candidate_meta.swizzle_masks[input] =
            candidate_meta.swizzle_bytes[input] == 0
            ? 0
            : (candidate_meta.swizzle_bytes[input] - 16) << 3;
        input_row_stride_bytes[input] =
            candidate_meta.swizzle_bytes[input] == 0
            ? static_cast<int32_t>(inner_bytes)
            : candidate_meta.swizzle_bytes[input];
        if (input_transposed) {
          candidate_meta.input_coord_stride_bytes[
              input * kVirtualRank] = input_row_stride_bytes[input];
          candidate_meta.input_coord_stride_bytes[
              input * kVirtualRank + 1] =
              static_cast<int32_t>(element_size);
        } else {
          candidate_meta.input_coord_stride_bytes[
              input * kVirtualRank] =
              static_cast<int32_t>(element_size);
          candidate_meta.input_coord_stride_bytes[
              input * kVirtualRank + 1] = input_row_stride_bytes[input];
        }

        smem_offset = align_up(smem_offset, kSmemAlignment);
        candidate_meta.smem_offsets[input] =
            static_cast<int32_t>(smem_offset);
        smem_offset +=
            static_cast<int64_t>(input_row_stride_bytes[input]) * box;
        total_tma_bytes +=
            static_cast<int64_t>(box) * box * element_size;
      }
      if (!descriptors_valid) {
        continue;
      }

      candidate_meta.tile_smem_offset = static_cast<int32_t>(
          align_up(
              kTilesPerBlock * static_cast<int64_t>(sizeof(uint64_t)),
              kSmemAlignment));
      candidate_meta.stage_smem_bytes =
          static_cast<int32_t>(align_up(smem_offset, kSmemAlignment));
      candidate_meta.total_tma_bytes =
          static_cast<int32_t>(total_tma_bytes);
      int64_t smem_bytes =
          candidate_meta.tile_smem_offset +
          kTilesPerBlock *
              static_cast<int64_t>(candidate_meta.stage_smem_bytes) +
          kSmemAlignment;
      if (smem_bytes > std::numeric_limits<int32_t>::max()) {
        continue;
      }
      candidate_meta.smem_bytes = static_cast<int32_t>(smem_bytes);

      constexpr cuuint32_t element_strides[kVirtualRank] = {1, 1};
      for (int input = 0; input < ninputs; input++) {
        int arg = input + 1;
        CUtensorMapDataType data_type;
        if (!tensor_map_data_type(
                input_element_size[input], &data_type)) {
          descriptors_valid = false;
          break;
        }
        bool input_transposed =
            candidate_meta.input_transposed[input] != 0;
        c10::ArrayRef<int32_t> second =
            input_transposed ? a_dims : b_dims;
        int64_t second_stride = iter.strides(arg)[second.front()];
        int64_t second_extent =
            (input_transposed ? shape_a : shape_b) * batch_count;
        if (second_stride % 16 != 0 ||
            second_stride >= (1LL << 40)) {
          descriptors_valid = false;
          break;
        }
        cuuint64_t global_dim[kVirtualRank] = {
            static_cast<cuuint64_t>(
                input_transposed ? shape_b : shape_a),
            static_cast<cuuint64_t>(second_extent)};
        cuuint64_t global_strides[kVirtualRank - 1] = {
            static_cast<cuuint64_t>(second_stride)};
        cuuint32_t box_dim[kVirtualRank] = {
            static_cast<cuuint32_t>(box),
            static_cast<cuuint32_t>(box)};
        CUresult result =
            at::globalContext().getNVRTC().cuTensorMapEncodeTiled(
                &candidate_maps.maps[input],
                data_type,
                kVirtualRank,
                iter.data_ptr(arg),
                global_dim,
                global_strides,
                box_dim,
                element_strides,
                CU_TENSOR_MAP_INTERLEAVE_NONE,
                tensor_map_swizzle(
                    candidate_meta.swizzle_bytes[input]),
                CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (result != CUDA_SUCCESS) {
          descriptors_valid = false;
          break;
        }
      }
      if (descriptors_valid) {
        meta = candidate_meta;
        maps = candidate_maps;
        return true;
      }
    }
  }
  return false;
}

template <typename func_t, typename array_t>
static bool launch_virtual_tma_elementwise_kernel(
    TensorIteratorBase& iter,
    const func_t& f,
    array_t data) {
  using traits = function_traits<func_t>;
  constexpr int ninputs = traits::arity;
  constexpr int64_t max_input_element_size =
      max_arg_element_size_impl<traits, ninputs>::value;
  if constexpr (
      ninputs == 0 ||
      (max_input_element_size != 1 &&
       max_input_element_size != 2 &&
       max_input_element_size != 4 &&
       max_input_element_size != 8)) {
    return false;
  } else {
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaDeviceProp* prop =
        at::cuda::getDeviceProperties(stream.device().index());
    if (prop->major != 10 || prop->minor != 7) {
      return false;
    }
    static const bool disabled = c10::utils::check_env(
        "PYTORCH_TMA_ELEMENTWISE_DISABLE").value_or(false);
    if (disabled) {
      return false;
    }
    auto kernel = tma_elementwise_kernel<func_t, array_t, ninputs>;
    cudaFuncAttributes attributes{};
    AT_CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    if (attributes.ptxVersion < 90) {
      return false;
    }

    TmaElementwiseMeta<ninputs> meta{};
    TmaElementwiseMaps<ninputs> maps{};
    int block_threads = 0;
    if (!build_virtual_tma_metadata<
            ninputs,
            max_input_element_size>(iter, meta, maps, block_threads)) {
      return false;
    }
    if (block_threads > kMaxThreadsPerBlock ||
        static_cast<size_t>(meta.smem_bytes) >
            prop->sharedMemPerBlockOptin) {
      return false;
    }

    if (meta.smem_bytes > 0xc000) {
      AT_CUDA_CHECK(cudaFuncSetAttribute(
          kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          meta.smem_bytes));
    }

    int grid_x =
        (meta.tiles_per_dim[0] + kTilesPerBlock - 1) /
        kTilesPerBlock;
    int grid_y = meta.tiles_per_dim[1];
    int grid_z = meta.batch_count;
    if (grid_x > prop->maxGridSize[0] ||
        grid_y > prop->maxGridSize[1] ||
        grid_z > prop->maxGridSize[2]) {
      return false;
    }

    dim3 grid(grid_x, grid_y, grid_z);
    kernel<<<grid, block_threads, meta.smem_bytes, stream>>>(
        f, data, maps, meta);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
  }
}

template <typename func_t, typename array_t>
static bool launch_tma_elementwise_kernel(
    TensorIteratorBase& iter,
    const func_t& f,
    array_t data) {
  using traits = function_traits<func_t>;
  constexpr int ninputs = traits::arity;
  if (iter.ndim() < kVirtualRank ||
      !iter.can_use_32bit_indexing() ||
      iter.noutputs() != 1 || iter.ninputs() != ninputs ||
      iter.element_size(0) != sizeof(typename traits::result_type)) {
    return false;
  }
  return launch_virtual_tma_elementwise_kernel(iter, f, data);
}

} // namespace tma_elementwise
#endif

#ifdef USE_ROCM
template <int nt, int vt, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_manual_unroll(int N, func_t f) {
  int tid = threadIdx.x;
  constexpr int nv = nt * vt;
  int idx = nv * blockIdx.x + tid;
  if ((idx + nt*(vt-1)) < N) {
    f(idx, true);
  } else {
#pragma unroll
    for (int i = 0; i < vt; i++) {
      if (idx < N) {
        f(idx, false);
        idx += nt;
      }
    }
  }
}

template <int nt, int vt, typename func_t>
static void launch_legacy_kernel_manual_unroll(int64_t N, const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_manual_unroll<nt, vt, func_t><<<grid, block, 0, stream>>>(N, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#endif

template <typename traits, typename func_t, typename index_t, size_t... INDEX>
C10_HOST_DEVICE typename traits::result_type invoke_impl(
    const func_t& f,
    char* const C10_RESTRICT data[],
    const index_t strides[],
    int i,
    std::index_sequence<INDEX...>) {
  (void)strides;
  (void)i;
  return f(c10::load<typename traits::template arg<INDEX>::type>(
      data[INDEX] + i * strides[INDEX])...);
}

template <
    typename func_t,
    typename index_t,
    typename traits>
C10_HOST_DEVICE typename traits::result_type invoke(
    const func_t& f,
    char* const C10_RESTRICT data[],
    const index_t strides[],
    int i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return invoke_impl<traits>(f, data, strides, i, Indices{});
}

template <typename traits, typename func_t, typename index_t, size_t... I>
C10_HOST_DEVICE typename traits::result_type invoke_impl(
    const func_t& f,
    char* const C10_RESTRICT data[],
    const index_t strides[],
    const ScalarType dtypes[],
    int i,
    std::index_sequence<I...>) {
  (void)strides;
  (void)i;
  return f(c10::fetch_and_cast<typename traits::template arg<I>::type>(
      dtypes[I], data[I] + i * strides[I])...);
}

template <
    typename func_t,
    typename index_t,
    typename traits>
C10_HOST_DEVICE typename traits::result_type invoke(
    const func_t& f,
    char* const C10_RESTRICT data[],
    const index_t strides[],
    const ScalarType dtypes[],
    int i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return invoke_impl<traits>(f, data, strides, dtypes, i, Indices{});
}

template <typename func_t>
void gpu_kernel_impl_nocast(TensorIteratorBase& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  constexpr int ntensors = traits::arity + 1;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);

  std::array<char*, ntensors> data;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  int64_t numel = iter.numel();

  bool contiguous = iter.is_contiguous();

  if (contiguous) {
    return launch_vectorized_kernel(numel, f, data);
  }
#if AT_CUDA_TMA_ELEMENTWISE_ENABLED
  if (tma_elementwise::launch_tma_elementwise_kernel(iter, f, data)) {
    return;
  }
#endif
  auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);
#ifndef USE_ROCM
  constexpr int unroll_factor = sizeof(arg0_t) >= 4 ? 2 : 4;
  launch_legacy_kernel<128, unroll_factor>(numel, [=] GPU_LAMBDA(int idx) {
    auto offsets = offset_calc.get(idx);
    arg0_t* out = (arg0_t*)(data[0] + offsets[0]);
    *out = invoke(f, &data[1], &offsets[1], 1);
  });
#else
  constexpr int unroll_factor = sizeof(arg0_t) >= 4 ? 4 : 8;
  constexpr int grp_sz = 128;
  launch_legacy_kernel_manual_unroll<grp_sz, unroll_factor>(numel, [=] GPU_LAMBDA(int idx, bool unrl) {
    if (unrl) {
      if constexpr (unroll_factor == 4) {
        auto offsets0 = offset_calc.get(idx);
        auto offsets1 = offset_calc.get(idx+grp_sz);
        auto offsets2 = offset_calc.get(idx+grp_sz*2);
        auto offsets3 = offset_calc.get(idx+grp_sz*3);
        arg0_t* out0 = (arg0_t*)(data[0] + offsets0[0]);
        arg0_t* out1 = (arg0_t*)(data[0] + offsets1[0]);
        arg0_t* out2 = (arg0_t*)(data[0] + offsets2[0]);
        arg0_t* out3 = (arg0_t*)(data[0] + offsets3[0]);
        auto tmp0 = invoke(f, &data[1], &offsets0[1], 1);
        auto tmp1 = invoke(f, &data[1], &offsets1[1], 1);
        auto tmp2 = invoke(f, &data[1], &offsets2[1], 1);
        auto tmp3 = invoke(f, &data[1], &offsets3[1], 1);
        *out0 = tmp0;
        *out1 = tmp1;
        *out2 = tmp2;
        *out3 = tmp3;
      } else {
        auto offsets0 = offset_calc.get(idx);
        auto offsets1 = offset_calc.get(idx+grp_sz);
        auto offsets2 = offset_calc.get(idx+grp_sz*2);
        auto offsets3 = offset_calc.get(idx+grp_sz*3);
        auto offsets4 = offset_calc.get(idx+grp_sz*4);
        auto offsets5 = offset_calc.get(idx+grp_sz*5);
        auto offsets6 = offset_calc.get(idx+grp_sz*6);
        auto offsets7 = offset_calc.get(idx+grp_sz*7);
        arg0_t* out0 = (arg0_t*)(data[0] + offsets0[0]);
        arg0_t* out1 = (arg0_t*)(data[0] + offsets1[0]);
        arg0_t* out2 = (arg0_t*)(data[0] + offsets2[0]);
        arg0_t* out3 = (arg0_t*)(data[0] + offsets3[0]);
        arg0_t* out4 = (arg0_t*)(data[0] + offsets4[0]);
        arg0_t* out5 = (arg0_t*)(data[0] + offsets5[0]);
        arg0_t* out6 = (arg0_t*)(data[0] + offsets6[0]);
        arg0_t* out7 = (arg0_t*)(data[0] + offsets7[0]);
        auto tmp0 = invoke(f, &data[1], &offsets0[1], 1);
        auto tmp1 = invoke(f, &data[1], &offsets1[1], 1);
        auto tmp2 = invoke(f, &data[1], &offsets2[1], 1);
        auto tmp3 = invoke(f, &data[1], &offsets3[1], 1);
        auto tmp4 = invoke(f, &data[1], &offsets4[1], 1);
        auto tmp5 = invoke(f, &data[1], &offsets5[1], 1);
        auto tmp6 = invoke(f, &data[1], &offsets6[1], 1);
        auto tmp7 = invoke(f, &data[1], &offsets7[1], 1);
        *out0 = tmp0;
        *out1 = tmp1;
        *out2 = tmp2;
        *out3 = tmp3;
        *out4 = tmp4;
        *out5 = tmp5;
        *out6 = tmp6;
        *out7 = tmp7;
      }
    } else {
      auto offsets = offset_calc.get(idx);
      arg0_t* out = (arg0_t*)(data[0] + offsets[0]);
      *out = invoke(f, &data[1], &offsets[1], 1);
    }
  });
#endif
}

#ifdef USE_ROCM
namespace {
template <
    typename TupleLike,
    typename FirstParamTy,
    typename SecondParamTy,
    size_t arity,
    size_t arg_num = 0>
struct check_binary_functor_types_for_specialization {
  constexpr static inline bool check() {
    if constexpr (arity != 2)
      return false;
    if constexpr (arg_num == 0) {
      using SelectedType = std::tuple_element_t<arg_num, TupleLike>;
      if constexpr (std::is_same_v<FirstParamTy, SelectedType>)
        return check_binary_functor_types_for_specialization<
            TupleLike,
            FirstParamTy,
            SecondParamTy,
            arity,
            arg_num + 1>::check();
    } else if constexpr (arg_num == 1) {
      using SelectedType2 = std::tuple_element_t<arg_num, TupleLike>;
      if constexpr (std::is_same_v<SecondParamTy, SelectedType2>)
        return check_binary_functor_types_for_specialization<
            TupleLike,
            FirstParamTy,
            SecondParamTy,
            arity,
            arg_num + 1>::check();
    }
    return false;
  }
};

// Bottom case: if we got this far, assume correct type matching except
// when there are no arguments (arity == 0).
template <
    typename TupleLike,
    typename FirstParamTy,
    typename SecondParamTy,
    size_t arity>
struct check_binary_functor_types_for_specialization<
    TupleLike,
    FirstParamTy,
    SecondParamTy,
    arity,
    arity> {
  constexpr static inline bool check() {
    if constexpr (arity != 0)
      return true;
    return false;
  }
};

template <typename TupleLike, typename FirstParamTy, typename SecondParamTy>
struct check_binary_functor_types_for_specialization<
    TupleLike,
    FirstParamTy,
    SecondParamTy,
    0,
    0> {
  constexpr static inline bool check() {
    return false;
  }
};

// The following is a list of type specializations for vectorized_templated
// elementwise kernel. The three types refer to runtime types of the output
// tensor, first tensor argument, and the second tensor argument used for a
// binary functor.
constexpr std::array rt_binary_specializations = {
    std::array<c10::ScalarType, 3>(
        {c10::CppTypeToScalarType<float>::value,
         c10::CppTypeToScalarType<float>::value,
         c10::CppTypeToScalarType<BFloat16>::value}),
    std::array<c10::ScalarType, 3>(
        {c10::CppTypeToScalarType<float>::value,
         c10::CppTypeToScalarType<BFloat16>::value,
         c10::CppTypeToScalarType<float>::value}),
    std::array<c10::ScalarType, 3>(
        {c10::CppTypeToScalarType<BFloat16>::value,
         c10::CppTypeToScalarType<BFloat16>::value,
         c10::CppTypeToScalarType<float>::value}),
    std::array<c10::ScalarType, 3>(
        {c10::CppTypeToScalarType<float>::value,
         c10::CppTypeToScalarType<float>::value,
         c10::CppTypeToScalarType<Half>::value}),
    std::array<c10::ScalarType, 3>(
        {c10::CppTypeToScalarType<float>::value,
         c10::CppTypeToScalarType<Half>::value,
         c10::CppTypeToScalarType<float>::value}),
    std::array<c10::ScalarType, 3>(
        {c10::CppTypeToScalarType<Half>::value,
         c10::CppTypeToScalarType<Half>::value,
         c10::CppTypeToScalarType<float>::value})};

bool check_binary_rt_types_for_specialization(TensorIteratorBase& iter) {
  if (iter.ninputs() != 2)
    return false;
  for (auto spec : rt_binary_specializations)
    if (iter.dtype(0) == spec[0] && iter.input_dtype(0) == spec[1] &&
        iter.input_dtype(1) == spec[2])
      return true;
  return false;
}

template <int arg_index>
struct type_specialized_kernel_launcher {
  template <
      typename func_t,
      typename array_t,
      typename inp_calc_t,
      typename out_calc_t,
      typename loader_t,
      typename storer_t>
  static void apply(
      ScalarType ret_t,
      ScalarType arg0_t,
      ScalarType arg1_t,
      int64_t numel,
      func_t f,
      array_t data,
      inp_calc_t input_offset_calculator,
      out_calc_t output_offset_calculator,
      loader_t loader,
      storer_t storer) {
    constexpr ScalarType sret_t = rt_binary_specializations[arg_index][0];
    constexpr ScalarType sarg0_t = rt_binary_specializations[arg_index][1];
    constexpr ScalarType sarg1_t = rt_binary_specializations[arg_index][2];
    if (ret_t == sret_t && arg0_t == sarg0_t && arg1_t == sarg1_t) {
      using cret_t = c10::impl::ScalarTypeToCPPTypeT<sret_t>;
      using carg0_t = c10::impl::ScalarTypeToCPPTypeT<sarg0_t>;
      using carg1_t = c10::impl::ScalarTypeToCPPTypeT<sarg1_t>;
      launch_vectorized_templated_kernel<
          func_t,
          array_t,
          inp_calc_t,
          out_calc_t,
          loader_t,
          storer_t,
          cret_t,
          carg0_t,
          carg1_t>(
          numel,
          f,
          data,
          input_offset_calculator,
          output_offset_calculator,
          loader,
          storer);
    }
  }
};

template <int arg_index>
struct type_specialized_broadcast_kernel_launcher {
  template <
      typename func_t,
      typename array_t,
      typename dtypes_t,
      typename calc_t>
  static void apply(
      int64_t numel,
      func_t f,
      array_t data,
      dtypes_t dtypes,
      calc_t offset_calc) {
        using traits = function_traits<func_t>;
        using ret_t = typename traits::result_type;
        using arg0_t = typename traits::template arg<0>::type;
        using arg1_t = typename traits::template arg<1>::type;
        if (dtypes[0] == rt_binary_specializations[arg_index][0] &&
          dtypes[1] == rt_binary_specializations[arg_index][1] &&
          dtypes[2] == rt_binary_specializations[arg_index][2]) {
            using ret_cpp_t = c10::impl::ScalarTypeToCPPTypeT<rt_binary_specializations[arg_index][0]>;
            using arg0_cpp_t = c10::impl::ScalarTypeToCPPTypeT<rt_binary_specializations[arg_index][1]>;
            using arg1_cpp_t = c10::impl::ScalarTypeToCPPTypeT<rt_binary_specializations[arg_index][2]>;
            constexpr int grp_sz = 128;
            launch_legacy_kernel_manual_unroll<grp_sz, 4>(numel, [=] GPU_LAMBDA(int idx, bool unrl) {
              if (unrl) {
                auto offsets0 = offset_calc.get(idx);
                auto offsets1 = offset_calc.get(idx + grp_sz);
                auto offsets2 = offset_calc.get(idx + grp_sz * 2);
                auto offsets3 = offset_calc.get(idx + grp_sz * 3);
                void* out0 = data[0] + offsets0[0];
                void* out1 = data[0] + offsets1[0];
                void* out2 = data[0] + offsets2[0];
                void* out3 = data[0] + offsets3[0];
                auto u = c10::load<arg0_cpp_t>(data[1] + offsets0[1]);
                auto v = c10::load<arg1_cpp_t>(data[2] + offsets0[2]);
                ret_t result0 = f(c10::convert<arg0_t>(u), c10::convert<arg1_t>(v));
                auto u1 = c10::load<arg0_cpp_t>(data[1] + offsets1[1]);
                auto v1 = c10::load<arg1_cpp_t>(data[2]+ offsets1[2]);
                ret_t result1 = f(c10::convert<arg0_t>(u1), c10::convert<arg1_t>(v1));
                auto u2 = c10::load<arg0_cpp_t>(data[1] + offsets2[1]);
                auto v2 = c10::load<arg1_cpp_t>(data[2] + offsets2[2]);
                ret_t result2 = f(c10::convert<arg0_t>(u2), c10::convert<arg1_t>(v2));
                auto u3 = c10::load<arg0_cpp_t>(data[1] + offsets3[1]);
                auto v3 = c10::load<arg1_cpp_t>(data[2] + offsets3[2]);
                ret_t result3 = f(c10::convert<arg0_t>(u3), c10::convert<arg1_t>(v3));
                *(ret_cpp_t*)out0 = c10::convert<ret_cpp_t>(result0);
                *(ret_cpp_t*)out1 = c10::convert<ret_cpp_t>(result1);
                *(ret_cpp_t*)out2 = c10::convert<ret_cpp_t>(result2);
                *(ret_cpp_t*)out3 = c10::convert<ret_cpp_t>(result3);
              } else {
                auto offsets = offset_calc.get(idx);
                void* out = data[0] + offsets[0];
                auto u = c10::load<arg0_cpp_t>(data[1] + offsets[1]);
                auto v = c10::load<arg1_cpp_t>(data[2] + offsets[2]);
                ret_t result = f(c10::convert<arg0_t>(u), c10::convert<arg1_t>(v));
                *(ret_cpp_t*)out = c10::convert<ret_cpp_t>(result);
              }
            });
        }
      }
};

} // namespace
#endif

template <typename func_t>
void gpu_kernel_impl(TensorIteratorBase& iter, const func_t& f) {
  if (!needs_dynamic_casting<func_t>::check(iter)) {
    return gpu_kernel_impl_nocast(iter, f);
  }
  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  constexpr int ntensors = traits::arity + 1;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);

  std::array<char*, ntensors> data;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  int64_t numel = iter.numel();

  bool contiguous = iter.is_contiguous();

  if (contiguous) {
#ifdef USE_ROCM
    // Attempt to call specialized vectorized elementwise kernel
    // that enables interleaving.
    if (check_binary_rt_types_for_specialization(iter) &&
        memory::can_vectorize_up_to<func_t>(data) > 1) {
      // constexpr to reduce the amount of kernels generated for
      // vectorized templated elementwise and limit which functors are actually
      // applied to the load and store at compile time.
      using func_tuple = typename traits::ArgsTuple;
      if constexpr (
          std::is_same_v<float, arg0_t> && traits::arity == 2 &&
          check_binary_functor_types_for_specialization<
              func_tuple,
              float,
              float,
              traits::arity,
              /*arg_num=*/0>::check()) {
        // If we got here, we know we are in one of the specialized cases. We
        // need to translate the runtime type to a statically known type. This
        // is effectively hoisting to the host the switch over runtime type in
        // the kernel in fetch_and_cast. Loader, storer, offset calculators are
        // only needed for the reminder loop.
        auto input_offset_calculator = TrivialOffsetCalculator<traits::arity>();
        auto output_offset_calculator = TrivialOffsetCalculator<1>();
        auto loader = memory::LoadWithCast<traits::arity>(iter);
        auto storer = memory::StoreWithCast<1>(iter);
        memory::detail::static_unroll<
            type_specialized_kernel_launcher,
            rt_binary_specializations.size()>::
            with_args(
                iter.dtype(0),
                iter.input_dtype(0),
                iter.input_dtype(1),
                numel,
                f,
                data,
                input_offset_calculator,
                output_offset_calculator,
                loader,
                storer);
        return;
      }
    }
    std::array<ScalarType, ntensors> dtypes;
    auto inner_strides = iter.get_inner_strides();
    std::array<int, ntensors> strides;
    for (int i = 0; i < ntensors; i++) {
      dtypes[i] = iter.dtype(i);
      strides[i] = inner_strides[i];
    }
    constexpr int grp_sz = 128;
    launch_legacy_kernel_manual_unroll<grp_sz, 4>(numel, [=] GPU_LAMBDA(int idx, bool unrl) {
      if (unrl) {
        void* out0 = data[0] + strides[0] * idx;
        void* out1 = data[0] + strides[0] * (idx + grp_sz);
        void* out2 = data[0] + strides[0] * (idx + grp_sz * 2);
        void* out3 = data[0] + strides[0] * (idx + grp_sz * 3);
        arg0_t result0 = invoke(f, &data[1], &strides[1], &dtypes[1], idx);
        arg0_t result1 = invoke(f, &data[1], &strides[1], &dtypes[1], (idx + grp_sz));
        arg0_t result2 = invoke(f, &data[1], &strides[1], &dtypes[1], (idx + grp_sz * 2));
        arg0_t result3 = invoke(f, &data[1], &strides[1], &dtypes[1], (idx + grp_sz * 3));
        c10::cast_and_store<arg0_t>(dtypes[0], out0, result0);
        c10::cast_and_store<arg0_t>(dtypes[0], out1, result1);
        c10::cast_and_store<arg0_t>(dtypes[0], out2, result2);
        c10::cast_and_store<arg0_t>(dtypes[0], out3, result3);
      } else {
        void* out = data[0] + strides[0] * idx;
        arg0_t result = invoke(f, &data[1], &strides[1], &dtypes[1], idx);
        c10::cast_and_store<arg0_t>(dtypes[0], out, result);
      }
    });
#else
    auto loader = memory::LoadWithCast<traits::arity>(iter);
    auto storer = memory::StoreWithCast<1>(iter);
    auto input_offset_calculator = TrivialOffsetCalculator<traits::arity>();
    auto output_offset_calculator = TrivialOffsetCalculator<1>();
    launch_unrolled_kernel(
        numel,
        f,
        data,
        input_offset_calculator,
        output_offset_calculator,
        loader,
        storer);
#endif
  } else {
    std::array<ScalarType, ntensors> dtypes;
    for (int i = 0; i < ntensors; i++) {
      dtypes[i] = iter.dtype(i);
    }
    auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);
#ifdef USE_ROCM
    if (check_binary_rt_types_for_specialization(iter)) {
      // constexpr to reduce the amount of kernels generated for
      // broadcast elementwise with mexed dtypes and limit which functors are actually
      // applied to the load and store at compile time.
      using func_tuple = typename traits::ArgsTuple;
      if constexpr (
        std::is_same_v<float, arg0_t> && traits::arity == 2 &&
        check_binary_functor_types_for_specialization<
          func_tuple,
          float,
          float,
          traits::arity,
          /*arg_num=*/0>::check()) {
            memory::detail::static_unroll<
              type_specialized_broadcast_kernel_launcher,
              rt_binary_specializations.size()>::with_args(
                numel,
                f,
                data,
                dtypes,
                offset_calc
            );
            return;
      }
    }

    constexpr int grp_sz = 128;
    launch_legacy_kernel_manual_unroll<grp_sz, 4>(numel, [=] GPU_LAMBDA(int idx, bool unrl) {
      if (unrl) {
        auto offsets0 = offset_calc.get(idx);
        auto offsets1 = offset_calc.get(idx + grp_sz);
        auto offsets2 = offset_calc.get(idx + grp_sz * 2);
        auto offsets3 = offset_calc.get(idx + grp_sz * 3);
        void* out0 = data[0] + offsets0[0];
        void* out1 = data[0] + offsets1[0];
        void* out2 = data[0] + offsets2[0];
        void* out3 = data[0] + offsets3[0];
        arg0_t result0 = invoke(f, &data[1], &offsets0[1], &dtypes[1], 1);
        arg0_t result1 = invoke(f, &data[1], &offsets1[1], &dtypes[1], 1);
        arg0_t result2 = invoke(f, &data[1], &offsets2[1], &dtypes[1], 1);
        arg0_t result3 = invoke(f, &data[1], &offsets3[1], &dtypes[1], 1);
        c10::cast_and_store<arg0_t>(dtypes[0], out0, result0);
        c10::cast_and_store<arg0_t>(dtypes[0], out1, result1);
        c10::cast_and_store<arg0_t>(dtypes[0], out2, result2);
        c10::cast_and_store<arg0_t>(dtypes[0], out3, result3);
      } else {
        auto offsets = offset_calc.get(idx);
        void* out = data[0] + offsets[0];
        arg0_t result = invoke(f, &data[1], &offsets[1], &dtypes[1], 1);
        c10::cast_and_store<arg0_t>(dtypes[0], out, result);
      }
    });
#else
    launch_legacy_kernel<128, 4>(numel, [=] GPU_LAMBDA(int idx) {
      auto offsets = offset_calc.get(idx);
      void* out = data[0] + offsets[0];
      arg0_t result = invoke(f, &data[1], &offsets[1], &dtypes[1], 1);
      c10::cast_and_store<arg0_t>(dtypes[0], out, result);
    });
#endif
  }
}

} // namespace at::native

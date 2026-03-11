#pragma once

// Simplified CUDALoops.cuh for the standalone gpu_kernel instantiation library.
//
// Compiles each functor/type combination as a __device__ __noinline__ function
// that applies the functor N_UNROLL=4 times using wide loads/stores for
// maximal memory throughput, returning results in registers.
// No __global__ kernels are emitted.
//
// Instantiation is forced by an extended __host__ __device__ lambda that
// references simple_gpu_kernel during the device compilation pass.
// __attribute__((used)) + CUDA_SEPARABLE_COMPILATION (-dc) ensure the
// device function survives in the object file.
//
// TAG (= __LINE__ from the .cu call site) gives every instantiation a unique
// mangled name that cuobjdump can trace back to the source.

#include <array>
#include <tuple>
#include <type_traits>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/DynamicCast.h>
#include <c10/core/ScalarType.h>
#include <c10/macros/Macros.h>
#include <c10/util/TypeCast.h>

#ifdef __NVCC__
#define ASSERT_HOST_DEVICE_LAMBDA(type)                       \
  static_assert(                                              \
      __nv_is_extended_host_device_lambda_closure_type(type), \
      #type " must be a __host__ __device__ lambda")
#else
#define ASSERT_HOST_DEVICE_LAMBDA(type)
#endif

namespace at::native {

#ifndef GPU_KERNEL_UNROLL
#define GPU_KERNEL_UNROLL 4
#endif

static constexpr int N_UNROLL = GPU_KERNEL_UNROLL;

#if GPU_KERNEL_UNROLL <= 1

// -----------------------------------------------------------------------
// Scalar path: simple_gpu_kernel has the same signature as the lambda.
// -----------------------------------------------------------------------

template <int TAG, typename func_t, typename... Args>
__device__ __noinline__ __attribute__((used))
typename function_traits<func_t>::result_type simple_gpu_kernel(
    func_t f, Args... args) {
  return f(args...);
}

template <int TAG, typename traits, typename func_t, size_t... Is>
__device__ __forceinline__ void _instantiate_simple_kernel(
    const func_t& f, std::index_sequence<Is...>) {
  simple_gpu_kernel<TAG>(f, typename traits::template arg<Is>::type{}...);
}

#else // GPU_KERNEL_UNROLL > 1

// -----------------------------------------------------------------------
// AtomT: maps a byte size to the widest naturally-aligned load/store type
// -----------------------------------------------------------------------
template <int ByteSize>
struct AtomT;

#if !defined(__CUDACC_VER_MAJOR__) || __CUDACC_VER_MAJOR__ < 13
struct alignas(32) aligned32_t {
  longlong4 data;
};
#else
using aligned32_t = longlong4_32a;
#endif

template <> struct AtomT<32> { using type = aligned32_t; };
template <> struct AtomT<16> { using type = float4; };
template <> struct AtomT<8>  { using type = float2; };
template <> struct AtomT<4>  { using type = float; };
template <> struct AtomT<2>  { using type = int16_t; };
template <> struct AtomT<1>  { using type = int8_t; };

// -----------------------------------------------------------------------
// Wide load / store helpers
// -----------------------------------------------------------------------

template <int MaxLoad, typename T>
__device__ __forceinline__ void wide_load(T* dst, const char* src) {
  constexpr int total = N_UNROLL * static_cast<int>(sizeof(T));
  constexpr int chunk = total <= MaxLoad ? total : MaxLoad;
  constexpr int nchunks = total / chunk;
  using atom = typename AtomT<chunk>::type;
#pragma unroll
  for (int c = 0; c < nchunks; ++c)
    reinterpret_cast<atom*>(dst)[c] = reinterpret_cast<const atom*>(src)[c];
}

template <int MaxStore, typename T>
__device__ __forceinline__ void wide_store(char* dst, const T* src) {
  constexpr int total = N_UNROLL * static_cast<int>(sizeof(T));
  constexpr int chunk = total <= MaxStore ? total : MaxStore;
  constexpr int nchunks = total / chunk;
  using atom = typename AtomT<chunk>::type;
#pragma unroll
  for (int c = 0; c < nchunks; ++c)
    reinterpret_cast<atom*>(dst)[c] = reinterpret_cast<const atom*>(src)[c];
}

// RegArray: wraps a C array so it can be stored in a std::tuple.
template <typename T>
struct RegArray {
  T v[N_UNROLL];
  __device__ __forceinline__ T& operator[](int i) { return v[i]; }
  __device__ __forceinline__ const T& operator[](int i) const { return v[i]; }
};

// Load all inputs, invoke the functor N_UNROLL times, store results.
template <int MaxLoad, typename traits, typename func_t, typename array_t,
          size_t... Is>
__device__ __forceinline__ void load_invoke_store_impl(
    const func_t& f, array_t& data, std::index_sequence<Is...>) {
  using result_t = typename traits::result_type;

  std::tuple<RegArray<typename traits::template arg<Is>::type>...> inputs;
  (wide_load<MaxLoad>(std::get<Is>(inputs).v, data[Is + 1]), ...);

  result_t out[N_UNROLL];
#pragma unroll
  for (int i = 0; i < N_UNROLL; ++i)
    out[i] = f(std::get<Is>(inputs)[i]...);

  wide_store<MaxLoad>(data[0], out);
}

// -----------------------------------------------------------------------
// Main device function — one instantiation per TAG/functor/type combo
// -----------------------------------------------------------------------

template <int TAG, typename func_t, typename array_t>
__device__ __noinline__ __attribute__((used)) void simple_gpu_kernel(
    func_t f, array_t data) {
  using traits = function_traits<func_t>;
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 1000
  static constexpr int kMaxLoadSize = 32;
#else
  static constexpr int kMaxLoadSize = 16;
#endif
  load_invoke_store_impl<kMaxLoadSize, traits>(
      f, data, std::make_index_sequence<traits::arity>{});
}

#endif // GPU_KERNEL_UNROLL

// -----------------------------------------------------------------------
// Host-side instantiation trigger
// -----------------------------------------------------------------------

template <int TAG, typename func_t>
void gpu_kernel_impl_nocast_tagged(TensorIteratorBase& iter, const func_t& f) {
  using traits = function_traits<func_t>;

  [[maybe_unused]] auto _inst = [f] __host__ __device__ () {
#ifdef __CUDA_ARCH__
#if GPU_KERNEL_UNROLL <= 1
    _instantiate_simple_kernel<TAG, traits>(
        f, std::make_index_sequence<traits::arity>{});
#else
    constexpr int ntensors = traits::arity + 1;
    std::array<char*, ntensors> data{};
    simple_gpu_kernel<TAG>(f, data);
#endif
#endif
  };
}

template <int TAG, typename func_t>
void gpu_kernel_impl_tagged(TensorIteratorBase& iter, const func_t& f) {
  return gpu_kernel_impl_nocast_tagged<TAG>(iter, f);
}

} // namespace at::native

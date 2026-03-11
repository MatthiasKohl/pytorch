#pragma once

// Simplified Loops.cuh for the standalone gpu_kernel instantiation library.
//
// Public API names (gpu_kernel, gpu_kernel_with_scalars, etc.) are macros
// that capture __LINE__ from the .cu call site and forward it as a TAG
// template parameter.  This TAG ends up in the mangled kernel name so
// cuobjdump output can be mapped back to the exact source line.
//
// IMPORTANT: the macros are defined at the BOTTOM of this file, AFTER
// all internal template functions, so that the internal functions never
// have their identifier accidentally macro-expanded.

#include <ATen/OpMathType.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TensorIteratorDynamicCasting.h>
#include <ATen/native/cuda/thread_constants.h>
#include <ATen/native/cuda/MemoryAccess.cuh>

#include <tuple>

// Our simplified CUDALoops.cuh (simple_gpu_kernel + gpu_kernel_impl_tagged)
#include <ATen/native/cuda/CUDALoops.cuh>

namespace at::native {

// -----------------------------------------------------------------------
// Helper functions from the original Loops.cuh that other files depend on
// -----------------------------------------------------------------------

template<int N>
static OffsetCalculator<N> make_input_offset_calculator(const TensorIteratorBase& iter) {
  constexpr int array_size = std::max<int>(N, 1);
  TORCH_INTERNAL_ASSERT(N == iter.ntensors() - iter.noutputs());
  std::array<const int64_t*, array_size> strides;
  int64_t element_sizes[array_size];
  for (int i = 0; i < N; i++) {
    strides[i] = iter.strides(i + iter.noutputs()).data();
    element_sizes[i] = iter.element_size(i + iter.noutputs());
  }
  return OffsetCalculator<N>(iter.ndim(), iter.shape().data(), strides.data(), element_sizes);
}

template <int num_outputs = 1>
static OffsetCalculator<num_outputs> make_output_offset_calculator(const TensorIteratorBase& iter) {
  TORCH_INTERNAL_ASSERT(num_outputs == iter.noutputs());
  std::array<const int64_t*, num_outputs> strides;
  int64_t element_sizes[num_outputs];
  for (int i = 0; i < num_outputs; i++) {
    strides[i] = iter.strides(i).data();
    element_sizes[i] = iter.element_size(i);
  }
  return OffsetCalculator<num_outputs>(iter.ndim(), iter.shape().data(), strides.data(), element_sizes);
}

template <bool reverted_idx = false, typename func_t, typename policy_t>
__device__ inline void elementwise_kernel_helper(func_t f, policy_t policy) {
  using traits = function_traits<func_t>;
  using return_t = typename traits::result_type;
  using args_t = typename traits::ArgsTuple;
  constexpr int elems_per_thread = policy_t::tws;

  int idx = blockIdx.x;
  if constexpr (reverted_idx)
    idx = gridDim.x - blockIdx.x - 1;

  return_t results[elems_per_thread];
  args_t args[elems_per_thread];

  policy.load(args, idx);

  #pragma unroll
  for (int i = 0; i < elems_per_thread; i++) {
    if (policy.check_inbounds(i)) {
      results[i] = std::apply(f, args[i]);
    }
  }

  policy.store(results, idx);
}

} // namespace at::native (close temporarily for Math.cuh fix-up)

// -----------------------------------------------------------------------
// Fix-up: Math.cuh's calc_i1/calc_i1e only compile for float/double
// (the Chebyshev coefficient templates lack Half/BFloat16 specializations).
// Provide explicit specializations that delegate to the float versions.
// Full specializations are found at the point of instantiation, avoiding
// the two-phase lookup issue with the inner chebyshev calls.
// -----------------------------------------------------------------------
#include <ATen/native/cuda/Math.cuh>

namespace at::native {

template <>
inline C10_HOST_DEVICE c10::Half calc_i1<c10::Half>(c10::Half _x) {
  return static_cast<c10::Half>(calc_i1<float>(static_cast<float>(_x)));
}

template <>
inline C10_HOST_DEVICE c10::BFloat16 calc_i1<c10::BFloat16>(c10::BFloat16 _x) {
  return static_cast<c10::BFloat16>(calc_i1<float>(static_cast<float>(_x)));
}

template <>
inline C10_HOST_DEVICE c10::Half calc_i1e<c10::Half>(c10::Half _x) {
  return static_cast<c10::Half>(calc_i1e<float>(static_cast<float>(_x)));
}

template <>
inline C10_HOST_DEVICE c10::BFloat16 calc_i1e<c10::BFloat16>(c10::BFloat16 _x) {
  return static_cast<c10::BFloat16>(calc_i1e<float>(static_cast<float>(_x)));
}

} // namespace at::native (close after fix-up)

namespace at::native {

// Mirrors the enum from the real Loops.cuh so that functors in .cu files
// can use FunctorType as a non-type template parameter in is_simple.
enum class FunctorType : int {
  AUnary = 0,
  BUnary = 1,
  Binary = 2,
};

template <typename func_t, int cc_major, int cc_minor, typename = void>
struct get_is_simple {
  static constexpr bool value = false;
};

template <typename func_t, int cc_major, int cc_minor>
struct get_is_simple<func_t, cc_major, cc_minor, std::void_t<
      decltype(func_t::template is_simple<cc_major, cc_minor>)>
    > {
  static constexpr bool value = func_t::template is_simple<cc_major, cc_minor>;
};

template <typename func_t, int cc_major, int cc_minor, FunctorType functor_type, typename = void>
struct get_is_simple_with_scalars : get_is_simple<func_t, cc_major, cc_minor> {};

template <typename func_t, int cc_major, int cc_minor, FunctorType functor_type>
struct get_is_simple_with_scalars<func_t, cc_major, cc_minor, functor_type, std::void_t<
      decltype(func_t::template is_simple<cc_major, cc_minor, functor_type>)>
    > {
  static constexpr bool value = func_t::template is_simple<cc_major, cc_minor, functor_type>;
};

// -----------------------------------------------------------------------
// Internal tagged function templates (never use public macro names here!)
// -----------------------------------------------------------------------

template <int TAG, typename func_t>
void _gpu_kernel_nocast_tagged(TensorIteratorBase& iter, const func_t& f) {
  // avoid warnings about unused template variable
  constexpr bool is_simple_func = get_is_simple<func_t, 0, 0>::value;
  (void)is_simple_func;

  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(
        iter.device(arg).is_cuda(),
        "argument ", arg, ": expected a CUDA device but found ",
        iter.device(arg));
  }
  if (iter.numel() == 0) return;
  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      _gpu_kernel_nocast_tagged<TAG>(sub_iter, f);
    }
    return;
  }
  gpu_kernel_impl_nocast_tagged<TAG>(iter, f);
}

template <int TAG, typename func_t>
void _gpu_kernel_tagged(TensorIteratorBase& iter, const func_t& f) {
  // avoid warnings about unused template variable
  constexpr bool is_simple_func = get_is_simple<func_t, 0, 0>::value;
  (void)is_simple_func;

  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(
        iter.device(arg).is_cuda(),
        "argument ", arg, ": expected a CUDA device but found ",
        iter.device(arg));
  }
  if (iter.numel() == 0) return;
  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      _gpu_kernel_tagged<TAG>(sub_iter, f);
    }
    return;
  }
  gpu_kernel_impl_tagged<TAG>(iter, f);
}

// -----------------------------------------------------------------------
// Wrapper functors (used by gpu_kernel_with_scalars and friends)
// -----------------------------------------------------------------------

template <typename arg1_t, typename arg2_t, typename return_t, typename func_t>
struct AUnaryFunctor {
  template <int cc_major, int cc_minor>
  static constexpr bool is_simple = get_is_simple_with_scalars<
    func_t, cc_major, cc_minor, FunctorType::AUnary>::value;

  using traits = function_traits<func_t>;
  using opmath_arg1_t = typename traits::template arg<0>::type;
  __device__ return_t operator()(arg2_t b) const { return f(a, b); }
  AUnaryFunctor(func_t f_, opmath_arg1_t a_) : f(f_), a(a_) {}
 private:
  func_t f;
  opmath_arg1_t a;
};

template <typename arg1_t, typename arg2_t, typename return_t, typename func_t>
struct BUnaryFunctor {
  template <int cc_major, int cc_minor>
  static constexpr bool is_simple = get_is_simple_with_scalars<
    func_t, cc_major, cc_minor, FunctorType::BUnary>::value;

  using traits = function_traits<func_t>;
  using opmath_arg2_t = typename traits::template arg<1>::type;
  __device__ return_t operator()(arg1_t a) const { return f(a, b); }
  BUnaryFunctor(func_t f_, opmath_arg2_t b_) : f(f_), b(b_) {}
 private:
  func_t f;
  opmath_arg2_t b;
};

template <typename arg1_t, typename arg2_t, typename return_t, typename func_t>
struct BinaryFunctor {
  template <int cc_major, int cc_minor>
  static constexpr bool is_simple = get_is_simple_with_scalars<
    func_t, cc_major, cc_minor, FunctorType::Binary>::value;

  __device__ return_t operator()(arg1_t a, arg2_t b) const { return f(a, b); }
  BinaryFunctor(func_t f_) : f(f_) {}
 private:
  func_t f;
};

// -----------------------------------------------------------------------
// Tagged gpu_kernel_with_scalars
// -----------------------------------------------------------------------

template <int TAG, typename func_t>
void _gpu_kernel_with_scalars_tagged(
    TensorIteratorBase& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  static_assert(
      traits::arity == 2,
      "gpu_kernel_with_scalars only supports two input arguments");
  using arg1_t = typename traits::template arg<0>::type;
  using arg2_t = typename traits::template arg<1>::type;
  using return_t = typename traits::result_type;

  TORCH_INTERNAL_ASSERT(iter.ntensors() == 3);

  using opmath_arg1_t = typename traits::template arg<0>::type;
  using opmath_arg2_t = typename traits::template arg<1>::type;

  if (iter.is_cpu_scalar(1)) {
    AUnaryFunctor<arg1_t, arg2_t, return_t, func_t> af(
        f, iter.scalar_value<opmath_arg1_t>(1));
    iter.remove_operand(1);
    const OptionalDeviceGuard device_guard(iter.device(1));
    _gpu_kernel_tagged<TAG>(iter, af);
  } else if (iter.is_cpu_scalar(2)) {
    BUnaryFunctor<arg1_t, arg2_t, return_t, func_t> bf(
        f, iter.scalar_value<opmath_arg2_t>(2));
    iter.remove_operand(2);
    _gpu_kernel_tagged<TAG>(iter, bf);
  } else {
    _gpu_kernel_tagged<TAG>(
        iter, BinaryFunctor<arg1_t, arg2_t, return_t, func_t>(f));
  }
}

// -----------------------------------------------------------------------
// Tagged opmath_gpu_kernel_with_scalars  (called as a template with
// explicit type args, so we use a helper struct to combine TAG + types)
// -----------------------------------------------------------------------

template <int TAG>
struct _opmath_helper {
  template <
      typename arg1_t,
      typename arg2_t = arg1_t,
      typename return_t = arg1_t,
      typename func_t>
  static void call(TensorIteratorBase& iter, const func_t& f) {
    TORCH_INTERNAL_ASSERT(iter.ntensors() == 3);

    using traits = function_traits<func_t>;
    using opmath_arg1_t = typename traits::template arg<0>::type;
    using opmath_arg2_t = typename traits::template arg<1>::type;
    static_assert(
        traits::arity == 2,
        "gpu_kernel_with_scalars only supports two input arguments");

    if (iter.is_cpu_scalar(1)) {
      AUnaryFunctor<arg1_t, arg2_t, return_t, func_t> af(
          f, iter.scalar_value<opmath_arg1_t>(1));
      iter.remove_operand(1);
      const OptionalDeviceGuard device_guard(iter.device(1));
      _gpu_kernel_tagged<TAG>(iter, af);
    } else if (iter.is_cpu_scalar(2)) {
      BUnaryFunctor<arg1_t, arg2_t, return_t, func_t> bf(
          f, iter.scalar_value<opmath_arg2_t>(2));
      iter.remove_operand(2);
      _gpu_kernel_tagged<TAG>(iter, bf);
    } else {
      _gpu_kernel_tagged<TAG>(
          iter, BinaryFunctor<arg1_t, arg2_t, return_t, func_t>(f));
    }
  }
};

// -----------------------------------------------------------------------
// Tagged opmath_symmetric_gpu_kernel_with_scalars
// -----------------------------------------------------------------------

template <int TAG>
struct _opmath_symmetric_helper {
  template <
      typename scalar_t,
      typename return_t = scalar_t,
      typename func_t>
  static void call(TensorIteratorBase& iter, const func_t& f) {
    TORCH_INTERNAL_ASSERT(iter.ntensors() == 3);

    using traits = function_traits<func_t>;
    using opmath_arg_t = typename traits::template arg<0>::type;
    static_assert(
        traits::arity == 2,
        "gpu_kernel_with_scalars only supports two input arguments");
    static_assert(
        std::is_same_v<
            opmath_arg_t, typename traits::template arg<1>::type>,
        "f is not symmetric");

    OptionalDeviceGuard device_guard;
    opmath_arg_t scalar_val{};

    if (iter.is_cpu_scalar(1)) {
      scalar_val = iter.scalar_value<opmath_arg_t>(1);
      iter.remove_operand(1);
      device_guard.reset_device(iter.device(1));
    } else if (iter.is_cpu_scalar(2)) {
      scalar_val = iter.scalar_value<opmath_arg_t>(2);
      iter.remove_operand(2);
    }

    if (iter.ninputs() == 2) {
      _gpu_kernel_tagged<TAG>(
          iter,
          BinaryFunctor<scalar_t, scalar_t, return_t, func_t>(f));
    } else {
      AUnaryFunctor<scalar_t, scalar_t, return_t, func_t> unary_f(
          f, scalar_val);
      _gpu_kernel_tagged<TAG>(iter, unary_f);
    }
  }
};

} // namespace at::native

// -----------------------------------------------------------------------
// Public API macros — capture __LINE__ from the .cu call site
// -----------------------------------------------------------------------
// These MUST be after all internal templates so they don't accidentally
// macro-expand identifiers inside the template definitions above.

#define gpu_kernel(iter, ...) \
    _gpu_kernel_tagged<__LINE__>(iter, __VA_ARGS__)

#define gpu_kernel_nocast(iter, ...) \
    _gpu_kernel_nocast_tagged<__LINE__>(iter, __VA_ARGS__)

#define gpu_kernel_with_scalars(iter, ...) \
    _gpu_kernel_with_scalars_tagged<__LINE__>(iter, __VA_ARGS__)

// For template-argument entry points we use the helper-struct trick:
//   opmath_gpu_kernel_with_scalars<scalar_t>(iter, f)
// becomes:
//   _opmath_helper<__LINE__>::call<scalar_t>(iter, f)
#define opmath_gpu_kernel_with_scalars \
    _opmath_helper<__LINE__>::call

#define opmath_symmetric_gpu_kernel_with_scalars \
    _opmath_symmetric_helper<__LINE__>::call

#define gpu_kernel_multiple_outputs(iter, ...) \
    do { (void)(iter); } while (0)

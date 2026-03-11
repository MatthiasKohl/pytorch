#define TORCH_ASSERT_NO_OPERATORS
#define _USE_MATH_DEFINES

#include <ATen/native/Activation.h>

#include <cmath>

#include <thrust/tuple.h>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/core/TensorBase.h>
#include <c10/core/Scalar.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/cuda/ApplyGridUtils.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/Loops.cuh>

namespace at::native {
namespace {

template <typename scalar_t, typename opmath_t>
struct HardsigmoidFunctor {
  // only double not simple + bf16 for SM 75-
  template <int cc_major, int /*cc_minor*/>
  static constexpr bool is_simple =
    !(std::is_same_v<scalar_t, double> ||
     (cc_major < 8 && std::is_same_v<scalar_t, c10::BFloat16>));

  GPU_LAMBDA scalar_t operator() (const scalar_t self_val) const {
    constexpr opmath_t zero = opmath_t(0.0f);
    constexpr opmath_t one_sixth = opmath_t(1.0f / 6.0f);
    constexpr opmath_t three = opmath_t(3.0f);
    constexpr opmath_t six = opmath_t(6.0f);
    opmath_t x = static_cast<opmath_t>(self_val);
    return std::min<opmath_t>(std::max<opmath_t>(x + three, zero), six) * one_sixth;
  }
};

void hardsigmoid_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "hardsigmoid_cuda",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto functor = HardsigmoidFunctor<scalar_t, opmath_t>();
        gpu_kernel(iter, functor);
      });
}

template <typename scalar_t, typename opmath_t>
struct HardsigmoidBackwardFunctor {
  // only double not simple + bf16 for SM 75-
  template <int cc_major, int /*cc_minor*/>
  static constexpr bool is_simple =
    !(std::is_same_v<scalar_t, double> ||
     (cc_major < 8 && std::is_same_v<scalar_t, c10::BFloat16>));

  GPU_LAMBDA scalar_t operator() (const scalar_t grad_val_, const scalar_t self_val_) const {
    constexpr opmath_t zero = opmath_t(0.0f);
    constexpr opmath_t three = opmath_t(3.0f);
    constexpr opmath_t neg_three = opmath_t(-3.0f);
    constexpr opmath_t one_sixth = opmath_t(1.0f / 6.0f);
    opmath_t grad_val = static_cast<opmath_t>(grad_val_);
    opmath_t self_val = static_cast<opmath_t>(self_val_);
    return (self_val > neg_three && self_val < three)
        ? grad_val * one_sixth
        : zero;
  }
};

void hardsigmoid_backward_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "hardsigmoid_backward_cuda",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto functor = HardsigmoidBackwardFunctor<scalar_t, opmath_t>();
        gpu_kernel(iter, functor);
      });
}

} // namespace

REGISTER_DISPATCH(hardsigmoid_stub, &hardsigmoid_kernel)
REGISTER_DISPATCH(hardsigmoid_backward_stub, &hardsigmoid_backward_kernel)

} // namespace at::native

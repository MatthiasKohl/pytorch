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
struct LeakyReluFunctor {
  // only double not simple + bf16 for SM 75-
  template <int cc_major, int /*cc_minor*/>
  static constexpr bool is_simple =
    !(std::is_same_v<scalar_t, double> ||
     (cc_major < 8 && std::is_same_v<scalar_t, c10::BFloat16>));

  GPU_LAMBDA scalar_t operator()(scalar_t a) const {
    opmath_t aop = static_cast<opmath_t>(a);
    return aop > opmath_t(0) ? aop : aop * negval;
  }
  LeakyReluFunctor(opmath_t negval_) : negval(negval_) {}
  private:
  opmath_t negval;
};

template <typename scalar_t, typename opmath_t>
struct LeakyReluBackwardFunctor {
  // only double not simple + bf16 for SM 75-
  template <int cc_major, int /*cc_minor*/>
  static constexpr bool is_simple =
    !(std::is_same_v<scalar_t, double> ||
     (cc_major < 8 && std::is_same_v<scalar_t, c10::BFloat16>));

  GPU_LAMBDA scalar_t operator()(scalar_t a, scalar_t b) const {
    opmath_t aop = static_cast<opmath_t>(a);
    opmath_t bop = static_cast<opmath_t>(b);
    return aop > opmath_t(0) ? bop : bop * negval;
  }
  LeakyReluBackwardFunctor(opmath_t negval_) : negval(negval_) {}
  private:
  opmath_t negval;
};

void leaky_relu_kernel(TensorIteratorBase& iter, const Scalar& negval_) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "leaky_relu_cuda",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto functor = LeakyReluFunctor<scalar_t, opmath_t>(negval_.to<opmath_t>());
        gpu_kernel(iter, functor);
      });
}

void leaky_relu_backward_kernel(
    TensorIteratorBase& iter,
    const Scalar& negval_) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "leaky_relu_backward_cuda",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto functor = LeakyReluBackwardFunctor<scalar_t, opmath_t>(negval_.to<opmath_t>());
        gpu_kernel(iter, functor);
      });
}
} // namespace

REGISTER_DISPATCH(leaky_relu_stub, &leaky_relu_kernel)
REGISTER_DISPATCH(leaky_relu_backward_stub, &leaky_relu_backward_kernel)

} // namespace at::native

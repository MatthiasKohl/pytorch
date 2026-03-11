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
#include <ATen/NumericUtils.h>
#include <ATen/cuda/ApplyGridUtils.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/Loops.cuh>

namespace at::native {
namespace {

template <typename scalar_t>
struct SoftshrinkFunctor {
  // note: code-gen for softshrink seems non-optimal, generating a lot of branching.
  // thus, this is never marked as simple.
  template <int /*cc_major*/, int /*cc_minor*/>
  static constexpr bool is_simple = false;

  GPU_LAMBDA scalar_t operator()(scalar_t a) const {
    return at::_isnan(a) ? a : (a > lambd ? a - lambd : (a < -lambd ? a + lambd : scalar_t(0)));
  }
  SoftshrinkFunctor(scalar_t lambd_) : lambd(lambd_) {}
  private:
  scalar_t lambd;
};

template <typename scalar_t>
struct ShrinkBackwardFunctor {
  // only double not simple + bf16 for SM 75-
  template <int cc_major, int /*cc_minor*/>
  static constexpr bool is_simple =
    !(std::is_same_v<scalar_t, double> ||
     (cc_major < 8 && std::is_same_v<scalar_t, c10::BFloat16>));

  GPU_LAMBDA scalar_t operator()(scalar_t grad_val, scalar_t self_val) const {
    return (self_val >= -lambd && self_val <= lambd) ? scalar_t(0) : grad_val;
  }
  ShrinkBackwardFunctor(scalar_t lambd_) : lambd(lambd_) {}
  private:
  scalar_t lambd;
};

void softshrink_kernel(TensorIteratorBase& iter, const Scalar& value) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "softshrink_cuda",
      [&]() {
        auto functor = SoftshrinkFunctor<scalar_t>(value.to<scalar_t>());
        gpu_kernel(iter, functor);
      });
}

void shrink_backward_kernel(TensorIteratorBase& iter, const Scalar& value) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "shrink_backward_cuda",
      [&]() {
        auto functor = ShrinkBackwardFunctor<scalar_t>(value.to<scalar_t>());
        gpu_kernel(iter, functor);
      });
}
} // namespace

REGISTER_DISPATCH(softshrink_stub, &softshrink_kernel)
REGISTER_DISPATCH(shrink_backward_stub, &shrink_backward_kernel)

} // namespace at::native

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

template <typename scalar_t>
struct ThresholdFunctor {
  // always simple
  template <int /*cc_major*/, int /*cc_minor*/, FunctorType /*functor_type*/>
  static constexpr bool is_simple = true;

  GPU_LAMBDA scalar_t operator()(scalar_t x, scalar_t other) const {
    return x <= threshold ? value : other;
  }
  ThresholdFunctor(scalar_t threshold_, scalar_t value_) : threshold(threshold_), value(value_) {}
  private:
  scalar_t threshold;
  scalar_t value;
};

template <typename scalar_t>
void threshold_kernel_impl(
    TensorIteratorBase& iter,
    scalar_t threshold,
    scalar_t value) {
  gpu_kernel_with_scalars(iter, ThresholdFunctor<scalar_t>(threshold, value));
}

static void threshold_kernel_cuda(
    TensorIteratorBase& iter,
    const Scalar& threshold,
    const Scalar& value) {
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "threshold_cuda",
      [&] {
        threshold_kernel_impl<scalar_t>(
            iter, threshold.to<scalar_t>(), value.to<scalar_t>());
      });
}

} // namespace

REGISTER_DISPATCH(threshold_stub, &threshold_kernel_cuda)

} // namespace at::native

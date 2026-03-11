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
struct HardshrinkFunctor {
  // only double not simple + bf16 for SM 75-
  template <int cc_major, int /*cc_minor*/>
  static constexpr bool is_simple =
    !(std::is_same_v<scalar_t, double> ||
      (cc_major < 8 && std::is_same_v<scalar_t, c10::BFloat16>));

  GPU_LAMBDA scalar_t operator() (const scalar_t a) const {
    return (a >= -lambd && a <= lambd) ? scalar_t(0) : a;
  }
  HardshrinkFunctor(scalar_t lambd_) : lambd(lambd_) {}
  private:
  scalar_t lambd;
};

void hardshrink_kernel(TensorIteratorBase& iter, const Scalar& value) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "hardshrink_cuda",
      [&]() {
        auto functor = HardshrinkFunctor<scalar_t>(value.to<scalar_t>());
        gpu_kernel(iter, functor);
      });
}
} // namespace

REGISTER_DISPATCH(hardshrink_stub, &hardshrink_kernel)

} // namespace at::native

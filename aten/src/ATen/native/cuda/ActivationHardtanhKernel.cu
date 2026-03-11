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
struct HardtanhBackwardFunctor {
  // only double not simple + bf16 for SM 75-
  template <int cc_major, int /*cc_minor*/>
  static constexpr bool is_simple =
    !(std::is_same_v<scalar_t, double> ||
     (cc_major < 8 && std::is_same_v<scalar_t, c10::BFloat16>));

  GPU_LAMBDA scalar_t operator()(scalar_t a, scalar_t b) const {
    opmath_t aop = static_cast<opmath_t>(a);
    opmath_t bop = static_cast<opmath_t>(b);
    return (bop <= min_val) || (bop >= max_val) ? opmath_t(0) : aop;
  }
  HardtanhBackwardFunctor(opmath_t min_val_, opmath_t max_val_) : min_val(min_val_), max_val(max_val_) {}
  private:
  opmath_t min_val;
  opmath_t max_val;
};

void hardtanh_backward_kernel(
    TensorIterator& iter,
    const Scalar& min,
    const Scalar& max) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      iter.dtype(), "hardtanh_backward_cuda", [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto functor = HardtanhBackwardFunctor<scalar_t, opmath_t>(min.to<opmath_t>(), max.to<opmath_t>());
        gpu_kernel(iter, functor);
      });
}
} // namespace

REGISTER_DISPATCH(hardtanh_backward_stub, &hardtanh_backward_kernel)

} // namespace at::native

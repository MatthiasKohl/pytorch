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

template <typename scalar_t>
struct PreluFunctor {
  // only half and float are simple
  template <int /*cc_major*/, int /*cc_minor*/>
  static constexpr bool is_simple = (
    std::is_same_v<scalar_t, c10::Half> ||
    std::is_same_v<scalar_t, float>);

  GPU_LAMBDA scalar_t operator()(scalar_t input, scalar_t weight) const {
    return (input > 0) ? input : weight * input;
  }
};

// -----------------------------------
// prelu
// -----------------------------------
void prelu_kernel(TensorIterator &iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "prelu_cuda", [&] {
    gpu_kernel(iter, PreluFunctor<scalar_t>());
  });
}

void prelu_backward_kernel(TensorIterator &iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "prelu_backward_cuda", [&] {
    gpu_kernel_multiple_outputs(iter,
      [] GPU_LAMBDA (scalar_t input, scalar_t weight, scalar_t grad) -> thrust::tuple<scalar_t, scalar_t> {
        auto mask = input > 0;
        auto grad_input = mask ? grad : weight * grad;
        auto grad_weight = mask ? scalar_t{0} : input * grad;
        return {grad_input, grad_weight};
      });
  });
}

REGISTER_DISPATCH(prelu_stub, &prelu_kernel)
REGISTER_DISPATCH(prelu_backward_stub, &prelu_backward_kernel)

} // namespace at::native

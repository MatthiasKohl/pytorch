#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <c10/util/BFloat16-math.h>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native {

template <typename scalar_t>
struct NextafterFunctor {
  // never simple
  template <int /*cc_major*/, int /*cc_minor*/, FunctorType /*functor_type*/>
  static constexpr bool is_simple = false;

  GPU_LAMBDA scalar_t operator()(scalar_t a, scalar_t b) const {
    return std::nextafter(a, b);
  }
};

template <typename scalar_t>
struct HeavisideFunctor {
  // only non-simple cases are bf16 in SM 75-, double with SM 10.3 / 11.x / 12.x,
  template <int cc_major, int cc_minor, FunctorType /*functor_type*/>
  static constexpr bool is_simple = !(
    (std::is_same_v<scalar_t, c10::BFloat16> && cc_major < 8) ||
    (std::is_same_v<scalar_t, double> && (
      (cc_major >= 10 && cc_minor == 3) || cc_major == 11 || cc_major == 12)));

  GPU_LAMBDA scalar_t operator()(scalar_t a, scalar_t b) const {
    return a == 0 ? b : static_cast<scalar_t>(a > 0);
  }
};

void nextafter_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "nextafter_cuda", [&]() {
    gpu_kernel_with_scalars(iter, NextafterFunctor<scalar_t>());
  });
}

void heaviside_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBool, kBFloat16, iter.dtype(), "heaviside_cuda", [&]() {
    gpu_kernel_with_scalars(iter, HeavisideFunctor<scalar_t>());
  });
}

REGISTER_DISPATCH(nextafter_stub, &nextafter_kernel_cuda)
REGISTER_DISPATCH(heaviside_stub, &heaviside_kernel_cuda)

} // namespace at::native

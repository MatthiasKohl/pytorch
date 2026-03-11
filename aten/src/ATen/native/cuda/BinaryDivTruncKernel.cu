#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/TypeSafeSignMath.h>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/cuda/Loops.cuh>

#include <type_traits>

namespace at::native {
namespace binary_internal {

template <typename scalar_t, typename accscalar_t>
struct DivTruncMulInvFunctor {
  // only bf16 for SM 75- is not simple
  template <int cc_major, int /*cc_minor*/>
  static constexpr bool is_simple =
    !(cc_major < 8 && std::is_same_v<scalar_t, c10::BFloat16>);

  GPU_LAMBDA scalar_t operator()(scalar_t a) const {
    return std::trunc(a * inv_b);
  }
  DivTruncMulInvFunctor(accscalar_t inv_b_) : inv_b(inv_b_) {}
  private:
  accscalar_t inv_b;
};

void div_trunc_kernel_cuda(TensorIteratorBase& iter) {
  auto dtype = iter.common_dtype();
  if (isIntegralType(dtype, /*includeBool*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(dtype, "div_trunc_cuda", [&]() {
      gpu_kernel_with_scalars(
          iter,
          [] GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t { return a / b; });
    });
  } else if (iter.is_cpu_scalar(2)) {
    // optimization for floating-point types: if the second operand is a CPU
    // scalar, compute a * reciprocal(b). Note that this may lose one bit of
    // precision compared to computing the division.
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, dtype, "div_trunc_cuda", [&]() {
          using accscalar_t = at::acc_type<scalar_t, true>;
          auto inv_b = accscalar_t(1.0) / iter.scalar_value<accscalar_t>(2);
          iter.remove_operand(2);
          gpu_kernel(iter, DivTruncMulInvFunctor<scalar_t, accscalar_t>(inv_b));
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, dtype, "div_trunc_cuda", [&]() {
          gpu_kernel_with_scalars(
              iter, [] GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
                return std::trunc(a / b);
              });
        });
  }
}
} // namespace binary_internal

REGISTER_DISPATCH(div_trunc_stub, &binary_internal::div_trunc_kernel_cuda)

} // namespace at::native

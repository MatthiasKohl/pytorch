#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>


// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native { namespace {

enum class EqOpType {EQ, NE};

template<typename scalar_t>
struct CompareEqFunctor{
  // only simple cases are:
  // c10::Float4_e2m1fn_x2 with SM 100+,
  // unsigned long with SM 100+,
  // unsigned int,
  // unsigned short,
  // bool with SM 100+,
  // bf16,
  // half,
  // complex float,
  // float,
  // double with non-binary functor,
  // short,
  // long with SM 100+,
  // int,
  // signed char with SM 100+,
  // unsigned char with SM 100+,
  template <int cc_major, int /*cc_minor*/, FunctorType functor_type>
  static constexpr bool is_simple =
    (std::is_same_v<scalar_t, c10::Float4_e2m1fn_x2> && cc_major >= 10) ||
    (std::is_same_v<scalar_t, unsigned long> && cc_major >= 10) ||
    (std::is_same_v<scalar_t, unsigned int>) ||
    (std::is_same_v<scalar_t, unsigned short>) ||
    (std::is_same_v<scalar_t, bool> && cc_major >= 10) ||
    (std::is_same_v<scalar_t, c10::BFloat16>) ||
    (std::is_same_v<scalar_t, c10::Half>) ||
    (std::is_same_v<scalar_t, c10::complex<float>>) ||
    (std::is_same_v<scalar_t, float>) ||
    (std::is_same_v<scalar_t, double> && functor_type != FunctorType::Binary) ||
    (std::is_same_v<scalar_t, short>) ||
    (std::is_same_v<scalar_t, long> && cc_major >= 10) ||
    (std::is_same_v<scalar_t, int>) ||
    (std::is_same_v<scalar_t, signed char> && cc_major >= 10) ||
    (std::is_same_v<scalar_t, unsigned char> && cc_major >= 10);

  CompareEqFunctor(EqOpType op): op_(op) {}
  const EqOpType op_;
  __device__ __forceinline__ bool operator() (scalar_t a, scalar_t b) const {
    if (op_ == EqOpType::EQ) {
      return a == b;
    } else { //NE
      return a != b;
    }

  }
 };
}

C10_NOINLINE void compare_eq_ne_kernel(TensorIteratorBase &iter, EqOpType op) {
  AT_DISPATCH_V2(iter.common_dtype(), "compare_eq_ne_cuda", AT_WRAP([&]() {
    opmath_symmetric_gpu_kernel_with_scalars<scalar_t, bool>(
        iter, CompareEqFunctor<scalar_t>(op));
  }), AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), kComplexHalf, kHalf, kBFloat16, kBool, AT_EXPAND(AT_FLOAT8_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES), kFloat4_e2m1fn_x2);
}

void eq_kernel_cuda(TensorIteratorBase& iter) {
  compare_eq_ne_kernel(iter, EqOpType::EQ);
}

void ne_kernel_cuda(TensorIteratorBase& iter) {
  compare_eq_ne_kernel(iter, EqOpType::NE);
}

REGISTER_DISPATCH(eq_stub, &eq_kernel_cuda)
REGISTER_DISPATCH(ne_stub, &ne_kernel_cuda)

} // namespace at::native

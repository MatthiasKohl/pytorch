#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native {

template <typename scalar_t>
struct LogicalAndFunctor {
  // only non-simple are: complex<double>, complex<float> with binary functor,
  // and double with binary functor.
  template <int /*cc_major*/, int /*cc_minor*/, FunctorType functor_type>
  static constexpr bool is_simple =
    !(std::is_same_v<scalar_t, c10::complex<double>> ||
      (std::is_same_v<scalar_t, c10::complex<float>> && functor_type == FunctorType::Binary) ||
      (std::is_same_v<scalar_t, double> && functor_type == FunctorType::Binary));

  GPU_LAMBDA bool operator()(scalar_t a, scalar_t b) const {
    return a && b;
  }
};

template <typename scalar_t>
struct LogicalOrFunctor {
  // only non-simple are: complex<double>, complex<float> with binary functor,
  // and double with binary functor.
  template <int /*cc_major*/, int /*cc_minor*/, FunctorType functor_type>
  static constexpr bool is_simple =
  !(std::is_same_v<scalar_t, c10::complex<double>> ||
    (std::is_same_v<scalar_t, c10::complex<float>> && functor_type == FunctorType::Binary) ||
    (std::is_same_v<scalar_t, double> && functor_type == FunctorType::Binary));

  GPU_LAMBDA bool operator()(scalar_t a, scalar_t b) const {
    return a || b;
  }
};

template <typename scalar_t>
struct LogicalXorFunctor {
  // only non-simple are: complex<double>, complex<float> with binary functor,
  // and double with binary functor.
  template <int /*cc_major*/, int /*cc_minor*/, FunctorType functor_type>
  static constexpr bool is_simple =
    !(std::is_same_v<scalar_t, c10::complex<double>> ||
      (std::is_same_v<scalar_t, c10::complex<float>> && functor_type == FunctorType::Binary) ||
      (std::is_same_v<scalar_t, double> && functor_type == FunctorType::Binary));

  GPU_LAMBDA bool operator()(scalar_t a, scalar_t b) const {
    return bool(a) != bool(b);
  }
};

constexpr char logical_and_name[] = "logical_and_kernel";
void logical_and_kernel_cuda(TensorIterator& iter) {
  auto dtype = iter.common_dtype();
  if (at::isComplexType(dtype)) {
#if AT_USE_JITERATOR()
    static const auto logical_and_string = jiterator_stringify(
        template <typename T>
        bool logical_and_kernel(T a, T b) {
          return a && b;
        }
    ); // logical_and_string
    AT_DISPATCH_COMPLEX_TYPES(dtype, "logical_and_cuda", [&]() {
      jitted_gpu_kernel<
        /*name=*/ logical_and_name,
        /*return_dtype=*/ scalar_t,
        /*common_dtype=*/ scalar_t,
        /*arity=*/ 2>(iter, logical_and_string);
    }); // logical_and_string
#else
    AT_DISPATCH_COMPLEX_TYPES(dtype, "logical_and_cuda", [&]() {
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t, bool>(
          iter, LogicalAndFunctor<scalar_t>());
    });
#endif
  } else {
    AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBool, ScalarType::BFloat16,
                               dtype, "logical_and_cuda", [&]() {
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t, bool>(
          iter, LogicalAndFunctor<scalar_t>());
   });
  }
}

constexpr char logical_or_name[] = "logical_or_kernel";
void logical_or_kernel_cuda(TensorIterator& iter) {
  auto dtype = iter.common_dtype();
  if (at::isComplexType(dtype)) {
#if AT_USE_JITERATOR()
    static const auto logical_or_string = jiterator_stringify(
      template <typename T>
      bool logical_or_kernel(T a, T b) {
        return a || b;
      }
    ); // logical_or_string
    AT_DISPATCH_COMPLEX_TYPES(dtype, "logical_or_cuda", [&]() {
      jitted_gpu_kernel<
        /*name=*/ logical_or_name,
        /*return_dtype=*/ scalar_t,
        /*common_dtype=*/ scalar_t,
        /*arity=*/ 2>(iter, logical_or_string);
    });
#else
    AT_DISPATCH_COMPLEX_TYPES(dtype, "logical_or_cuda", [&]() {
      gpu_kernel_with_scalars(iter, LogicalOrFunctor<scalar_t>());
    });
#endif
  } else {
  AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBool, ScalarType::BFloat16,
                             dtype, "logical_or_cuda", [&]() {
    opmath_symmetric_gpu_kernel_with_scalars<scalar_t, bool>(
        iter, LogicalOrFunctor<scalar_t>());
  });
  }
}

constexpr char logical_xor_name[] = "logical_xor_kernel";
void logical_xor_kernel_cuda(TensorIterator& iter) {
  auto dtype = iter.common_dtype();
  if (at::isComplexType(dtype)) {
#if AT_USE_JITERATOR()
    static const auto logical_xor_string = jiterator_stringify(
        template <typename T>
        bool logical_xor_kernel(T a, T b) {
          return bool(a) != bool(b);
        }
    );
    AT_DISPATCH_COMPLEX_TYPES(dtype, "logical_xor_cuda", [&]() {
      jitted_gpu_kernel<
        /*name=*/ logical_xor_name,
        /*return_dtype=*/ scalar_t,
        /*common_dtype=*/ scalar_t,
        /*arity=*/ 2>(iter, logical_xor_string);
    }); // logical_xor_string
#else
    AT_DISPATCH_COMPLEX_TYPES(dtype, "logical_xor_cuda", [&]() {
      gpu_kernel_with_scalars(iter, LogicalXorFunctor<scalar_t>());
    });
#endif
  } else {
  AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBool, ScalarType::BFloat16,
                             dtype, "logical_xor_cuda", [&]() {
    opmath_symmetric_gpu_kernel_with_scalars<scalar_t, bool>(
        iter, LogicalXorFunctor<scalar_t>());
  });
  }
}

REGISTER_DISPATCH(logical_and_stub, &logical_and_kernel_cuda)
REGISTER_DISPATCH(logical_or_stub, &logical_or_kernel_cuda)
REGISTER_DISPATCH(logical_xor_stub, &logical_xor_kernel_cuda)


} // namespace at::native

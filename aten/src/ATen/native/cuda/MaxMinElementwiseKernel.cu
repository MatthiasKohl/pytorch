#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native {

struct MaximumBoolFunctor {
  // always simple
  template <int /*cc_major*/, int /*cc_minor*/>
  static constexpr bool is_simple = true;

  GPU_LAMBDA bool operator()(bool a, bool b) const {
    return a || b;
  }
};

template <typename scalar_t>
struct MaximumIntFunctor {
  // always simple
  template <int /*cc_major*/, int /*cc_minor*/>
  static constexpr bool is_simple = true;

  GPU_LAMBDA scalar_t operator()(scalar_t a, scalar_t b) const {
    return ::max(a, b);
  }
};

template <typename scalar_t>
struct MaximumFloatFunctor {
  // never simple: the below has too much branching
  template <int /*cc_major*/, int /*cc_minor*/>
  static constexpr bool is_simple = false;

  GPU_LAMBDA scalar_t operator()(scalar_t a, scalar_t b) const {
    if (a != a) {
      return a;
    } else if (b != b) {
      return b;
    } else {
      return ::max(a, b);
    }
  }
};

struct MinimumBoolFunctor {
  // always simple
  template <int /*cc_major*/, int /*cc_minor*/>
  static constexpr bool is_simple = true;

  GPU_LAMBDA bool operator()(bool a, bool b) const {
    return a && b;
  }
};

template <typename scalar_t>
struct MinimumIntFunctor {
  // always simple
  template <int /*cc_major*/, int /*cc_minor*/>
  static constexpr bool is_simple = true;

  GPU_LAMBDA scalar_t operator()(scalar_t a, scalar_t b) const {
    return ::min(a, b);
  }
};

template <typename scalar_t>
struct MinimumFloatFunctor {
  // never simple: the below has too much branching
  template <int /*cc_major*/, int /*cc_minor*/>
  static constexpr bool is_simple = false;

  GPU_LAMBDA scalar_t operator()(scalar_t a, scalar_t b) const {
    if (a != a) {
      return a;
    } else if (b != b) {
      return b;
    } else {
      return ::min(a, b);
    }
  }
};

template <typename scalar_t>
struct FmaxFunctor {
  // only non-simple cases is double and bf16 with SM 75-
  template <int cc_major, int /*cc_minor*/>
  static constexpr bool is_simple = !(
    std::is_same_v<scalar_t, double> ||
    (std::is_same_v<scalar_t, c10::BFloat16> && cc_major < 8));

  GPU_LAMBDA scalar_t operator()(scalar_t a, scalar_t b) const {
    return ::fmax(a, b);
  }
};

template <typename scalar_t>
struct FminFunctor {
  // only non-simple cases is double and bf16 with SM 75-
  template <int cc_major, int /*cc_minor*/>
  static constexpr bool is_simple = !(
    std::is_same_v<scalar_t, double> ||
    (std::is_same_v<scalar_t, c10::BFloat16> && cc_major < 8));

  GPU_LAMBDA scalar_t operator()(scalar_t a, scalar_t b) const {
    return ::fmin(a, b);
  }
};

void maximum_kernel_cuda(TensorIteratorBase& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    opmath_symmetric_gpu_kernel_with_scalars<bool>(iter, MaximumBoolFunctor());
  } else if (isIntegralType(iter.dtype(), /*includeBool=*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "max_elementwise_cuda", [&]() {
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, MaximumIntFunctor<scalar_t>());
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "max_elementwise_cuda", [&]() {
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, MaximumFloatFunctor<scalar_t>());
    });
  }
}

void minimum_kernel_cuda(TensorIteratorBase& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    opmath_symmetric_gpu_kernel_with_scalars<bool>(iter, MinimumBoolFunctor());
  } else if (isIntegralType(iter.dtype(), /*includeBool=*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "minimum_cuda", [&]() {
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, MinimumIntFunctor<scalar_t>());
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "min_elementwise_cuda", [&]() {
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, MinimumFloatFunctor<scalar_t>());
    });
  }
}

void fmax_kernel_cuda(TensorIteratorBase& iter) {
  if (isFloatingType(iter.common_dtype())) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.common_dtype(), "fmax_cuda", [&]() {
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, FmaxFunctor<scalar_t>());
    });
  } else {
    maximum_kernel_cuda(iter);
  }
}

void fmin_kernel_cuda(TensorIteratorBase& iter) {
  if (isFloatingType(iter.common_dtype())) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.common_dtype(), "fmin_cuda", [&]() {
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, FminFunctor<scalar_t>());
    });
  } else {
    minimum_kernel_cuda(iter);
  }
}

REGISTER_DISPATCH(maximum_stub, &maximum_kernel_cuda)
REGISTER_DISPATCH(minimum_stub, &minimum_kernel_cuda)
REGISTER_DISPATCH(fmax_stub, &fmax_kernel_cuda)
REGISTER_DISPATCH(fmin_stub, &fmin_kernel_cuda)

} // namespace at::native

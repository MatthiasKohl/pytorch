#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/NumericUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/cuda/Loops.cuh>
#include <c10/core/Scalar.h>


namespace at::native {

template <typename scalar_t>
struct WhereFunctor {
  // only non simple cases are double and complex double with SM 90-
  template <int cc_major, int /*cc_minor*/>
  static constexpr bool is_simple = !(
    (std::is_same_v<scalar_t, double> && cc_major <= 9) ||
    std::is_same_v<scalar_t, c10::complex<double>>);

  GPU_LAMBDA scalar_t operator()(bool cond_val, scalar_t self_val, scalar_t other_val) const {
    return cond_val ? self_val : other_val;
  }
};

template <typename scalar_t>
struct IsPosInfFunctor {
  // always simple
  template <int /*cc_major*/, int /*cc_minor*/>
  static constexpr bool is_simple = true;

  GPU_LAMBDA bool operator()(scalar_t a) const {
    return a == std::numeric_limits<scalar_t>::infinity();
  }
};

template <typename scalar_t>
struct IsNegInfFunctor {
  // always simple
  template <int /*cc_major*/, int /*cc_minor*/>
  static constexpr bool is_simple = true;

  GPU_LAMBDA bool operator()(scalar_t a) const {
    return a == -std::numeric_limits<scalar_t>::infinity();
  }
};

template <typename scalar_t>
struct ClampFunctor {
  // only simple cases are float, short and int
  template <int /*cc_major*/, int /*cc_minor*/>
  static constexpr bool is_simple = (
    std::is_same_v<scalar_t, float> ||
    std::is_same_v<scalar_t, short> ||
    std::is_same_v<scalar_t, int>);

  GPU_LAMBDA scalar_t operator()(scalar_t v, scalar_t lower, scalar_t upper) const {
    scalar_t result = ::min(::max(v, lower), upper);

    result = at::_isnan(upper) ? upper : result;
    result = at::_isnan(lower) ? lower : result;
    result = at::_isnan(v) ? v : result;

    return result;
  }
};

template <typename scalar_t, typename opmath_t>
struct ClampScalarFunctor {
  // never simple, note: too much branching
  template <int /*cc_major*/, int /*cc_minor*/>
  static constexpr bool is_simple = false;

  GPU_LAMBDA scalar_t operator()(scalar_t v) const {
    if (_isnan(static_cast<opmath_t>(v))) {
      return v;
    } else if (minmax==at::native::detail::ClampLimits::Min){
      return ::max(static_cast<opmath_t>(v), lim0_val);
    } else if (minmax==at::native::detail::ClampLimits::Max){
      return ::min(static_cast<opmath_t>(v), lim0_val);
    } else {
      return ::min(::max(static_cast<opmath_t>(v), lim0_val), lim1_val);
    }
  }
  ClampScalarFunctor(opmath_t lim0_val_, opmath_t lim1_val_, at::native::detail::ClampLimits minmax_)
    : lim0_val(lim0_val_), lim1_val(lim1_val_), minmax(minmax_) {}
 private:
  opmath_t lim0_val;
  opmath_t lim1_val;
  at::native::detail::ClampLimits minmax;
};

namespace {

void where_kernel_impl(TensorIterator &iter) {
  AT_DISPATCH_V2(iter.dtype(), "where_cuda", [&] {
      gpu_kernel(iter, WhereFunctor<scalar_t>());
  },
  kComplexHalf, kHalf, kBFloat16, kBool, AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), AT_EXPAND(AT_FLOAT8_TYPES));
}

void isposinf_kernel_impl(TensorIteratorBase &iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.input_dtype(), "isposinf_cuda", [&]() {
    gpu_kernel(iter, IsPosInfFunctor<scalar_t>());
  });
}

void isneginf_kernel_impl(TensorIteratorBase &iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.input_dtype(), "isneginf_cuda", [&]() {
    gpu_kernel(iter, IsNegInfFunctor<scalar_t>());
  });
}

void clamp_kernel_impl(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.common_dtype(), "clamp_cuda", [&] {
    gpu_kernel(iter, ClampFunctor<scalar_t>());
  });
}

void inline launch_clamp_scalar(TensorIteratorBase& iter, Scalar lim0, Scalar lim1, at::native::detail::ClampLimits minmax){
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.common_dtype(), "clamp_scalar_cuda", [&] {
    using opmath_t = at::opmath_type<scalar_t>;
    auto lim0_val = lim0.to<opmath_t>();
    auto lim1_val = lim1.to<opmath_t>();

    gpu_kernel(iter, ClampScalarFunctor<scalar_t, opmath_t>(lim0_val, lim1_val, minmax));
  });
}


void clamp_scalar_kernel_impl(TensorIteratorBase& iter, const Scalar& min, const Scalar& max) {
  launch_clamp_scalar(iter, min, max, at::native::detail::ClampLimits::MinMax);
}

void clamp_min_scalar_kernel_impl(TensorIteratorBase& iter, Scalar min) {
  launch_clamp_scalar(iter, min, min, at::native::detail::ClampLimits::Min);
}

void clamp_max_scalar_kernel_impl(TensorIteratorBase& iter, Scalar max) {
  launch_clamp_scalar(iter, max, max, at::native::detail::ClampLimits::Max);
}

} // anonymous namespace


REGISTER_DISPATCH(where_kernel, &where_kernel_impl)
REGISTER_DISPATCH(isposinf_stub, &isposinf_kernel_impl)
REGISTER_DISPATCH(isneginf_stub, &isneginf_kernel_impl)
REGISTER_DISPATCH(clamp_stub, &clamp_kernel_impl)
REGISTER_DISPATCH(clamp_scalar_stub, &clamp_scalar_kernel_impl)
REGISTER_DISPATCH(clamp_min_scalar_stub, &clamp_min_scalar_kernel_impl)
REGISTER_DISPATCH(clamp_max_scalar_stub, &clamp_max_scalar_kernel_impl)

struct Msg {
 static constexpr size_t MAX_MSG_LENGTH = 256;
 char msg[MAX_MSG_LENGTH];
};
template <typename scalar_t>
__global__ void _assert_async_cuda_kernel(const scalar_t* input, Msg msg) {
  CUDA_KERNEL_ASSERT_MSG(input[0] != 0, msg.msg);
}

__global__ void _assert_async_cuda_kernel(const c10::complex<float>* input, Msg msg) {
  CUDA_KERNEL_ASSERT_MSG(input[0] != c10::complex<float>(0, 0), msg.msg);
}
__global__ void _assert_async_cuda_kernel(const c10::complex<double>* input, Msg msg) {
  CUDA_KERNEL_ASSERT_MSG(input[0] != c10::complex<double>(0, 0), msg.msg);
}

void _assert_async_msg_cuda(const Tensor& self_tensor, std::string_view assert_msg) {
  const TensorBase &self = get_tensor_base(self_tensor);
  auto n = self.numel();
  TORCH_CHECK(n != 0, "Boolean value of Tensor with no values is ambiguous");
  TORCH_CHECK(n < 2, "Boolean value of Tensor with more than one value is ambiguous");
  auto stream = at::cuda::getCurrentCUDAStream();
  Msg msg;
  size_t copy_length = assert_msg.length();
  TORCH_CHECK(copy_length < Msg::MAX_MSG_LENGTH - 1, "Message length must be smaller than " + std::to_string(Msg::MAX_MSG_LENGTH - 1));
  std::copy_n(assert_msg.data(), copy_length, msg.msg);
  msg.msg[copy_length] = '\0';  // Ensure null-termination
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16, self.scalar_type(), "_assert_async_cuda", [&] {
    _assert_async_cuda_kernel<<<1, 1, 0, stream>>>(self.const_data_ptr<scalar_t>(), msg);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

void _assert_async_cuda(const Tensor& self_tensor) {
  _assert_async_msg_cuda(self_tensor, "");
}

} // namespace at::native

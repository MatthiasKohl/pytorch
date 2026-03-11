#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/_empty_per_channel_affine_quantized.h>
#include <ATen/ops/_make_per_channel_quantized_tensor_native.h>
#include <ATen/ops/_make_per_tensor_quantized_tensor_native.h>
#include <ATen/ops/empty.h>
#endif

namespace at::native {

template <typename scalar_t, typename underlying_t>
struct AssignQuantizedFunctor {
  // always simple
  template <int /*cc_major*/, int /*cc_minor*/>
  static constexpr bool is_simple = true;

  GPU_LAMBDA scalar_t operator()(underlying_t value) const {
    return scalar_t(value);
  }
};

void assign_quantized_tensor_cuda(
  const Tensor& self, Tensor& dst) {
  AT_DISPATCH_QINT_TYPES(
      dst.scalar_type(), "assign_quantized_tensor_cuda", [&]() {
        auto iter = TensorIteratorConfig()
          .check_all_same_dtype(false)
          .add_output(dst)
          .add_input(self)
          .build();
        gpu_kernel(iter, AssignQuantizedFunctor<scalar_t, underlying_t>());
      });
}

Tensor make_per_tensor_quantized_tensor_cuda(
    const Tensor& self,
    double scale,
    int64_t zero_point) {
  Tensor dst = at::_empty_affine_quantized(
      self.sizes(),
      self.options().dtype(toQIntType(self.scalar_type())),
      scale,
      zero_point);
  assign_quantized_tensor_cuda(self, dst);
  return dst;
}

Tensor make_per_channel_quantized_tensor_cuda(
  const Tensor& self,
  const Tensor& scales,
  const Tensor& zero_points,
  int64_t axis) {
      Tensor dst = at::_empty_per_channel_affine_quantized(
      self.sizes(),
      scales,
      zero_points,
      axis,
      self.options().dtype(toQIntType(self.scalar_type())));
  assign_quantized_tensor_cuda(self, dst);
  return dst;
}

} // namespace at::native

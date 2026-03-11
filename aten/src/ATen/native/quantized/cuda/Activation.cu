#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

namespace at::native {

template <typename scalar_t, typename underlying_t>
struct ReluQuantizedFunctor {
  // always simple
  template <int /*cc_major*/, int /*cc_minor*/>
  static constexpr bool is_simple = true;

  GPU_LAMBDA scalar_t operator()(scalar_t value) const {
    return scalar_t(std::max<underlying_t>(value.val_, zero_point));
  }
  ReluQuantizedFunctor(int64_t zero_point_) : zero_point(zero_point_) {}
  private:
  int64_t zero_point;
};

Tensor& relu_quantized_cuda_(Tensor& self) {
  const auto zero_point = self.q_zero_point();
  AT_DISPATCH_QINT_TYPES(
    self.scalar_type(), "qrelu_cuda", [&]() {
      auto iter = TensorIterator::unary_op(self, self);
      gpu_kernel(iter, ReluQuantizedFunctor<scalar_t, underlying_t>(zero_point));
  });
  return self;
}

}  // namespace at::native

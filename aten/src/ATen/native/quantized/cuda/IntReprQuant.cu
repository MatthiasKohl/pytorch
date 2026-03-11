#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/int_repr_native.h>
#endif

namespace at::native {

template <typename scalar_t, typename underlying_t>
struct IntReprFunctor {
  // always simple
  template <int /*cc_major*/, int /*cc_minor*/>
  static constexpr bool is_simple = true;

  GPU_LAMBDA underlying_t operator()(scalar_t value) const {
    return value.val_;
  }
};

Tensor int_repr_quantized_cuda(const Tensor& self) {
  Tensor dst;
  AT_DISPATCH_QINT_TYPES(self.scalar_type(), "int_repr_quantized_cuda", [&]() {
    dst = at::empty(
        self.sizes(),
        self.options().dtype(UNDERLYING_TYPE),
        self.suggest_memory_format());
    auto iter = TensorIteratorConfig()
      .check_all_same_dtype(false)
      .add_output(dst)
      .add_input(self)
      .build();
    gpu_kernel(iter, IntReprFunctor<scalar_t, underlying_t>());
  });
  return dst;
}

} // namespace at::native

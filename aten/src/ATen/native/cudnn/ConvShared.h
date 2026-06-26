#pragma once
#include <ATen/core/Tensor.h>

#include <ATen/cuda/CUDAConfig.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/cudnn-wrapper.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/utils/ParamsHash.h>

#include <cstdint>

namespace at {
namespace native {

struct CuDNNStreamResource {
  uint32_t sm_count = 0;
  uint32_t sm_coscheduled_alignment = 0;
};

CuDNNStreamResource getCuDNNStreamResource(c10::DeviceIndex device_id);

template <typename Key>
decltype(auto) cudnn_cache_key_pod(Key& key) {
  if constexpr (requires { key.pod; }) {
    return (key.pod);
  } else {
    return (key);
  }
}

template <typename Key>
decltype(auto) cudnn_cache_key_pod(const Key& key) {
  if constexpr (requires { key.pod; }) {
    return (key.pod);
  } else {
    return (key);
  }
}

template <typename Params>
decltype(auto) cudnn_cache_key_resource_fields(Params& params) {
  if constexpr (requires {
                  params.params.sm_count;
                  params.params.sm_coscheduled_alignment;
                }) {
    return (params.params);
  } else {
    return (params);
  }
}

template <typename Params>
decltype(auto) cudnn_cache_key_resource_fields(const Params& params) {
  if constexpr (requires {
                  params.params.sm_count;
                  params.params.sm_coscheduled_alignment;
                }) {
    return (params.params);
  } else {
    return (params);
  }
}

template <typename Key>
decltype(auto) cudnn_cache_key_resource(Key& key) {
  return cudnn_cache_key_resource_fields(cudnn_cache_key_pod(key));
}

template <typename Key>
decltype(auto) cudnn_cache_key_resource(const Key& key) {
  return cudnn_cache_key_resource_fields(cudnn_cache_key_pod(key));
}

template <typename Key>
Key cudnn_cache_key_without_alignment(const Key& key) {
  Key normalized_key(key);
  cudnn_cache_key_resource(normalized_key).sm_coscheduled_alignment = 0;
  return normalized_key;
}

template <typename Key>
bool cudnn_cache_key_resource_compatible(
    const Key& cached_key,
    const Key& requested_key) {
  const auto& cached = cudnn_cache_key_resource(cached_key);
  const auto& requested = cudnn_cache_key_resource(requested_key);
  if (cached.sm_count != requested.sm_count) {
    return false;
  }
  if (cached.sm_coscheduled_alignment ==
      requested.sm_coscheduled_alignment) {
    return true;
  }
  if (cached.sm_coscheduled_alignment == 0 ||
      requested.sm_coscheduled_alignment == 0) {
    return false;
  }
  return cached.sm_coscheduled_alignment %
      requested.sm_coscheduled_alignment == 0;
}

// ---------------------------------------------------------------------
//
// Helper classes
//
// ---------------------------------------------------------------------

// This POD struct is used to let us easily compute hashes of the
// parameters
struct ConvolutionParams {
  c10::DeviceIndex device_id;
  cudnnDataType_t dataType;
  int input_size[2 + max_dim];
  uint8_t input_dim;
  at::MemoryFormat memory_format;
  int weight_size[2 + max_dim];
  int padding[max_dim];
  int stride[max_dim];
  int dilation[max_dim];
  int64_t groups;
  bool deterministic;
  bool allow_tf32;
  uint32_t sm_count;
  uint32_t sm_coscheduled_alignment;
  // NB: transposed purposely omitted: transposed just swaps
  // forward and backward, so you can reuse the benchmark entry,
};

std::ostream& operator<<(std::ostream& out, const ConvolutionParams& params);

// NB: This can't be a constructor, because then ConvolutionParams
// would not be a POD anymore.
// TODO: Use TensorGeometry here instead of the entire Tensor, which we
// don't actually need.  (OTOH: We can always pass in
// grad_input/grad_output, so this is not very pressing)
void setConvolutionParams(
    ConvolutionParams* params,
    const at::Tensor& input,
    const at::Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool deterministic,
    bool allow_tf32,
    at::MemoryFormat memory_format);

std::string repro_from_args(const ConvolutionParams& args);

// ---------------------------------------------------------------------
//
// Raw functions
//
// ---------------------------------------------------------------------

void raw_cudnn_convolution_forward_out(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32);

void raw_cudnn_convolution_backward_input_out(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32);

void raw_cudnn_convolution_backward_weight_out(
    const Tensor& grad_weight,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32);

void raw_cudnn_convolution_add_relu_out(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& z,
    float alpha,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32);

void raw_cudnn_convolution_add_relu_fallback_out(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& z,
    float alpha,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32);

#if AT_CUDNN_ENABLED()

// v7 functions are preserved here to allow for runtime switching to v7
// (e.g., TORCH_CUDNN_V8_API_DISABLED=1).
// Note that v7 forward/backward out can have different behavior from the v8
// versions, as v7 explicitly splits large tensors as a 32-bit indexing
// workaround whereas v8 expects cuDNN to handle large tensors.
void raw_cudnn_convolution_forward_out_v7(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32);

void raw_cudnn_convolution_backward_input_out_v7(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32);

void raw_cudnn_convolution_backward_weight_out_v7(
    const Tensor& grad_weight,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32);

void raw_cudnn_convolution_add_relu_out_v7(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& z,
    float alpha,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32);
#endif
} // namespace native
} // namespace at

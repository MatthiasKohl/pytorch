#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/native/cudnn/ConvShared.h>

using namespace at;
using namespace at::native;

TEST(CUDNNTest, CUDNNTestCUDA) {
  if (!at::cuda::is_available()) return;
  manual_seed(123);
}

namespace {

struct ResourceCacheTestParams {
  int payload;
  uint32_t sm_count;
  uint32_t sm_coscheduled_alignment;
};

struct ResourceCacheTestKey : ParamsWrapper<ResourceCacheTestParams> {
  ResourceCacheTestKey(
      int payload,
      uint32_t sm_count,
      uint32_t sm_coscheduled_alignment) {
    pod.payload = payload;
    pod.sm_count = sm_count;
    pod.sm_coscheduled_alignment = sm_coscheduled_alignment;
  }
};

} // namespace

TEST(CUDNNTest, ResourceCacheKeyCompatibility) {
  ResourceCacheTestKey cached(1, 16, 8);
  ResourceCacheTestKey exact(1, 16, 8);
  ResourceCacheTestKey multiple(1, 16, 4);
  ResourceCacheTestKey non_multiple(1, 16, 6);
  ResourceCacheTestKey different_sm_count(1, 8, 4);
  ResourceCacheTestKey zero_cached(1, 16, 0);
  ResourceCacheTestKey zero_requested(1, 16, 0);
  ResourceCacheTestKey nonzero_requested(1, 16, 4);
  ResourceCacheTestKey nonzero_cached(1, 16, 4);

  EXPECT_TRUE(cudnn_cache_key_resource_compatible(cached, exact));
  EXPECT_TRUE(cudnn_cache_key_resource_compatible(cached, multiple));
  EXPECT_FALSE(cudnn_cache_key_resource_compatible(cached, non_multiple));
  EXPECT_FALSE(
      cudnn_cache_key_resource_compatible(cached, different_sm_count));
  EXPECT_TRUE(
      cudnn_cache_key_resource_compatible(zero_cached, zero_requested));
  EXPECT_FALSE(
      cudnn_cache_key_resource_compatible(zero_cached, nonzero_requested));
  EXPECT_FALSE(
      cudnn_cache_key_resource_compatible(nonzero_cached, zero_requested));

  auto normalized = cudnn_cache_key_without_alignment(cached);
  EXPECT_EQ(normalized.pod.payload, cached.pod.payload);
  EXPECT_EQ(normalized.pod.sm_count, cached.pod.sm_count);
  EXPECT_EQ(normalized.pod.sm_coscheduled_alignment, 0);
}

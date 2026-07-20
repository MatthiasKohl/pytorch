#pragma once

#include <ATen/cuda/CUDAContext.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace at::cuda {

struct TORCH_CUDA_CPP_API VMMHalfSplitApiStat {
  uint64_t call_count = 0;
  uint64_t total_size_bytes = 0;
  std::map<size_t, uint64_t> size_histogram;
};

struct TORCH_CUDA_CPP_API VMMHalfSplitAllocatorStats {
  uint64_t requested_bytes = 0;
  uint64_t active_bytes = 0;
  uint64_t cached_mapped_bytes = 0;
  uint64_t reserved_bytes = 0;
  uint64_t peak_requested_bytes = 0;
  uint64_t peak_active_bytes = 0;
  uint64_t peak_cached_mapped_bytes = 0;
  uint64_t peak_reserved_bytes = 0;
  uint64_t allocation_requests = 0;
  uint64_t free_completions = 0;
  uint64_t mapped_slot_cache_hits = 0;
  uint64_t mapped_slot_creations = 0;
  uint64_t physical_handle_cache_hits = 0;
  uint64_t physical_handle_reuses = 0;
  uint64_t property_0_reserved_bytes = 0;
  uint64_t property_1_reserved_bytes = 0;
  uint64_t address_reservation_bytes = 0;
  std::map<std::string, VMMHalfSplitApiStat> api;
};

struct TORCH_CUDA_CPP_API VMMHalfSplitTraceEntry {
  std::string action;
  c10::DeviceIndex device = 0;
  uintptr_t address = 0;
  size_t size = 0;
  uintptr_t stream = 0;
  std::string user_metadata;
};

struct TORCH_CUDA_CPP_API VMMHalfSplitBlockSnapshot {
  uintptr_t address = 0;
  size_t size = 0;
  size_t requested_size = 0;
  std::string state;
};

struct TORCH_CUDA_CPP_API VMMHalfSplitSegmentSnapshot {
  c10::DeviceIndex device = 0;
  uintptr_t address = 0;
  size_t total_size = 0;
  uintptr_t stream = 0;
  std::vector<VMMHalfSplitBlockSnapshot> blocks;
};

struct TORCH_CUDA_CPP_API VMMHalfSplitAllocatorSnapshot {
  std::vector<VMMHalfSplitSegmentSnapshot> segments;
  std::vector<VMMHalfSplitTraceEntry> trace;
  VMMHalfSplitAllocatorStats stats;
  bool trace_overflow = false;
};

struct TORCH_CUDA_CPP_API VMMHalfSplitAllocation {
  void* ptr = nullptr;
  size_t size = 0;
  c10::DeviceIndex device = 0;
  cudaStream_t stream = nullptr;
};

class TORCH_CUDA_CPP_API VMMHalfSplitAllocatorCheckpointState final {
 public:
  ~VMMHalfSplitAllocatorCheckpointState() = default;

  VMMHalfSplitAllocatorCheckpointState(
      const VMMHalfSplitAllocatorCheckpointState&) = delete;
  VMMHalfSplitAllocatorCheckpointState& operator=(
      const VMMHalfSplitAllocatorCheckpointState&) = delete;

  const std::vector<VMMHalfSplitAllocation>& allocations() const;

 private:
  struct Impl;
  explicit VMMHalfSplitAllocatorCheckpointState(std::shared_ptr<const Impl>);

  std::shared_ptr<const Impl> impl_{};

  friend class VMMHalfSplitAllocator;
};

class TORCH_CUDA_CPP_API VMMHalfSplitAllocator final {
 public:
  explicit VMMHalfSplitAllocator(c10::DeviceIndex device);
  // NOLINTNEXTLINE(performance-trivially-destructible)
  ~VMMHalfSplitAllocator();

  VMMHalfSplitAllocator(const VMMHalfSplitAllocator&) = delete;
  VMMHalfSplitAllocator& operator=(const VMMHalfSplitAllocator&) = delete;

  void* allocate(size_t size, c10::DeviceIndex device, cudaStream_t stream);
  void free(
      void* ptr,
      size_t size,
      c10::DeviceIndex device,
      cudaStream_t stream);
  void emptyCache();
  void startRecording(size_t max_entries, bool clear_history = true);
  void stopRecording();
  void setUserMetadata(std::string metadata);
  VMMHalfSplitAllocatorSnapshot snapshot();
  VMMHalfSplitAllocatorStats stats();
  void resetPeakStats();
  void resetAccumulatedStats();
  c10::DeviceIndex device() const;
  std::shared_ptr<const VMMHalfSplitAllocatorCheckpointState>
  getCheckpointState();
  void setCheckpointState(
      const std::shared_ptr<const VMMHalfSplitAllocatorCheckpointState>& state);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_{};
};

TORCH_CUDA_CPP_API std::shared_ptr<VMMHalfSplitAllocator>
createVMMHalfSplitAllocator(c10::DeviceIndex device);

} // namespace at::cuda

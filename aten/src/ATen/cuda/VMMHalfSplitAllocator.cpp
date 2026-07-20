#include <ATen/cuda/VMMHalfSplitAllocator.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

#if !defined(USE_ROCM)
#include <cuda.h>
#endif

#ifndef ATEN_VMM_HALF_SPLIT_SUPPORTED
#if !defined(USE_ROCM) && !defined(_WIN32) && defined(CUDA_VERSION) && \
    CUDA_VERSION >= 13000
#define ATEN_VMM_HALF_SPLIT_SUPPORTED 1
#else
#define ATEN_VMM_HALF_SPLIT_SUPPORTED 0
#endif
#endif

#if ATEN_VMM_HALF_SPLIT_SUPPORTED
#include <c10/cuda/driver_api.h>
#endif

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <list>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace at::cuda {

struct VMMHalfSplitAllocatorCheckpointState::Impl {
  struct Slot {
    uintptr_t base = 0;
    size_t size = 0;
    size_t canonical_size = 0;
    size_t requested_size = 0;
    void* user_ptr = nullptr;
    uintptr_t stream = 0;
    bool active = false;
    std::shared_ptr<const void> pin{};
  };

  std::shared_ptr<const void> allocator_identity{};
  c10::DeviceIndex device = 0;
  std::vector<Slot> slots{};
  std::vector<VMMHalfSplitAllocation> allocations{};
};

VMMHalfSplitAllocatorCheckpointState::VMMHalfSplitAllocatorCheckpointState(
    std::shared_ptr<const Impl> impl)
    : impl_(std::move(impl)) {}

const std::vector<VMMHalfSplitAllocation>&
VMMHalfSplitAllocatorCheckpointState::allocations() const {
  return impl_->allocations;
}

#if ATEN_VMM_HALF_SPLIT_SUPPORTED

namespace {

constexpr size_t kMiB = size_t{1024} * 1024;
constexpr size_t kPatternAlignment = 4 * kMiB;
constexpr size_t kArenaAlignment = 64 * kMiB;
constexpr std::array<size_t, 6> kHandleSizes = {
    64 * kMiB, 32 * kMiB, 16 * kMiB, 8 * kMiB, 4 * kMiB, 2 * kMiB};
// The installed CUDA 13 headers predate names for these driver ABI values.
constexpr auto kLocalityDomainCount =
    static_cast<CUdevice_attribute>(149);
constexpr auto kDeviceLocalityDomain = static_cast<CUmemLocationType>(0x6);

size_t roundUp(size_t value, size_t alignment) {
  TORCH_CHECK(value <= std::numeric_limits<size_t>::max() - alignment + 1);
  return (value + alignment - 1) / alignment * alignment;
}

uintptr_t streamKey(cudaStream_t stream) {
  return reinterpret_cast<uintptr_t>(stream);
}

} // namespace

struct VMMHalfSplitAllocator::Impl {
  struct Handle {
    CUmemGenericAllocationHandle value = 0;
    size_t size = 0;
    int property = 0;
    uintptr_t stream = 0;
  };

  enum class SlotState { Active, Free };

  struct Slot {
    CUdeviceptr base = 0;
    size_t size = 0;
    size_t canonical_size = 0;
    size_t requested_size = 0;
    void* user_ptr = nullptr;
    uintptr_t stream = 0;
    SlotState state = SlotState::Active;
    std::vector<Handle> handles{};
    std::shared_ptr<const void> checkpoint_pin = std::make_shared<char>();
  };

  struct SlotLess {
    bool operator()(const Slot* lhs, const Slot* rhs) const {
      return std::tie(lhs->size, lhs->base) <
          std::tie(rhs->size, rhs->base);
    }
  };

  struct HandleCacheKey {
    uintptr_t stream = 0;
    int property = 0;
    size_t size = 0;

    bool operator<(const HandleCacheKey& other) const {
      return std::tie(stream, property, size) <
          std::tie(other.stream, other.property, other.size);
    }
  };

  explicit Impl(c10::DeviceIndex device) : device(device) {
    c10::cuda::CUDAGuard guard(device);
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    int driver_version = 0;
    C10_CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    TORCH_CHECK(
        driver_version >= 13040,
        "VMM half-split allocator requires a CUDA 13.4 or newer driver");

    int domain_count = 0;
    C10_CUDA_DRIVER_CHECK(
        c10::cuda::DriverAPI::get()->cuDeviceGetAttribute_(
            &domain_count, kLocalityDomainCount, device));
    TORCH_CHECK(
        domain_count >= 2,
        "VMM half-split allocator requires at least two locality domains");

    for (int property : {0, 1}) {
      auto prop = deviceAllocationProperty(property);
      size_t granularity = 0;
      C10_CUDA_DRIVER_CHECK(
          c10::cuda::DriverAPI::get()->cuMemGetAllocationGranularity_(
              &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
      TORCH_CHECK(
          granularity <= 2 * kMiB && 2 * kMiB % granularity == 0,
          "VMM half-split allocator requires a locality allocation "
          "granularity that divides 2 MiB, but locality domain ",
          property,
          " has granularity ",
          granularity);
    }

    const size_t total_memory = properties.totalGlobalMem;
    arena_size = roundUp(total_memory + total_memory / 8, kArenaAlignment);
    C10_CUDA_DRIVER_CHECK(
        c10::cuda::DriverAPI::get()->cuMemAddressReserve_(
            &arena_base, arena_size, kArenaAlignment, 0, 0));
    try {
      free_address_ranges.emplace(0, arena_size);
      recordApi("cuMemAddressReserve", arena_size);
      stats.address_reservation_bytes = arena_size;
    } catch (...) {
      (void)c10::cuda::DriverAPI::get()->cuMemAddressFree_(
          arena_base, arena_size);
      arena_base = 0;
      throw;
    }
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;

  ~Impl() noexcept {
    try {
      c10::cuda::CUDAGuard guard(device);
      (void)cudaDeviceSynchronize();
      for (auto& slot : slots) {
        if (slot->base) {
          (void)c10::cuda::DriverAPI::get()->cuMemUnmap_(
              slot->base, slot->size);
        }
        for (const Handle& handle : slot->handles) {
          (void)c10::cuda::DriverAPI::get()->cuMemRelease_(handle.value);
        }
      }
      for (const auto& [key, handles] : free_handles) {
        (void)key;
        for (CUmemGenericAllocationHandle handle : handles) {
          (void)c10::cuda::DriverAPI::get()->cuMemRelease_(handle);
        }
      }
      if (arena_base) {
        (void)c10::cuda::DriverAPI::get()->cuMemAddressFree_(
            arena_base, arena_size);
      }
    } catch (...) {
      return;
    }
  }

  CUmemAllocationProp deviceAllocationProperty(int property) const {
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = kDeviceLocalityDomain;
    static_assert(sizeof(prop.location.id) >= 2);
    // The locality ABI overlays two byte-sized IDs on CUmemLocation::id.
    const std::array<uint8_t, 2> localized = {
        static_cast<uint8_t>(device), static_cast<uint8_t>(property)};
    std::memcpy(&prop.location.id, localized.data(), localized.size());
    return prop;
  }

  void recordApi(const std::string& name, size_t size) noexcept {
    try {
      auto& api = stats.api[name];
      ++api.call_count;
      api.total_size_bytes += size;
      ++api.size_histogram[size];
    } catch (...) {
      return;
    }
  }

  void updatePeaks() {
    stats.peak_requested_bytes =
        std::max(stats.peak_requested_bytes, stats.requested_bytes);
    stats.peak_active_bytes =
        std::max(stats.peak_active_bytes, stats.active_bytes);
    stats.peak_cached_mapped_bytes =
        std::max(stats.peak_cached_mapped_bytes, stats.cached_mapped_bytes);
    stats.peak_reserved_bytes =
        std::max(stats.peak_reserved_bytes, stats.reserved_bytes);
  }

  void trace(
      std::string action,
      uintptr_t address,
      size_t size,
      uintptr_t stream) noexcept {
    if (!recording) {
      return;
    }
    if (trace_entries.size() >= max_trace_entries) {
      trace_overflow = true;
      return;
    }
    try {
      trace_entries.push_back(VMMHalfSplitTraceEntry{
          std::move(action), device, address, size, stream, user_metadata});
    } catch (...) {
      trace_overflow = true;
    }
  }

  size_t reserveAddress(size_t size) {
    auto best = free_address_ranges.end();
    for (auto it = free_address_ranges.begin();
         it != free_address_ranges.end();
         ++it) {
      if (it->second >= size &&
          (best == free_address_ranges.end() || it->second < best->second)) {
        best = it;
      }
    }
    TORCH_CHECK_WITH(
        OutOfMemoryError,
        best != free_address_ranges.end(),
        "VMM half-split allocator exhausted its virtual address reservation");
    const size_t offset = best->first;
    const size_t remaining = best->second - size;
    if (remaining) {
      free_address_ranges.emplace(offset + size, remaining);
    }
    free_address_ranges.erase(best);
    return offset;
  }

  void releaseAddress(size_t offset, size_t size) {
    auto next = free_address_ranges.lower_bound(offset);
    if (next != free_address_ranges.begin()) {
      auto previous = std::prev(next);
      if (previous->first + previous->second == offset) {
        offset = previous->first;
        size += previous->second;
        free_address_ranges.erase(previous);
      }
    }
    next = free_address_ranges.lower_bound(offset);
    if (next != free_address_ranges.end() && offset + size == next->first) {
      size += next->second;
      free_address_ranges.erase(next);
    }
    free_address_ranges.emplace(offset, size);
  }

  std::vector<std::pair<int, size_t>> handleLayout(size_t slot_size) const {
    std::vector<std::pair<int, size_t>> result;
    const size_t half = slot_size / 2;
    for (int property : {0, 1}) {
      size_t remaining = half;
      for (size_t handle_size : kHandleSizes) {
        while (remaining >= handle_size) {
          result.emplace_back(property, handle_size);
          remaining -= handle_size;
        }
      }
      TORCH_INTERNAL_ASSERT(remaining == 0);
    }
    return result;
  }

  Handle acquireHandle(uintptr_t stream, int property, size_t size) {
    const HandleCacheKey key{stream, property, size};
    auto& cached = free_handles[key];
    if (!cached.empty()) {
      CUmemGenericAllocationHandle handle = cached.back();
      cached.pop_back();
      ++stats.physical_handle_reuses;
      return Handle{handle, size, property, stream};
    }

    CUmemGenericAllocationHandle handle = 0;
    const auto prop = deviceAllocationProperty(property);
    const CUresult result =
        c10::cuda::DriverAPI::get()->cuMemCreate_(&handle, size, &prop, 0);
    TORCH_CHECK_WITH(
        OutOfMemoryError,
        result != CUDA_ERROR_OUT_OF_MEMORY,
        "CUDA out of memory creating a VMM physical allocation of ",
        size,
        " bytes");
    C10_CUDA_DRIVER_CHECK(result);
    recordApi("cuMemCreate", size);
    stats.reserved_bytes += size;
    if (property == 0) {
      stats.property_0_reserved_bytes += size;
    } else {
      stats.property_1_reserved_bytes += size;
    }
    updatePeaks();
    return Handle{handle, size, property, stream};
  }

  void releaseHandle(const Handle& handle) {
    C10_CUDA_DRIVER_CHECK(
        c10::cuda::DriverAPI::get()->cuMemRelease_(handle.value));
    recordApi("cuMemRelease", handle.size);
    stats.reserved_bytes -= handle.size;
    if (handle.property == 0) {
      stats.property_0_reserved_bytes -= handle.size;
    } else {
      stats.property_1_reserved_bytes -= handle.size;
    }
  }

  Slot* createSlot(size_t size, uintptr_t stream) {
    const auto layout = handleLayout(size);
    auto slot = std::make_unique<Slot>();
    slot->handles.reserve(layout.size());
    std::map<HandleCacheKey, size_t> handle_counts;
    for (const auto& [property, handle_size] : layout) {
      ++handle_counts[HandleCacheKey{stream, property, handle_size}];
    }
    for (const auto& [key, count] : handle_counts) {
      auto& cached = free_handles[key];
      cached.reserve(cached.capacity() + count);
    }

    const size_t offset = reserveAddress(size);
    const CUdeviceptr base = arena_base + offset;
    slot->base = base;
    slot->size = size;
    slot->stream = stream;

    const size_t reused_before = stats.physical_handle_reuses;
    size_t mapped_bytes = 0;
    try {
      for (const auto& [property, handle_size] : layout) {
        slot->handles.push_back(acquireHandle(stream, property, handle_size));
      }
      if (stats.physical_handle_reuses - reused_before == layout.size()) {
        ++stats.physical_handle_cache_hits;
      }
      for (const Handle& handle : slot->handles) {
        C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemMap_(
            base + mapped_bytes, handle.size, 0, handle.value, 0));
        recordApi("cuMemMap", handle.size);
        mapped_bytes += handle.size;
      }
      CUmemAccessDesc access{};
      access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      access.location.id = static_cast<int>(device);
      access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      C10_CUDA_DRIVER_CHECK(
          c10::cuda::DriverAPI::get()->cuMemSetAccess_(
              base, size, &access, 1));
      recordApi("cuMemSetAccess", size);
    } catch (...) {
      if (mapped_bytes) {
        (void)c10::cuda::DriverAPI::get()->cuMemUnmap_(base, mapped_bytes);
      }
      for (const Handle& handle : slot->handles) {
        (void)c10::cuda::DriverAPI::get()->cuMemRelease_(handle.value);
        stats.reserved_bytes -= handle.size;
        if (handle.property == 0) {
          stats.property_0_reserved_bytes -= handle.size;
        } else {
          stats.property_1_reserved_bytes -= handle.size;
        }
      }
      releaseAddress(offset, size);
      throw;
    }

    ++stats.mapped_slot_creations;
    trace("segment_alloc", base, size, stream);
    trace("segment_map", base, size, stream);
    slots.push_back(std::move(slot));
    return slots.back().get();
  }

  Slot* findReusableSlot(uintptr_t stream, size_t size) {
    auto& available = free_slots[stream];
    for (auto it = available.begin(); it != available.end(); ++it) {
      Slot* slot = *it;
      if (slot->size >= size) {
        available.erase(it);
        stats.cached_mapped_bytes -= slot->size;
        ++stats.mapped_slot_cache_hits;
        return slot;
      }
    }
    return nullptr;
  }

  void releaseFreeSlots() {
    for (auto it = slots.begin(); it != slots.end();) {
      Slot& slot = **it;
      if (slot.state != SlotState::Free ||
          slot.checkpoint_pin.use_count() != 1) {
        ++it;
        continue;
      }
      C10_CUDA_DRIVER_CHECK(
          c10::cuda::DriverAPI::get()->cuMemUnmap_(slot.base, slot.size));
      free_slots[slot.stream].erase(&slot);
      recordApi("cuMemUnmap", slot.size);
      trace("segment_unmap", slot.base, slot.size, slot.stream);
      trace("segment_free", slot.base, slot.size, slot.stream);
      stats.cached_mapped_bytes -= slot.size;
      for (const Handle& handle : slot.handles) {
        free_handles[HandleCacheKey{
                         slot.stream, handle.property, handle.size}]
            .push_back(handle.value);
      }
      releaseAddress(slot.base - arena_base, slot.size);
      it = slots.erase(it);
    }

    for (auto& [key, handles] : free_handles) {
      while (!handles.empty()) {
        const Handle handle{handles.back(), key.size, key.property, key.stream};
        releaseHandle(handle);
        handles.pop_back();
      }
    }
  }

  VMMHalfSplitAllocatorSnapshot makeSnapshot() const {
    VMMHalfSplitAllocatorSnapshot result;
    result.trace = trace_entries;
    result.stats = stats;
    result.trace_overflow = trace_overflow;
    result.segments.reserve(slots.size());
    for (const auto& slot : slots) {
      VMMHalfSplitSegmentSnapshot segment;
      segment.device = device;
      segment.address = slot->base;
      segment.total_size = slot->size;
      segment.stream = slot->stream;
      if (slot->state == SlotState::Free) {
        segment.blocks.push_back(VMMHalfSplitBlockSnapshot{
            slot->base, slot->size, 0, "inactive"});
      } else {
        const uintptr_t user_address =
            reinterpret_cast<uintptr_t>(slot->user_ptr);
        const size_t prefix_size = user_address - slot->base;
        if (prefix_size) {
          segment.blocks.push_back(VMMHalfSplitBlockSnapshot{
              slot->base, prefix_size, 0, "inactive"});
        }
        segment.blocks.push_back(VMMHalfSplitBlockSnapshot{
            user_address,
            slot->canonical_size,
            slot->requested_size,
            "active_allocated"});
        const size_t suffix_size =
            slot->size - prefix_size - slot->canonical_size;
        if (suffix_size) {
          segment.blocks.push_back(VMMHalfSplitBlockSnapshot{
              user_address + slot->canonical_size,
              suffix_size,
              0,
              "inactive"});
        }
      }
      result.segments.push_back(std::move(segment));
    }
    return result;
  }

  c10::DeviceIndex device;
  std::shared_ptr<const void> identity = std::make_shared<char>();
  CUdeviceptr arena_base = 0;
  size_t arena_size = 0;
  mutable std::mutex mutex{};
  std::list<std::unique_ptr<Slot>> slots{};
  std::unordered_map<void*, Slot*> live_allocations{};
  std::unordered_map<uintptr_t, std::set<Slot*, SlotLess>> free_slots{};
  std::map<HandleCacheKey, std::vector<CUmemGenericAllocationHandle>>
      free_handles{};
  std::map<size_t, size_t> free_address_ranges{};
  VMMHalfSplitAllocatorStats stats{};
  bool recording = false;
  bool trace_overflow = false;
  size_t max_trace_entries = 0;
  std::string user_metadata{};
  std::vector<VMMHalfSplitTraceEntry> trace_entries{};
};

VMMHalfSplitAllocator::VMMHalfSplitAllocator(c10::DeviceIndex device)
    : impl_(std::make_unique<Impl>(device)) {}

VMMHalfSplitAllocator::~VMMHalfSplitAllocator() = default;

void* VMMHalfSplitAllocator::allocate(
    size_t size,
    c10::DeviceIndex device,
    cudaStream_t stream) {
  TORCH_CHECK(
      device == impl_->device, "VMM half-split allocator device mismatch");
  if (size == 0) {
    return nullptr;
  }
  c10::cuda::CUDAGuard guard(device);
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const size_t canonical_size = roundUp(size, kPatternAlignment);
  Impl::Slot* slot = nullptr;
  slot = impl_->findReusableSlot(streamKey(stream), canonical_size);
  if (!slot) {
    slot = impl_->createSlot(canonical_size, streamKey(stream));
  }
  slot->state = Impl::SlotState::Active;
  slot->canonical_size = canonical_size;
  slot->requested_size = size;
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  slot->user_ptr = reinterpret_cast<void*>(
      slot->base + (slot->size - canonical_size) / 2);
  const auto insertion = impl_->live_allocations.emplace(slot->user_ptr, slot);
  TORCH_INTERNAL_ASSERT(insertion.second);
  ++impl_->stats.allocation_requests;
  impl_->stats.requested_bytes += size;
  impl_->stats.active_bytes += slot->size;
  impl_->updatePeaks();
  impl_->trace(
      "alloc",
      reinterpret_cast<uintptr_t>(slot->user_ptr),
      size,
      slot->stream);
  return slot->user_ptr;
}

void VMMHalfSplitAllocator::free(
    void* ptr,
    size_t size,
    c10::DeviceIndex device,
    cudaStream_t stream) {
  if (!ptr) {
    return;
  }
  TORCH_CHECK(
      device == impl_->device, "VMM half-split allocator device mismatch");
  c10::cuda::CUDAGuard guard(impl_->device);
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto allocation = impl_->live_allocations.find(ptr);
  TORCH_CHECK(
      allocation != impl_->live_allocations.end(),
      "invalid VMM half-split allocator pointer: ",
      ptr);
  Impl::Slot* slot = allocation->second;
  TORCH_CHECK(
      size == slot->requested_size,
      "VMM half-split allocator allocation size mismatch: expected ",
      slot->requested_size,
      ", got ",
      size);
  TORCH_CHECK(
      streamKey(stream) == slot->stream,
      "VMM half-split allocator allocation stream mismatch");
  const auto insertion = impl_->free_slots[slot->stream].insert(slot);
  TORCH_INTERNAL_ASSERT(insertion.second);
  impl_->live_allocations.erase(allocation);
  ++impl_->stats.free_completions;
  impl_->stats.requested_bytes -= slot->requested_size;
  impl_->stats.active_bytes -= slot->size;
  impl_->stats.cached_mapped_bytes += slot->size;
  slot->state = Impl::SlotState::Free;
  impl_->trace(
      "slot_cached",
      reinterpret_cast<uintptr_t>(slot->user_ptr),
      slot->requested_size,
      slot->stream);
  impl_->updatePeaks();
}

void VMMHalfSplitAllocator::emptyCache() {
  c10::cuda::CUDAGuard guard(impl_->device);
  std::lock_guard<std::mutex> lock(impl_->mutex);
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  impl_->releaseFreeSlots();
  impl_->updatePeaks();
}

void VMMHalfSplitAllocator::startRecording(
    size_t max_entries,
    bool clear_history) {
  TORCH_CHECK(max_entries > 0, "max_entries must be positive");
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (clear_history) {
    impl_->trace_entries.clear();
    impl_->trace_overflow = false;
  }
  impl_->trace_entries.reserve(max_entries);
  impl_->max_trace_entries = max_entries;
  impl_->recording = true;
}

void VMMHalfSplitAllocator::stopRecording() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->recording = false;
}

void VMMHalfSplitAllocator::setUserMetadata(std::string metadata) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->user_metadata = std::move(metadata);
}

VMMHalfSplitAllocatorSnapshot VMMHalfSplitAllocator::snapshot() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->makeSnapshot();
}

VMMHalfSplitAllocatorStats VMMHalfSplitAllocator::stats() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->stats;
}

void VMMHalfSplitAllocator::resetPeakStats() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->stats.peak_requested_bytes = impl_->stats.requested_bytes;
  impl_->stats.peak_active_bytes = impl_->stats.active_bytes;
  impl_->stats.peak_cached_mapped_bytes = impl_->stats.cached_mapped_bytes;
  impl_->stats.peak_reserved_bytes = impl_->stats.reserved_bytes;
}

void VMMHalfSplitAllocator::resetAccumulatedStats() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->stats.allocation_requests = 0;
  impl_->stats.free_completions = 0;
  impl_->stats.mapped_slot_cache_hits = 0;
  impl_->stats.mapped_slot_creations = 0;
  impl_->stats.physical_handle_cache_hits = 0;
  impl_->stats.physical_handle_reuses = 0;
  impl_->stats.api.clear();
}

c10::DeviceIndex VMMHalfSplitAllocator::device() const {
  return impl_->device;
}

std::shared_ptr<const VMMHalfSplitAllocatorCheckpointState>
VMMHalfSplitAllocator::getCheckpointState() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto checkpoint =
      std::make_shared<VMMHalfSplitAllocatorCheckpointState::Impl>();
  checkpoint->allocator_identity = impl_->identity;
  checkpoint->device = impl_->device;
  checkpoint->slots.reserve(impl_->slots.size());
  checkpoint->allocations.reserve(impl_->live_allocations.size());
  for (const auto& slot : impl_->slots) {
    const bool active = slot->state == Impl::SlotState::Active;
    checkpoint->slots.push_back(
        VMMHalfSplitAllocatorCheckpointState::Impl::Slot{
            slot->base,
            slot->size,
            active ? slot->canonical_size : 0,
            active ? slot->requested_size : 0,
            active ? slot->user_ptr : nullptr,
            slot->stream,
            active,
            slot->checkpoint_pin});
    if (active) {
      checkpoint->allocations.push_back(VMMHalfSplitAllocation{
          slot->user_ptr,
          slot->requested_size,
          impl_->device,
          reinterpret_cast<cudaStream_t>(slot->stream)});
    }
  }
  return std::shared_ptr<const VMMHalfSplitAllocatorCheckpointState>(
      new VMMHalfSplitAllocatorCheckpointState(std::move(checkpoint)));
}

void VMMHalfSplitAllocator::setCheckpointState(
    const std::shared_ptr<const VMMHalfSplitAllocatorCheckpointState>& state) {
  TORCH_CHECK(state, "VMM half-split allocator checkpoint state is null");
  const auto& checkpoint = state->impl_;
  TORCH_CHECK(
      checkpoint->allocator_identity == impl_->identity,
      "VMM half-split allocator checkpoint belongs to a different allocator");
  TORCH_CHECK(
      checkpoint->device == impl_->device,
      "VMM half-split allocator checkpoint device mismatch");

  std::lock_guard<std::mutex> lock(impl_->mutex);
  TORCH_CHECK(
      impl_->live_allocations.empty(),
      "VMM half-split allocator allocations must be freed before restoring a "
      "checkpoint");

  std::unordered_map<const void*, Impl::Slot*> current_slots;
  current_slots.reserve(impl_->slots.size());
  for (const auto& slot : impl_->slots) {
    TORCH_INTERNAL_ASSERT(slot->state == Impl::SlotState::Free);
    const auto insertion =
        current_slots.emplace(slot->checkpoint_pin.get(), slot.get());
    TORCH_INTERNAL_ASSERT(insertion.second);
  }

  std::unordered_map<
      const void*,
      const VMMHalfSplitAllocatorCheckpointState::Impl::Slot*>
      checkpoint_slots;
  checkpoint_slots.reserve(checkpoint->slots.size());
  std::vector<VMMHalfSplitAllocation> allocations;
  allocations.reserve(checkpoint->allocations.size());
  for (const auto& saved_slot : checkpoint->slots) {
    TORCH_CHECK(
        saved_slot.pin,
        "invalid VMM half-split allocator checkpoint slot pin");
    const auto insertion =
        checkpoint_slots.emplace(saved_slot.pin.get(), &saved_slot);
    TORCH_CHECK(
        insertion.second,
        "invalid VMM half-split allocator checkpoint with duplicate slots");
    const auto current = current_slots.find(saved_slot.pin.get());
    TORCH_CHECK(
        current != current_slots.end(),
        "VMM half-split allocator checkpoint references an unavailable slot");
    const Impl::Slot& slot = *current->second;
    TORCH_CHECK(
        saved_slot.base == slot.base && saved_slot.size == slot.size &&
            saved_slot.stream == slot.stream,
        "VMM half-split allocator checkpoint slot metadata mismatch");
    if (!saved_slot.active) {
      continue;
    }
    TORCH_CHECK(
        saved_slot.requested_size > 0 &&
            saved_slot.canonical_size ==
                roundUp(saved_slot.requested_size, kPatternAlignment) &&
            saved_slot.canonical_size <= saved_slot.size &&
            reinterpret_cast<uintptr_t>(saved_slot.user_ptr) ==
                saved_slot.base +
                    (saved_slot.size - saved_slot.canonical_size) / 2,
        "invalid VMM half-split allocator checkpoint allocation metadata");
    allocations.push_back(VMMHalfSplitAllocation{
        saved_slot.user_ptr,
        saved_slot.requested_size,
        checkpoint->device,
        reinterpret_cast<cudaStream_t>(saved_slot.stream)});
  }
  TORCH_CHECK(
      allocations.size() == checkpoint->allocations.size(),
      "invalid VMM half-split allocator checkpoint allocation manifest");
  for (size_t i = 0; i < allocations.size(); ++i) {
    const auto& actual = checkpoint->allocations[i];
    const auto& expected = allocations[i];
    TORCH_CHECK(
        actual.ptr == expected.ptr && actual.size == expected.size &&
            actual.device == expected.device &&
            actual.stream == expected.stream,
        "invalid VMM half-split allocator checkpoint allocation manifest");
  }

  std::unordered_map<void*, Impl::Slot*> live_allocations;
  live_allocations.reserve(checkpoint->allocations.size());
  std::unordered_map<uintptr_t, std::set<Impl::Slot*, Impl::SlotLess>>
      free_slots;
  uint64_t requested_bytes = 0;
  uint64_t active_bytes = 0;
  uint64_t cached_mapped_bytes = 0;
  for (const auto& slot : impl_->slots) {
    const auto saved = checkpoint_slots.find(slot->checkpoint_pin.get());
    if (saved != checkpoint_slots.end() && saved->second->active) {
      const auto& saved_slot = *saved->second;
      const auto insertion =
          live_allocations.emplace(saved_slot.user_ptr, slot.get());
      TORCH_CHECK(
          insertion.second,
          "invalid VMM half-split allocator checkpoint with duplicate "
          "allocation pointers");
      requested_bytes += saved_slot.requested_size;
      active_bytes += slot->size;
    } else {
      const auto insertion = free_slots[slot->stream].insert(slot.get());
      TORCH_INTERNAL_ASSERT(insertion.second);
      cached_mapped_bytes += slot->size;
    }
  }

  impl_->live_allocations.swap(live_allocations);
  impl_->free_slots.swap(free_slots);
  for (const auto& slot : impl_->slots) {
    const auto saved = checkpoint_slots.find(slot->checkpoint_pin.get());
    if (saved != checkpoint_slots.end() && saved->second->active) {
      const auto& saved_slot = *saved->second;
      slot->state = Impl::SlotState::Active;
      slot->canonical_size = saved_slot.canonical_size;
      slot->requested_size = saved_slot.requested_size;
      slot->user_ptr = saved_slot.user_ptr;
    } else {
      slot->state = Impl::SlotState::Free;
      slot->canonical_size = 0;
      slot->requested_size = 0;
      slot->user_ptr = nullptr;
    }
  }
  impl_->stats.requested_bytes = requested_bytes;
  impl_->stats.active_bytes = active_bytes;
  impl_->stats.cached_mapped_bytes = cached_mapped_bytes;
  impl_->updatePeaks();
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
std::shared_ptr<VMMHalfSplitAllocator> createVMMHalfSplitAllocator(
    c10::DeviceIndex device) {
  return std::make_shared<VMMHalfSplitAllocator>(device);
}

#else

namespace {

[[noreturn]] void reportUnsupportedVMMHalfSplitAllocator() {
  TORCH_CHECK(
      false,
      "VMMHalfSplitAllocator requires an NVIDIA CUDA 13.0 or newer build "
      "with CUDA Driver API support and locality-domain allocation APIs");
}

} // namespace

struct VMMHalfSplitAllocator::Impl {};

VMMHalfSplitAllocator::VMMHalfSplitAllocator(c10::DeviceIndex device)
    : impl_(nullptr) {
  (void)device;
  reportUnsupportedVMMHalfSplitAllocator();
}

VMMHalfSplitAllocator::~VMMHalfSplitAllocator() = default;

void* VMMHalfSplitAllocator::allocate(size_t, c10::DeviceIndex, cudaStream_t) {
  reportUnsupportedVMMHalfSplitAllocator();
}

void VMMHalfSplitAllocator::free(
    void*,
    size_t,
    c10::DeviceIndex,
    cudaStream_t) {
  reportUnsupportedVMMHalfSplitAllocator();
}

void VMMHalfSplitAllocator::emptyCache() {
  reportUnsupportedVMMHalfSplitAllocator();
}

void VMMHalfSplitAllocator::startRecording(size_t, bool) {
  reportUnsupportedVMMHalfSplitAllocator();
}

void VMMHalfSplitAllocator::stopRecording() {
  reportUnsupportedVMMHalfSplitAllocator();
}

void VMMHalfSplitAllocator::setUserMetadata(std::string) {
  reportUnsupportedVMMHalfSplitAllocator();
}

VMMHalfSplitAllocatorSnapshot VMMHalfSplitAllocator::snapshot() {
  reportUnsupportedVMMHalfSplitAllocator();
}

VMMHalfSplitAllocatorStats VMMHalfSplitAllocator::stats() {
  reportUnsupportedVMMHalfSplitAllocator();
}

void VMMHalfSplitAllocator::resetPeakStats() {
  reportUnsupportedVMMHalfSplitAllocator();
}

void VMMHalfSplitAllocator::resetAccumulatedStats() {
  reportUnsupportedVMMHalfSplitAllocator();
}

c10::DeviceIndex VMMHalfSplitAllocator::device() const {
  reportUnsupportedVMMHalfSplitAllocator();
}

std::shared_ptr<const VMMHalfSplitAllocatorCheckpointState>
VMMHalfSplitAllocator::getCheckpointState() {
  reportUnsupportedVMMHalfSplitAllocator();
}

void VMMHalfSplitAllocator::setCheckpointState(
    const std::shared_ptr<const VMMHalfSplitAllocatorCheckpointState>&) {
  reportUnsupportedVMMHalfSplitAllocator();
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
std::shared_ptr<VMMHalfSplitAllocator> createVMMHalfSplitAllocator(
    c10::DeviceIndex device) {
  return std::make_shared<VMMHalfSplitAllocator>(device);
}

#endif

} // namespace at::cuda

#include <torch/csrc/cuda/VMMHalfSplitAllocator.h>

#include <ATen/cuda/VMMHalfSplitAllocator.h>
#include <torch/csrc/cuda/CUDAPluggableAllocator.h>
#include <torch/csrc/utils/pybind.h>

#include <memory>
#include <unordered_map>
#include <utility>

namespace torch::cuda {
namespace {

using VMMEngine = at::cuda::VMMHalfSplitAllocator;
using AllocatorState = c10::cuda::CUDACachingAllocator::AllocatorState;
using CheckpointDelta = c10::cuda::CUDACachingAllocator::CheckpointDelta;
using AllocationMetadata = CUDAPluggableAllocator::_AllocationMetadata;
using PluggableAllocator =
    CUDAPluggableAllocator::CUDAPluggableManagedPoolAllocator;

class VMMHalfSplitManagedCheckpoint final : public AllocatorState {
 public:
  explicit VMMHalfSplitManagedCheckpoint(
      std::shared_ptr<const at::cuda::VMMHalfSplitAllocatorCheckpointState>
          state)
      : state_(std::move(state)) {}

  const std::shared_ptr<const at::cuda::VMMHalfSplitAllocatorCheckpointState>&
  state() const {
    return state_;
  }

 private:
  std::shared_ptr<const at::cuda::VMMHalfSplitAllocatorCheckpointState>
      state_{};
};

class VMMHalfSplitAllocatorBackend final : public PluggableAllocator {
 public:
  explicit VMMHalfSplitAllocatorBackend(std::shared_ptr<VMMEngine> engine)
      : PluggableAllocator(
            [owner = engine](size_t size, int device, cudaStream_t stream)
                -> void* { return owner->allocate(size, device, stream); },
            [owner = engine](
                void* ptr,
                size_t size,
                int device,
                cudaStream_t stream) {
              owner->free(ptr, size, device, stream);
            },
            [owner = engine] { owner->emptyCache(); }),
        engine_(std::move(engine)) {}

  std::shared_ptr<AllocatorState> getCheckpointState(
      c10::DeviceIndex /*device*/,
      c10::cuda::MempoolId_t /*mempool_id*/) override {
    return std::make_shared<VMMHalfSplitManagedCheckpoint>(
        engine_->getCheckpointState());
  }

  CheckpointDelta setCheckpointPoolState(
      c10::DeviceIndex /*device*/,
      std::shared_ptr<AllocatorState> state) override {
    auto vmm_state =
        std::dynamic_pointer_cast<const VMMHalfSplitManagedCheckpoint>(state);
    TORCH_CHECK(
        vmm_state, "Expected a VMM half-split allocator checkpoint state");

    std::unordered_map<void*, AllocationMetadata> metadata;
    const auto& allocations = vmm_state->state()->allocations();
    metadata.reserve(allocations.size());
    for (const auto& allocation : allocations) {
      const auto [_, inserted] = metadata.emplace(
          allocation.ptr,
          AllocationMetadata(
              allocation.size, allocation.device, allocation.stream));
      TORCH_CHECK(
          inserted,
          "VMM half-split allocator checkpoint contains duplicate pointers");
    }

    {
      std::lock_guard<std::mutex> lock(allocator_mutex_);
      TORCH_INTERNAL_ASSERT(allocation_metadata_.empty());
    }
    engine_->setCheckpointState(vmm_state->state());
    {
      std::lock_guard<std::mutex> lock(allocator_mutex_);
      TORCH_INTERNAL_ASSERT(allocation_metadata_.empty());
      allocation_metadata_.swap(metadata);
    }
    return {};
  }

 private:
  std::shared_ptr<VMMEngine> engine_{};
};

struct VMMHalfSplitAllocatorBinding {
  explicit VMMHalfSplitAllocatorBinding(c10::DeviceIndex device)
      : engine(at::cuda::createVMMHalfSplitAllocator(device)),
        allocator(std::make_shared<VMMHalfSplitAllocatorBackend>(engine)) {}

  std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator> cudaAllocator()
      const {
    return allocator;
  }

  std::shared_ptr<VMMEngine> engine{};
  std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator> allocator{};
};

py::dict apiStatToDict(const at::cuda::VMMHalfSplitApiStat& stat) {
  py::dict histogram;
  for (const auto& [size, count] : stat.size_histogram) {
    histogram[py::int_(size)] = count;
  }
  py::dict result;
  result["call_count"] = stat.call_count;
  result["total_size_bytes"] = stat.total_size_bytes;
  result["size_histogram"] = std::move(histogram);
  return result;
}

py::dict statsToDict(const at::cuda::VMMHalfSplitAllocatorStats& stats) {
  py::dict api;
  for (const auto& [name, stat] : stats.api) {
    api[py::str(name)] = apiStatToDict(stat);
  }
  py::dict result;
  result["requested_bytes"] = stats.requested_bytes;
  result["active_bytes"] = stats.active_bytes;
  result["cached_mapped_bytes"] = stats.cached_mapped_bytes;
  result["reserved_bytes"] = stats.reserved_bytes;
  result["peak_requested_bytes"] = stats.peak_requested_bytes;
  result["peak_active_bytes"] = stats.peak_active_bytes;
  result["peak_cached_mapped_bytes"] = stats.peak_cached_mapped_bytes;
  result["peak_reserved_bytes"] = stats.peak_reserved_bytes;
  result["allocation_requests"] = stats.allocation_requests;
  result["free_completions"] = stats.free_completions;
  result["mapped_slot_cache_hits"] = stats.mapped_slot_cache_hits;
  result["mapped_slot_creations"] = stats.mapped_slot_creations;
  result["physical_handle_cache_hits"] = stats.physical_handle_cache_hits;
  result["physical_handle_reuses"] = stats.physical_handle_reuses;
  result["property_0_reserved_bytes"] = stats.property_0_reserved_bytes;
  result["property_1_reserved_bytes"] = stats.property_1_reserved_bytes;
  result["address_reservation_bytes"] = stats.address_reservation_bytes;
  result["api"] = std::move(api);
  return result;
}

py::dict traceEntryToDict(const at::cuda::VMMHalfSplitTraceEntry& entry) {
  py::dict result;
  result["action"] = entry.action;
  result["device"] = entry.device;
  result["address"] = entry.address;
  result["size"] = entry.size;
  result["stream"] = entry.stream;
  result["user_metadata"] = entry.user_metadata;
  return result;
}

py::dict blockToDict(const at::cuda::VMMHalfSplitBlockSnapshot& block) {
  py::dict result;
  result["address"] = block.address;
  result["size"] = block.size;
  result["requested_size"] = block.requested_size;
  result["state"] = block.state;
  return result;
}

py::dict segmentToDict(const at::cuda::VMMHalfSplitSegmentSnapshot& segment) {
  py::list blocks;
  for (const auto& block : segment.blocks) {
    blocks.append(blockToDict(block));
  }
  py::dict result;
  result["device"] = segment.device;
  result["address"] = segment.address;
  result["total_size"] = segment.total_size;
  result["stream"] = segment.stream;
  result["blocks"] = std::move(blocks);
  return result;
}

py::dict snapshotToDict(
    const at::cuda::VMMHalfSplitAllocatorSnapshot& snapshot) {
  py::list segments;
  for (const auto& segment : snapshot.segments) {
    segments.append(segmentToDict(segment));
  }
  py::list trace;
  for (const auto& entry : snapshot.trace) {
    trace.append(traceEntryToDict(entry));
  }
  py::dict result;
  result["segments"] = std::move(segments);
  result["trace"] = std::move(trace);
  result["stats"] = statsToDict(snapshot.stats);
  result["trace_overflow"] = snapshot.trace_overflow;
  return result;
}

} // namespace

void initVMMHalfSplitAllocatorBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  py::class_<
      VMMHalfSplitAllocatorBinding,
      std::shared_ptr<VMMHalfSplitAllocatorBinding>>(
      m, "_VMMHalfSplitAllocator")
      .def(py::init<c10::DeviceIndex>(), py::arg("device"))
      .def_property_readonly(
          "allocator", &VMMHalfSplitAllocatorBinding::cudaAllocator)
      .def_property_readonly(
          "device",
          [](const VMMHalfSplitAllocatorBinding& self) {
            return self.engine->device();
          })
      .def(
          "start_recording",
          [](VMMHalfSplitAllocatorBinding& self,
             size_t max_entries,
             bool clear_history) {
            self.engine->startRecording(max_entries, clear_history);
          },
          py::arg("max_entries"),
          py::arg("clear_history") = true)
      .def(
          "stop_recording",
          [](VMMHalfSplitAllocatorBinding& self) {
            self.engine->stopRecording();
          })
      .def(
          "set_metadata",
          [](VMMHalfSplitAllocatorBinding& self, std::string metadata) {
            self.engine->setUserMetadata(std::move(metadata));
          })
      .def(
          "snapshot",
          [](VMMHalfSplitAllocatorBinding& self) {
            return snapshotToDict(self.engine->snapshot());
          })
      .def(
          "stats",
          [](VMMHalfSplitAllocatorBinding& self) {
            return statsToDict(self.engine->stats());
          })
      .def(
          "reset_peak_stats",
          [](VMMHalfSplitAllocatorBinding& self) {
            self.engine->resetPeakStats();
          })
      .def("reset_accumulated_stats", [](VMMHalfSplitAllocatorBinding& self) {
        self.engine->resetAccumulatedStats();
      });
}

} // namespace torch::cuda

#include <torch/csrc/python_headers.h>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/pybind.h>

#include <ATen/cuda/MemPool.h>
#include <c10/cuda/CUDACachingAllocator.h>

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void THCPMemPool_init(PyObject* module) {
  auto torch_C_m = py::handle(module).cast<py::module>();
  shared_ptr_class_<::at::cuda::MemPool>(torch_C_m, "_MemPool")
      .def(py::init(
          [](std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
                 allocator,
             bool is_user_created,
             bool use_on_oom,
             bool no_split,
             bool allocator_managed) {
            torch::utils::device_lazy_init(at::kCUDA);
            return std::make_shared<::at::cuda::MemPool>(
                std::move(allocator),
                is_user_created,
                use_on_oom,
                no_split,
                allocator_managed);
          }))
      .def_property_readonly("id", &::at::cuda::MemPool::id)
      .def("use_count", &::at::cuda::MemPool::use_count);
  torch_C_m.def(
      "_cuda_emptyAllocatorCache",
      [](const std::shared_ptr<::at::cuda::MemPool>& pool) {
        py::gil_scoped_release no_gil;
        c10::cuda::CUDACachingAllocator::emptyAllocatorCache(
            pool->device(), pool->id());
      });
}

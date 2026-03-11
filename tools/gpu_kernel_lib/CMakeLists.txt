# Standalone GPU kernel instantiation library.
#
# Compiles all elementwise CUDA kernel .cu files with a simplified
# gpu_kernel_impl that compiles __device__ functions for every functor
# instantiation.
#
# Usage: add_subdirectory(${CMAKE_SOURCE_DIR}/tools/gpu_kernel_lib
#            ${CMAKE_BINARY_DIR}/gpu_kernel_lib)
# from aten/src/ATen/CMakeLists.txt (after ATen_CUDA_INCLUDE is set).

if(NOT USE_CUDA)
  return()
endif()

set(GPU_KERNEL_UNROLL 4 CACHE STRING
    "Unroll factor for simple_gpu_kernel (<=1 uses scalar path)")

set(GPU_KERNEL_LIB_SHADOW_INCLUDE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")

# ---------------------------------------------------------------------------
# Collect the elementwise .cu files that use gpu_kernel (Loops.cuh).
# Scan the source tree AND the generated files from codegen.
# ---------------------------------------------------------------------------
set(_scan_dirs
    "${CMAKE_SOURCE_DIR}/aten/src/ATen/cuda"
    "${CMAKE_SOURCE_DIR}/aten/src/ATen/native/cuda"
    "${CMAKE_SOURCE_DIR}/aten/src/ATen/native/nested/cuda"
    "${CMAKE_SOURCE_DIR}/aten/src/ATen/native/quantized/cuda"
    "${CMAKE_SOURCE_DIR}/aten/src/ATen/native/sparse/cuda"
    "${CMAKE_SOURCE_DIR}/aten/src/ATen/native/transformers/cuda"
)

set(_all_cu)
foreach(_dir ${_scan_dirs})
  file(GLOB _cu "${_dir}/*.cu")
  list(APPEND _all_cu ${_cu})
endforeach()

# Keep only source-tree files that include Loops.cuh (directly or via known
# intermediary headers).
# note: TensorModeKernel.cu and NestedTensorMatmul.cu seem to produce SASS which
# nvdisasm cannot correctly disassemble for some architectures.
# This doesn't matter because these files don't actually call gpu_kernel,
# but we still look for files including e.g. TensorModeKernel.cuh as they could
# pull in Loops.cuh transitively this way.
set(SIMPLE_KERNEL_SRCS)
foreach(f ${_all_cu})
  file(STRINGS "${f}" _has_loops REGEX "#include.*cuda/Loops\\.cuh")
  file(STRINGS "${f}" _has_bininternal REGEX "#include.*BinaryInternal\\.h")
  file(STRINGS "${f}" _has_disttmpl REGEX "#include.*DistributionTemplates\\.h")
  file(STRINGS "${f}" _has_multitensor REGEX "#include.*MultiTensorApply\\.cuh")
  file(STRINGS "${f}" _has_tensormode REGEX "#include.*TensorModeKernel\\.cuh")
  if(_has_loops OR _has_bininternal OR _has_disttmpl OR _has_multitensor OR _has_tensormode)
    list(APPEND SIMPLE_KERNEL_SRCS "${f}")
  endif()
endforeach()

# Auto-generated CUDA sources from codegen (e.g. UfuncCUDA_*.cu).
# These files don't exist at configure time, so we can't glob or grep them.
# ${cuda_generated_sources} is set by Codegen.cmake (which runs its dry-run
# before this subdirectory is processed) and lists all generated file paths.
# All generated CUDA sources use Loops.cuh, so add them unconditionally.
foreach(f ${cuda_generated_sources})
  if(f MATCHES "\\.cu$")
    list(APPEND SIMPLE_KERNEL_SRCS "${f}")
  endif()
endforeach()

if(NOT SIMPLE_KERNEL_SRCS)
  message(WARNING "gpu_kernel_lib: no source files found")
  return()
endif()

list(LENGTH SIMPLE_KERNEL_SRCS _nsrcs)
message(STATUS "gpu_kernel_lib: building with ${_nsrcs} .cu files")

add_library(simple_gpu_kernels STATIC ${SIMPLE_KERNEL_SRCS})

# Ensure generated headers/sources are available before compilation.
if(TARGET ATEN_CPU_FILES_GEN_TARGET)
  add_dependencies(simple_gpu_kernels ATEN_CPU_FILES_GEN_TARGET)
endif()
if(TARGET ATEN_CUDA_FILES_GEN_TARGET)
  add_dependencies(simple_gpu_kernels ATEN_CUDA_FILES_GEN_TARGET)
endif()

set_target_properties(simple_gpu_kernels PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gpu_kernel_lib")

# Our shadow headers MUST come first so they override the real
# Loops.cuh / CUDALoops.cuh / JitLoops.cuh / jit_macros.h.
target_include_directories(simple_gpu_kernels BEFORE PRIVATE
    "${GPU_KERNEL_LIB_SHADOW_INCLUDE}")

# Then add the normal PyTorch include paths (+ generated include dir).
target_include_directories(simple_gpu_kernels PRIVATE
    ${ATen_CUDA_INCLUDE}
    "${CMAKE_BINARY_DIR}/aten/src")

# Inherit CUDA standard and flags from the main build.
set_property(TARGET simple_gpu_kernels PROPERTY CUDA_STANDARD ${CMAKE_CUDA_STANDARD})
set_property(TARGET simple_gpu_kernels PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Suppress linker warnings for unresolved symbols (we're a static lib
# that will never be linked into a real executable).
target_compile_definitions(simple_gpu_kernels PRIVATE
    GPU_KERNEL_UNROLL=${GPU_KERNEL_UNROLL})

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(simple_gpu_kernels PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()

#pragma once

#include <torch/csrc/python_headers.h>

namespace torch::cuda {

void initVMMHalfSplitAllocatorBindings(PyObject* module);

} // namespace torch::cuda

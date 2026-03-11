#pragma once
#include <string>

// Disable jiterator so all code paths use gpu_kernel (our simple version).
#define AT_USE_JITERATOR() 0
#define jiterator_stringify(...) std::string(#__VA_ARGS__);

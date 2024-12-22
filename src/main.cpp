#include <cstdint>
#include <iostream>

#include <nanobind/nanobind.h>
#include <drjit/jit.h>
#include <drjit/tensor.h>

#include "distance_marcher.h"

namespace nb = nanobind;
using namespace nb::literals;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

NB_MODULE(fastsweep_ext, m) {
    // Ensure Dr.Jit was initialized by importing it.
    nb::module_::import_("drjit");
    m.def("redistance", &redistance<dr::CUDAArray<float>>, "init_distance"_a);
    m.def("redistance", &redistance<dr::LLVMArray<float>>, "init_distance"_a);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

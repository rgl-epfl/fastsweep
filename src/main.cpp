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

static const char *docstring = "Re-distances the input distance field by solving the Eikonal equation";

NB_MODULE(fastsweep_ext, m) {
    // Ensure Dr.Jit was initialized by importing it.
    nb::module_::import_("drjit");
    m.def("redistance", &redistance<dr::CUDAArray<float>>, "distance_field"_a, docstring);
    m.def("redistance", &redistance<dr::LLVMArray<float>>, "distance_field"_a, docstring);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

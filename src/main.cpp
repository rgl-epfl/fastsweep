#include <cstdint>

#include <pybind11/pybind11.h>

#include "distance_marcher.h"

#include <drjit/jit.h>
#include <drjit/tensor.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(_fastsweep_core, m) {

    // Import DrJit python modules to initialize the JIT
    py::module drjit = py::module::import("drjit");
    m.def("redistance", &redistance<dr::CUDAArray<float>>);
    m.def("redistance", &redistance<dr::LLVMArray<float>>);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

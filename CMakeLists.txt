cmake_minimum_required(VERSION 3.15...3.19)

project(fastsweep DESCRIPTION "Fast sweep solver")

option(FASTWEEPING_USE_CUDA "Use high-performance CUDA kernels instead of
                             Dr.Jit's GPU mode" ON)

if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")

endif()

find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

find_package(nanobind CONFIG REQUIRED)

include(ext/cmake-defaults/CMakeLists.txt)

# Find the drjit package
if("${FASTSWEEP_DRJIT_CMAKE_DIR}" STREQUAL "")
  execute_process(
    COMMAND
    "${PYTHON_EXECUTABLE}" -c
    "import drjit; print(drjit.get_cmake_dir())"
    OUTPUT_VARIABLE FASTSWEEP_DRJIT_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
endif()
list(APPEND CMAKE_PREFIX_PATH "${FASTSWEEP_DRJIT_CMAKE_DIR}")
find_package(drjit CONFIG REQUIRED)

if (FASTWEEPING_USE_CUDA)
  add_definitions(-DFASTSWEEPING_USE_CUDA)
endif()

nanobind_add_module(fastsweep_ext NB_DOMAIN drjit NB_STATIC src/main.cpp src/distance_marcher.cpp)

target_link_libraries(fastsweep_ext PUBLIC drjit-core)
target_compile_features(fastsweep_ext PRIVATE cxx_std_17)
target_compile_definitions(fastsweep_ext PRIVATE VERSION_INFO=${SKBUILD_PROJECT_VERSION})

install(TARGETS fastsweep_ext DESTINATION fastsweep)

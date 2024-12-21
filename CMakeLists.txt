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

# Import nanobind through CMake's find_package mechanism
find_package(nanobind CONFIG REQUIRED)

include(cmake-defaults/CMakeLists.txt)

# Find the drjit package
if (APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  # When cross-compiling on MacOS, we cannot run drjit, but rather need to pass the relevant
  # path into cmake
  set(drjit_DIR ${DRJIT_CMAKE_DIR})
else()
  execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "import drjit;print(drjit.get_cmake_dir())" OUTPUT_VARIABLE drjit_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

find_package(drjit)
if (NOT ${drjit_FOUND})
  message(FATAL_ERROR "Dr.Jit not found. Please install Dr.Jit using \"pip install drjit\"")
endif()

if (FASTWEEPING_USE_CUDA)
  add_definitions(-DFASTSWEEPING_USE_CUDA)
endif()

include_directories(${drjit_INCLUDE_DIR})

nanobind_add_module(_fastsweep_core STABLE_ABI NB_STATIC src/main.cpp src/distance_marcher.cpp)

target_link_libraries(_fastsweep_core PUBLIC nanothread drjit-core)
target_compile_features(_fastsweep_core PRIVATE cxx_std_17)
target_include_directories(_fastsweep_core PUBLIC ext/drjit/include)
target_compile_definitions(_fastsweep_core PRIVATE VERSION_INFO=${PROJECT_VERSION_INFO})

install(TARGETS _fastsweep_core DESTINATION .)

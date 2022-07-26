# Fast sweeping SDF solver
<p align="center">
<img src="https://raw.githubusercontent.com/rgl-epfl/fastsweep/main/redistancing.svg" width="70%" centering/>
</p>

This repository contains a Python package providing an efficient solver for the Eikonal equation in 3D. The primary use for this package is to *redistance* a signed distance function (SDF) from its zero level set (e.g., during an optimization that optimizes the SDF). In particular, this implementation was created for the use in our paper on [differentiable signed distance function rendering](http://rgl.epfl.ch/publications/Vicini2022SDF). You can find the
code for that paper [here](https://github.com/rgl-epfl/differentiable-sdf-rendering.git).

This library does **not** convert meshes to SDFs, even though it can be used for such applications. This implementation runs efficiently on GPUs (using CUDA) and also provides a CPU implementation as a fallback. The solver is exposed via Python bindings and uses [Dr.Jit](https://github.com/mitsuba-renderer/drjit) for some of its implementation.

The code implements the parallel fast sweeping algorithm for the Eikonal equation:

*A parallel fast sweeping method for the Eikonal equation. Miles Detrixhe,  Frédéric Gibou, Chohong Min, Journal of Computational Physics 237 (2013)*

The implementation is in part based on [PDFS](https://github.com/GEM3D/PDFS), see also `LICENSE` for license details.

# Installation
Pre-build binaries are provided on PyPi and can be installed using
```bash
pip install fastsweep
```

Alternatively, the package is also relatively easy to build and install from source. The build setup uses CMake and [scikit build](https://scikit-build.readthedocs.io/en/latest/). Please clone the repository including submodules using
```bash
git clone --recursive git@github.com:rgl-epfl/fastsweep.git
```
The Python module can then be built and installed by invoking:
```bash
pip install ./fastsweep
```

**Important**: It is important that this solver and `drjit` are compiled with exactly the same compiler and settings for binary compatibility. If you installed a pre-built `drjit` package using `pip`, you most likely will want to use the pre-built package for `fastsweep` as well. Conversely, if you want to compile one of these packages locally, you will most likely need to compile the other one locally as well. If
there is a problem with binary compatibility, invoking the functionality of the solver will most likely throw a type-mismatch error.


# Usage
The solver takes a Dr.Jit 3D `TensorXf` as input and solves the Eikonal equation from its zero level set. It returns a valid SDF that reproduces the zero level set of the input. The solver does not support 1D or 2D problems, for these one can for example use [scikit-fmm](https://pythonhosted.org/scikit-fmm/).

Given an initial 3D tensor, the solver can be invoked as
```Python
import fastsweep

data = drjit.cuda.TensorXf(...)
sdf = fastsweep.redistance(data)
```
The resulting array `sdf` is then a valid SDF. The solver returns either a `drjit.cuda.TensorXf` or `dfjit.llvm.TensorXf`, depending on the type of the input. A complete
example script is provided [here](https://github.com/rgl-epfl/fastsweep/blob/main/python/example.py).

# Limitations
- The code currently assumes the SDF to be contained in the unit cube volume and hasn't been tested for non-uniform volumes or other scales.
- The CPU version isn't very efficient, this code is primarily designed for GPU execution and the CPU version is really just a fallback.
- The computation of the zero level set does not consider different grid interpolation modes.

# Citation
If you use this solver for an academic paper, consider citing the following paper:
```bibtex
@article{Vicini2022sdf,
    title   = {Differentiable Signed Distance Function Rendering},
    author  = {Delio Vicini and Sébastien Speierer and Wenzel Jakob},
    year    = 2022,
    month   = jul,
    journal = {Transactions on Graphics (Proceedings of SIGGRAPH)},
    volume  = 41,
    number  = 4,
    pages   = {125:1--125:18},
    doi     = {10.1145/3528223.3530139}
}
```

#include "../kernels/cuda_kernels.h"

// This file looks up various cuda driver calls
// and loads the precompiled kernels from "cuda_kernels.h"

#define CUDA_ERROR_DEINITIALIZED 4
#define CUDA_SUCCESS 0

using CUcontext  = struct CUctx_st *;
using CUmodule   = struct CUmod_st *;
using CUfunction = struct CUfunc_st *;
using CUstream   = struct CUstream_st *;
using CUresult   = int;

CUresult (*cuGetErrorName)(CUresult, const char **)                   = nullptr;
CUresult (*cuGetErrorString)(CUresult, const char **)                 = nullptr;
CUresult (*cuModuleLoadData)(CUmodule *, const void *)                = nullptr;
CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *) = nullptr;
CUresult (*cuCtxSynchronize)()                                        = nullptr;
CUresult (*cuLaunchKernel)(CUfunction f, unsigned int, unsigned int, unsigned int, unsigned int,
                           unsigned int, unsigned int, unsigned int, CUstream, void **,
                           void **)                                   = nullptr;
CUresult (*cuCtxPushCurrent)(CUcontext)                               = nullptr;
CUresult (*cuCtxPopCurrent)(CUcontext *)                              = nullptr;

/// Assert that a CUDA operation is correctly issued
#define cuda_check(err) cuda_check_impl(err, __FILE__, __LINE__)

void cuda_check_impl(CUresult errval, const char *file, const int line) {
    if (errval != CUDA_SUCCESS && errval != CUDA_ERROR_DEINITIALIZED) {
        const char *name = nullptr, *msg = nullptr;
        cuGetErrorName(errval, &name);
        cuGetErrorString(errval, &msg);
        fprintf(stderr,
                "cuda_check(): API error = %04d (%s): \"%s\" in "
                "%s:%i.\n",
                (int) errval, name, msg, file, line);
    }
}

struct scoped_set_context {
    scoped_set_context(CUcontext ctx) { cuda_check(cuCtxPushCurrent(ctx)); }
    ~scoped_set_context() { cuda_check(cuCtxPopCurrent(nullptr)); }
};

CUmodule cu_module;
CUfunction fast_sweep_kernel;
bool init = false;

void cuda_load_kernels() {
    if (init)
        return;
    cuGetErrorName      = decltype(cuGetErrorName)(jit_cuda_lookup("cuGetErrorName"));
    cuGetErrorString    = decltype(cuGetErrorString)(jit_cuda_lookup("cuGetErrorString"));
    cuModuleLoadData    = decltype(cuModuleLoadData)(jit_cuda_lookup("cuModuleLoadData"));
    cuModuleGetFunction = decltype(cuModuleGetFunction)(jit_cuda_lookup("cuModuleGetFunction"));
    cuCtxPushCurrent    = decltype(cuCtxPushCurrent)(jit_cuda_lookup("cuCtxPushCurrent_v2"));
    cuCtxPopCurrent     = decltype(cuCtxPopCurrent)(jit_cuda_lookup("cuCtxPopCurrent_v2"));
    cuCtxSynchronize    = decltype(cuCtxSynchronize)(jit_cuda_lookup("cuCtxSynchronize"));
    cuLaunchKernel      = decltype(cuLaunchKernel)(jit_cuda_lookup("cuLaunchKernel_ptsz"));
    CUcontext ctx       = CUcontext(jit_cuda_context());
    scoped_set_context guard(ctx);
    cuda_check(cuModuleLoadData(&cu_module, (void *) imageBytes));
    cuda_check(cuModuleGetFunction(&fast_sweep_kernel, cu_module, (char *) "fast_sweep_kernel"));
    init = true;
}
#pragma once

#ifdef __CUDACC__
#define DEVICE __device__ __host__
#else
#define DEVICE
#endif

template <typename T> struct Vec3 {
    T x, y, z;
    DEVICE Vec3() {}
    DEVICE Vec3(T x, T y, T z) : x(x), y(y), z(z) {}
};

using Vec3i = Vec3<int>;
using Vec3d = Vec3<double>;

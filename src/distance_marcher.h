#pragma once

#include <math.h>
#include <vector>

#include <drjit/array.h>
#include <drjit/jit.h>
#include <drjit/tensor.h>

#define BORDER_SIZE 1
#define DEFAULT_INTERIOR_DISTANCE 90000

namespace dr = drjit;

using ScalarVector3i = dr::Array<int32_t, 3>;
using ScalarVector3u = dr::Array<uint32_t, 3>;
using ScalarVector3f = dr::Array<float, 3>;

template<typename Float>
dr::Tensor<Float> redistance(const dr::Tensor<Float> &init_distance);

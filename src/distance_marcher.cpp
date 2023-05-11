#include "distance_marcher.h"

#include <tuple>

#include <drjit/util.h>

#include "../kernels/vec.h"
#include "cuda_helpers.h"

#ifdef FASTSWEEPING_USE_CUDA
constexpr bool USE_CUDA = true;
#else
constexpr bool USE_CUDA = false;
#endif


int int_div_up(int a, int b) { return ((a % b) != 0) ? (a / b + 1) : (a / b); }

std::pair<Vec3i, Vec3i> get_launch_parameters(int x_res, int y_res) {
    int total_num_threads = x_res * y_res;
    Vec3i block_size(16, 16, 1);
    if (total_num_threads < 256)
        block_size = Vec3i(x_res, y_res, 1);
    Vec3i grid_size(int_div_up(x_res, block_size.x), int_div_up(y_res, block_size.y), 1);
    return {grid_size, block_size};
}

template <typename T> std::tuple<T, T, T> meshgrid3d(const T &x, const T &y, const T &z) {
    uint32_t sizes[3] = { (uint32_t) x.size(), (uint32_t) y.size(), (uint32_t) z.size() };
    dr::Array<T, 3> args(x, y, z);
    uint32_t size               = sizes[0] * sizes[1] * sizes[2];
    dr::uint32_array_t<T> index = dr::arange<dr::uint32_array_t<T>>(size);
    dr::Array<T, 3> result;
    for (size_t i = 0; i < 3; i++) {
        size         = size / sizes[i];
        auto index_v = index / size;
        index        = dr::fnmadd(index_v, size, index);
        result[i]    = dr::gather<T>(args[i], index_v);
    }
    return { result[0], result[1], result[2] };
}

template <typename Int32>
Int32 idx(const Int32 &x, const Int32 &y, const Int32 &z,
          const dr::Array<dr::scalar_t<Int32>, 3> &shape) {
    return z * shape[1] * shape[0] + y * shape[0] + x;
}

template <typename Float, typename Int32 = dr::int32_array_t<Float>>
std::pair<dr::Tensor<dr::float64_array_t<Float>>, dr::Tensor<Int32>>
initialize_distance(const dr::Tensor<Float> &init_distance) {
    using Bool     = dr::mask_t<Float>;
    using UInt32   = dr::uint32_array_t<Float>;
    using Int8     = dr::int8_array_t<Float>;
    using UInt8    = dr::uint8_array_t<Float>;
    using Float64  = dr::float64_array_t<Float>;
    using Vector3i = dr::Array<Int32, 3>;
    using Vector3f = dr::Array<Float, 3>;

    // The implementation follows the code in https://github.com/scikit-fmm/scikit-fmm
    ScalarVector3i shape_in(
        init_distance.shape()[0], init_distance.shape()[1], init_distance.shape()[2]);
    ScalarVector3i shape = shape_in + 2 * BORDER_SIZE;

    auto [z, y, x] = meshgrid3d(
        dr::arange<Int32>(shape[0]), dr::arange<Int32>(shape[1]), dr::arange<Int32>(shape[2]));
    Vector3i coord(x, y, z);

    Vector3f ldistance(dr::Infinity<Float>);

    // Assume SDF is in [0,1]
    Bool active = dr::all((coord >= BORDER_SIZE) & (coord < shape - BORDER_SIZE));

    auto linear_idx = idx(
        coord[0] - BORDER_SIZE, coord[1] - BORDER_SIZE, coord[2] - BORDER_SIZE, shape_in);
    Float64 init_phi_v(dr::gather<Float>(init_distance.array(), linear_idx, active));
    Vector3f deltas = 1 / Vector3f(shape_in);
    Bool borders    = dr::zeros<Bool>(dr::prod(shape));
    for (int dim = 0; dim < 3; ++dim) {
        for (int j = -1; j < 2; j += 2) {
            Vector3i offset_coord(coord);
            offset_coord[dim] += j;
            Bool valid = active & (offset_coord[dim] >= BORDER_SIZE) &
                         (offset_coord[dim] < shape[dim] - BORDER_SIZE);
            Float init_phi_n = dr::gather<Float>(
                init_distance.array(),
                idx(offset_coord[0] - BORDER_SIZE, offset_coord[1] - BORDER_SIZE,
                    offset_coord[2] - BORDER_SIZE, shape_in),
                valid);
            valid &= init_phi_v * init_phi_n < 0;
            borders |= valid;
            Float64 d = deltas[dim] * init_phi_v / (init_phi_v - Float64(init_phi_n));
            ldistance[dim][valid & (ldistance[dim] > d)] = d;
        }
    }

    // If we actually are on  the transition zone
    Float64 dsum     = dr::sum(dr::select(ldistance > 0, 1 / dr::sqr(ldistance), 0));
    Bool zero_init   = active & dr::eq(init_phi_v, 0.0);
    Float64 distance = dr::select(zero_init | ~borders, 0.0, dr::sqrt(1 / dsum));
    Int32 frozen     = dr::select(borders | zero_init, 1, 0);

    // High default value for non-frozen nodes
    distance[~(borders | zero_init)] = 90000;

    size_t tensor_shape[3] = { (size_t) shape[0], (size_t) shape[1], (size_t) shape[2] };
    return { dr::Tensor<Float64>(distance, 3, tensor_shape),
             dr::Tensor<Int32>(frozen, 3, tensor_shape) };
}

template <typename Float, typename Vector3f = dr::Array<Float, 3>>
Float solve_eikonal(const Float &cur_dist, Vector3f &m, const ScalarVector3f &dx) {

    using Bool     = dr::mask_t<Float>;
    using Int32    = dr::int32_array_t<Float>;
    using UInt32   = dr::uint32_array_t<Float>;
    using Vector3i = dr::Array<Int32, 3>;

    Vector3f d(dx);
    // Sort according to m
    for (int i = 1; i < 3; i++) {
        for (int j = 0; j < 3 - i; j++) {
            Float tmp_m(m[j]);
            Float tmp_d(d[j]);
            Bool swap                  = m[j] > m[j + 1];
            dr::masked(m[j], swap)     = m[j + 1];
            dr::masked(d[j], swap)     = d[j + 1];
            dr::masked(m[j + 1], swap) = tmp_m;
            dr::masked(d[j + 1], swap) = tmp_d;
        }
    }

    // Solve the eikonal equation locally to update distance
    auto m2_0      = dr::sqr(m[0]);
    auto m2_1      = dr::sqr(m[1]);
    auto m2_2      = dr::sqr(m[2]);
    auto d2_0      = dr::sqr(d[0]);
    auto d2_1      = dr::sqr(d[1]);
    auto d2_2      = dr::sqr(d[2]);
    Float dist_new = m[0] + d[0];
    Float s        = dr::sqrt(-m2_0 + 2 * m[0] * m[1] - m2_1 + d2_0 + d2_1);

    dr::masked(dist_new, dist_new > m[1]) = (m[1] * d2_0 + m[0] * d2_1 + d[0] * d[1] * s) /
                                            (d2_0 + d2_1);
    Float a = dr::sqrt(-m2_0 * d2_1 - m2_0 * d2_2 + 2 * m[0] * m[1] * d2_2 - m2_1 * d2_0 -
                       m2_1 * d2_2 + 2 * m[0] * m[2] * d2_1 - m2_2 * d2_0 - m2_2 * d2_1 +
                       2 * m[1] * m[2] * d2_0 + d2_0 * d2_1 + d2_0 * d2_2 + d2_1 * d2_2);
    dr::masked(dist_new, dist_new > m[2]) =
        (m[2] * d2_0 * d2_1 + m[1] * d2_0 * d2_2 + m[0] * d2_1 * d2_2 + d[0] * d[1] * d[2] * a) /
        (d2_0 * d2_1 + d2_0 * d2_2 + d2_1 * d2_2);
    return dr::minimum(cur_dist, dist_new);
}

template <typename Float, typename Int32 = dr::int32_array_t<Float>,
          typename Vector3i = dr::Array<Int32, 3>, typename UInt8 = dr::uint8_array_t<Float>>
void sweep_step(const Int32 &x, const Int32 &y,
                const dr::Tensor<dr::float64_array_t<Float>> &distance,
                dr::Tensor<dr::float64_array_t<Float>> &distance_target,
                const dr::Tensor<Int32> &frozen, const Vector3i &shape_in, const Int32 &level,
                const Vector3i &sweep_offset, const ScalarVector3f &dx) {

    using Bool     = dr::mask_t<Float>;
    using UInt32   = dr::uint32_array_t<Float>;
    using Float64  = dr::float64_array_t<Float>;
    using Vector3d = dr::Array<Float64, 3>;

    Int32 z     = level - x - y;
    Bool active = (x <= shape_in.x()) & (y <= shape_in.y()) & (z > 0) & (z <= shape_in.z());

    Int32 i = dr::abs(z - sweep_offset.z());
    Int32 j = dr::abs(y - sweep_offset.y());
    Int32 k = dr::abs(x - sweep_offset.x());

    Vector3i shape(shape_in);
    shape += 2 * BORDER_SIZE;

    Int32 linear_idx = i * shape.y() * shape.x() + j * shape.x() + k;
    Int32 frozen_v   = dr::gather<Int32>(frozen.array(), linear_idx, active);
    active &= dr::neq(frozen_v, 1);

    auto &dist_data = distance.array();
    Float64 center  = dr::gather<Float64>(dist_data, linear_idx, active);
    Float64 left    = dr::gather<Float64>(dist_data, linear_idx - 1, active);
    Float64 right   = dr::gather<Float64>(dist_data, linear_idx + 1, active);
    Float64 up      = dr::gather<Float64>(dist_data, linear_idx - shape.x(), active);
    Float64 down    = dr::gather<Float64>(dist_data, linear_idx + shape.x(), active);
    Float64 front   = dr::gather<Float64>(dist_data, linear_idx - shape.y() * shape.x(), active);
    Float64 back    = dr::gather<Float64>(dist_data, linear_idx + shape.y() * shape.x(), active);

    Vector3d min_values(dr::minimum(left, right), dr::minimum(up, down), dr::minimum(front, back));
    Float64 eik = solve_eikonal(center, min_values, dx);

    // Update the current distance information
    dr::scatter(distance_target.array(), eik, linear_idx, active);
}

template <typename Float, typename UInt8 = dr::uint8_array_t<Float>,
          typename Int32 = dr::int32_array_t<Float>, typename Float64 = dr::float64_array_t<Float>>
void fast_sweep(dr::Tensor<Float64> &distance, const dr::Tensor<Int32> &frozen, size_t n_iter = 1) {

    // On the GPU, use the faster CUDA implementation if appropriate
    constexpr bool use_drjit = !dr::is_cuda_v<Float> || !USE_CUDA;

    using Bool     = dr::mask_t<Float>;
    using UInt32   = dr::uint32_array_t<Float>;
    using Vector3i = dr::Array<Int32, 3>;

    ScalarVector3i shape_in(distance.shape()[0], distance.shape()[1], distance.shape()[2]);
    shape_in                        = shape_in - 2 * BORDER_SIZE;
    size_t total_levels             = dr::sum(shape_in);
    ScalarVector3f dx               = 1 / ScalarVector3f(shape_in);
    ScalarVector3i sweep_offsets[8] = { { 0, 0, 0 },
                                        { 0, shape_in[1] + 1, 0 },
                                        { 0, 0, shape_in[2] + 1 },
                                        { shape_in[0] + 1, 0, 0 },
                                        { 0, 0, 0 },
                                        { 0, shape_in[1] + 1, 0 },
                                        { 0, 0, shape_in[2] + 1 },
                                        { shape_in[0] + 1, 0, 0 } };
    Vector3i shape_in_opaque(shape_in);
    dr::make_opaque(shape_in_opaque);

    // Write to a separate target array if we use CUDA (Dr.Jit handles that automatically)
    dr::Tensor<Float64> distance_target;
    if constexpr (!use_drjit) {
        distance_target = dr::Tensor<Float64>(distance.array(), 3, distance.shape().data());
        dr::eval(distance_target);
        dr::sync_thread();
    }

    for (size_t i = 0; i < n_iter; i++) {
        for (size_t sw_count = 1; sw_count < 9; ++sw_count) {
            int start, end, delta;
            if (sw_count == 2 || sw_count == 5 || sw_count == 7 || sw_count == 8) {
                start = total_levels;
                end   = 2;
                delta = -1;
            } else {
                start = 3;
                end   = total_levels + 1;
                delta = 1;
            }
            Vector3i sweep_offset(sweep_offsets[sw_count - 1]);
            dr::make_opaque(sweep_offset);

            for (int level = start; level != end; level += delta) {
                int xs = std::max(1, level - (shape_in.y() + shape_in.z()));
                int ys = std::max(1, level - (shape_in.x() + shape_in.z()));
                int xe = std::min(shape_in.x(), level - 2);
                int ye = std::min(shape_in.y(), level - 2);
                int xr = xe - xs + 1;
                int yr = ye - ys + 1;

                if constexpr (use_drjit) {
                    Int32 level_opaque = dr::opaque<Int32>(level);
                    // This seems to be slower
                    // auto [x, y]        = dr::meshgrid(dr::arange<Int32>(xr), dr::arange<Int32>(yr));
                    Int32 indices   = dr::arange<Int32>(xr * yr);
                    Int32 xr_opaque = dr::opaque<Int32>(xr);
                    Int32 x         = indices / xr_opaque;
                    Int32 y         = indices % xr_opaque;
                    x               = x + dr::opaque<Int32>(xs);
                    y               = y + dr::opaque<Int32>(ys);
                    sweep_step<Float>(x, y, distance, distance, frozen, shape_in_opaque,
                                      level_opaque, sweep_offset, dx);
                    dr::eval(distance);
                } else {
                    ScalarVector3i sweep_offset(sweep_offsets[sw_count - 1]);
                    Vec3i shape(distance.shape()[0], distance.shape()[1], distance.shape()[2]);
                    Vec3i offset(xs, ys, 0);
                    Vec3i shape_in_v(shape_in.x(), shape_in.y(), shape_in.z());
                    Vec3i sweep_offset_v(sweep_offset.x(), sweep_offset.y(), sweep_offset.z());
                    Vec3d dx_v(dx.x(), dx.y(), dx.z());
                    const double *distance_ptr = distance.array().data();
                    double *distance_target_ptr = distance_target.array().data();
                    const int *frozen_ptr = frozen.array().data();
                    void *args[9] = {&distance_ptr, &distance_target_ptr, &frozen_ptr,
                                     &level, &sweep_offset_v, &offset, &dx_v, &shape, &shape_in_v};
                    auto [grid_size, block_size] = get_launch_parameters(xr, yr);

                    CUcontext ctx = CUcontext(jit_cuda_context());
                    scoped_set_context guard(ctx);
                    cuda_check(cuLaunchKernel(fast_sweep_kernel,
                                              grid_size.x, grid_size.y, grid_size.z,
                                              block_size.x, block_size.y, block_size.z,
                                              0, 0, args, 0));
                    cuda_check(cuCtxSynchronize());
                    std::swap(distance, distance_target);
                }
            }
        }
    }
}

template <typename Float> dr::Tensor<Float> redistance(const dr::Tensor<Float> &init_distance) {
    using Float64           = dr::float64_array_t<Float>;
    using Int32             = dr::int32_array_t<Float>;

    if constexpr (dr::is_cuda_v<Float>)
        cuda_load_kernels();

    auto [distance, frozen] = initialize_distance(init_distance);
    dr::eval(distance, frozen);
    fast_sweep<Float>(distance, frozen);

    // Remove border passing and multiply by sign of the input
    auto [z, y, x] = meshgrid3d(dr::arange<Int32>(init_distance.shape()[0]) + BORDER_SIZE,
                                dr::arange<Int32>(init_distance.shape()[1]) + BORDER_SIZE,
                                dr::arange<Int32>(init_distance.shape()[2]) + BORDER_SIZE);
    Float result   = dr::gather<Float64>(
        distance.array(),
        z * distance.shape()[0] * distance.shape()[1] + y * distance.shape()[0] + x);
    return dr::Tensor<Float>(result * dr::sign(init_distance), 3, init_distance.shape().data());
}

template dr::Tensor<dr::CUDAArray<float>> redistance(const dr::Tensor<dr::CUDAArray<float>> &);
template dr::Tensor<dr::LLVMArray<float>> redistance(const dr::Tensor<dr::LLVMArray<float>> &);

#include "vec.h"

template <typename T> static T max(T a, T b) { return (a > b) ? a : b; }
template <typename T> static T min(T a, T b) { return (a < b) ? a : b; }

__device__ double solve_eikonal(double cur_dist, const Vec3d &min_vals, const Vec3d &dx) {
    double m[] = { min_vals.x, min_vals.y, min_vals.z };
    double d[] = { dx.x, dx.y, dx.z };
    // sort the min values
    for (int i = 1; i < 3; i++) {
        for (int j = 0; j < 3 - i; j++) {
            if (m[j] > m[j + 1]) {
                double tmp_m = m[j];
                double tmp_d = d[j];
                m[j]         = m[j + 1];
                d[j]         = d[j + 1];
                m[j + 1]     = tmp_m;
                d[j + 1]     = tmp_d;
            }
        }
    }
    double m2_0 = m[0] * m[0], m2_1 = m[1] * m[1], m2_2 = m[2] * m[2];
    double d2_0 = d[0] * d[0], d2_1 = d[1] * d[1], d2_2 = d[2] * d[2];
    double dist_new = m[0] + d[0];
    if (dist_new > m[1]) {
        double s = sqrt(-m2_0 + 2 * m[0] * m[1] - m2_1 + d2_0 + d2_1);
        dist_new = (m[1] * d2_0 + m[0] * d2_1 + d[0] * d[1] * s) / (d2_0 + d2_1);
        if (dist_new > m[2]) {
            double a = sqrt(-m2_0 * d2_1 - m2_0 * d2_2 + 2 * m[0] * m[1] * d2_2 - m2_1 * d2_0 -
                            m2_1 * d2_2 + 2 * m[0] * m[2] * d2_1 - m2_2 * d2_0 - m2_2 * d2_1 +
                            2 * m[1] * m[2] * d2_0 + d2_0 * d2_1 + d2_0 * d2_2 + d2_1 * d2_2);
            dist_new = (m[2] * d2_0 * d2_1 + m[1] * d2_0 * d2_2 + m[0] * d2_1 * d2_2 +
                        d[0] * d[1] * d[2] * a) /
                       (d2_0 * d2_1 + d2_0 * d2_2 + d2_1 * d2_2);
        }
    }
    return min(cur_dist, dist_new);
}

extern "C" __global__ void fast_sweep_kernel(const double *distance, double *distance_target,
                                             const int *frozen, int level, Vec3i sweep_offset,
                                             Vec3i offset, Vec3d dx, Vec3i shape, Vec3i shape_in) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + offset.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + offset.y;
    int z = level - x - y;
    if ((x <= shape_in.x) && (y <= shape_in.y) && (z > 0) && (z <= shape_in.z)) {
        int i          = abs(z - sweep_offset.z);
        int j          = abs(y - sweep_offset.y);
        int k          = abs(x - sweep_offset.x);
        int linear_idx = i * shape.y * shape.x + j * shape.x + k;
        if (frozen[linear_idx]) // Don't update frozen voxels
            return;

        double center = distance[linear_idx];
        double left   = distance[linear_idx - 1];
        double right  = distance[linear_idx + 1];
        double up     = distance[linear_idx - shape.x];
        double down   = distance[linear_idx + shape.x];
        double front  = distance[linear_idx - shape.y * shape.x];
        double back   = distance[linear_idx + shape.y * shape.x];
        Vec3d min_values(min(left, right), min(up, down), min(front, back));
        distance_target[linear_idx] = solve_eikonal(center, min_values, dx);
    }
}

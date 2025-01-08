#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define GRAVITATIONAL_FORCE 0.0001

__kernel void nbody_calculate_force_global(
    __global float * pxs, __global float * pys,
    __global float *vxs, __global float *vys,
    __global const float *mxs,
    __global float * dvx2d, __global float * dvy2d,
    int N,
    int t)
{
    unsigned int i = get_global_id(0);

    if (i >= N)
        return;

    __global float * dvx = dvx2d + t * N;
    __global float * dvy = dvy2d + t * N;

    float x0 = pxs[i];
    float y0 = pys[i];
    float m0 = mxs[i];

//    dvx[i] = 0;
//    dvy[i] = 0;

    for (int j = 0; j < N; j++) {
        if (i == j) {
            continue;
        }

        /*
        M1 - cur
        M2 - other

        F1 = M1 * M2 / ||p1 - p2||^2

        F1 = M * A = M1 * dV / dT

        dV = F1 * dT / M1 = M1 * M2 * dT / (M1 * ||p1 - p2||^2)

        dV = M2 * dT / ||p1 - p2||^2

        f1x = (x1 - x0) / fabs(y1-y0) * f1
        f1y = (y1 - y0) / fabs(x1-x0) * f1

        r2 = ||p1 - p2||^2

        dVy = (x1 - x0) / fabs(y1-y0) * M2 * dT / r2
        dVy = (y1 - y0) / fabs(x1-x0) * M2 * dT / r2


         (dx1^2 + dx2^2) / (dx1^2 + dx2^2) = 1

         dx1^2 = (dx1^2 + dx2^2) - dx2^2
        */

        float x1 = pxs[j];
        float y1 = pys[j];
        float m1 = mxs[j];

        float dx = x1 - x0;
        float dy = y1 - y0;

        float r2 = max(100.0f, pow(dx, 2) + pow(dy, 2));
        float r = sqrt(r2);

        float f = m1 / r2;

        dvx[i] += f * dx / r;
        dvy[i] += f * dy / r;
    }

    dvx[i] *= GRAVITATIONAL_FORCE;
    dvy[i] *= GRAVITATIONAL_FORCE;
}

__kernel void nbody_integrate(
        __global float * pxs, __global float * pys,
        __global float *vxs, __global float *vys,
        __global const float *mxs,
        __global float * dvx2d, __global float * dvy2d,
        int N,
        int t)
{
    unsigned int i = get_global_id(0);

    if (i >= N)
        return;

    __global float * dvx = dvx2d + t * N;
    __global float * dvy = dvy2d + t * N;

    vxs[i] += dvx[i];
    vys[i] += dvy[i];
    pxs[i] += vxs[i];
    pys[i] += vys[i];
}

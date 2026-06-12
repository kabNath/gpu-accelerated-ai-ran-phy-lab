#include <cuda_runtime.h>
extern "C" __global__
void complex_scale_kernel(const float2* x, float2* y, float scale, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        y[i].x = scale * x[i].x;
        y[i].y = scale * x[i].y;
    }
}

CUDA_KERNEL_SRC = r'''
extern "C" __global__
void complex_scale_kernel(const float2* x, float2* y, float scale, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        y[i].x = scale * x[i].x;
        y[i].y = scale * x[i].y;
    }
}
'''
def cupy_available() -> bool:
    try:
        import cupy
        return True
    except Exception:
        return False

def complex_scale_gpu(x, scale: float):
    try:
        import cupy as cp
    except ImportError as e:
        raise ImportError('CuPy is required for GPU kernel demo. Install cupy-cuda12x.') from e
    x_gpu=cp.asarray(x,dtype=cp.complex64); y_gpu=cp.empty_like(x_gpu)
    mod=cp.RawModule(code=CUDA_KERNEL_SRC,options=('--std=c++11',)); kernel=mod.get_function('complex_scale_kernel')
    n=x_gpu.size; block=256; grid=((n+block-1)//block,)
    kernel(grid,(block,),(x_gpu,y_gpu,cp.float32(scale),n))
    return y_gpu

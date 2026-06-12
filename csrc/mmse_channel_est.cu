// mmse_channel_est.cu — standalone CUDA C++ MMSE channel estimator.
//
// This is the compiled-CUDA companion to airan_phy_lab/cuda/mmse_kernels.py.
// It implements the SAME full-grid MMSE smoother used on the CPU:
//     h_mmse = R (R + sigma^2 I)^{-1} h_ls
// with R built from an exponential power-delay profile.
//
//   * build_corr_kernel / add_diag_kernel : hand-written CUDA kernels
//   * cuSolver Cpotrf/Cpotrs              : Hermitian Cholesky solve A^{-1} h_ls
//   * cuBLAS  Cgemm                       : R @ (A^{-1} h_ls)
//
// Modes:
//   --bench            generate data, time the hot path (potrs + gemm), print ms
//   --in F --out G     read h_ls (complex64, column-major n x B) from F,
//                      write h_mmse to G  (used by the CPU-vs-CUDA test)
//
// Build:  nvcc -O3 -std=c++14 -arch=sm_89 mmse_channel_est.cu -o mmse_est -lcublas -lcusolver
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuComplex.h>

#define CK(x)  do{cudaError_t e=(x); if(e){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
#define BK(x)  do{ if((x)!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cuBLAS %s:%d\n",__FILE__,__LINE__);exit(1);} }while(0)
#define SK(x)  do{ if((x)!=CUSOLVER_STATUS_SUCCESS){fprintf(stderr,"cuSolver %s:%d\n",__FILE__,__LINE__);exit(1);} }while(0)
using cf = cuFloatComplex;

__global__ void build_corr_kernel(cf* R, const float* pdp, int n, int n_taps) {
    int r = blockIdx.x*blockDim.x + threadIdx.x;
    int c = blockIdx.y*blockDim.y + threadIdx.y;
    if (r >= n || c >= n) return;
    float dk = (float)(r - c), re = 0.f, im = 0.f;
    const float TWO_PI = 6.283185307179586f;
    for (int l = 0; l < n_taps; ++l) {
        float th = -TWO_PI*dk*(float)l/(float)n, s, co; __sincosf(th, &s, &co);
        re += pdp[l]*co; im += pdp[l]*s;
    }
    R[r + c*n] = make_cuFloatComplex(re, im);   // column-major for cuSolver/cuBLAS
}

__global__ void add_diag_kernel(cf* A, int n, float v) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) { cf d = A[i+i*n]; A[i+i*n] = make_cuFloatComplex(cuCrealf(d)+v, cuCimagf(d)); }
}

int main(int argc, char** argv) {
    int n = 64, B = 4096, n_taps = 8, iters = 200;
    float nv = 0.1f, decay = 0.5f;
    std::string in_path, out_path; bool bench = false;
    for (int i = 1; i < argc; ++i) { std::string a = argv[i]; auto nx=[&]{return std::string(argv[++i]);};
        if(a=="--n")n=std::stoi(nx()); else if(a=="--B")B=std::stoi(nx());
        else if(a=="--taps")n_taps=std::stoi(nx()); else if(a=="--nv")nv=std::stof(nx());
        else if(a=="--decay")decay=std::stof(nx()); else if(a=="--iters")iters=std::stoi(nx());
        else if(a=="--bench")bench=true; else if(a=="--in")in_path=nx(); else if(a=="--out")out_path=nx(); }

    std::vector<float> pdp(n_taps); float s=0; for(int l=0;l<n_taps;++l){pdp[l]=std::exp(-decay*l);s+=pdp[l];} for(auto&p:pdp)p/=s;

    cublasHandle_t blas; BK(cublasCreate(&blas));
    cusolverDnHandle_t sol; SK(cusolverDnCreate(&sol));
    cudaStream_t st; CK(cudaStreamCreate(&st)); BK(cublasSetStream(blas,st)); SK(cusolverDnSetStream(sol,st));

    float* d_pdp; CK(cudaMalloc(&d_pdp,n_taps*sizeof(float)));
    CK(cudaMemcpy(d_pdp,pdp.data(),n_taps*sizeof(float),cudaMemcpyHostToDevice));
    cf *d_R,*d_A,*d_Hls,*d_Z,*d_Hmmse;
    CK(cudaMalloc(&d_R,(size_t)n*n*sizeof(cf)));  CK(cudaMalloc(&d_A,(size_t)n*n*sizeof(cf)));
    CK(cudaMalloc(&d_Hls,(size_t)n*B*sizeof(cf)));CK(cudaMalloc(&d_Z,(size_t)n*B*sizeof(cf)));
    CK(cudaMalloc(&d_Hmmse,(size_t)n*B*sizeof(cf)));

    dim3 tb(16,16), gb((n+15)/16,(n+15)/16);
    build_corr_kernel<<<gb,tb,0,st>>>(d_R,d_pdp,n,n_taps);
    CK(cudaMemcpyAsync(d_A,d_R,(size_t)n*n*sizeof(cf),cudaMemcpyDeviceToDevice,st));
    add_diag_kernel<<<(n+127)/128,128,0,st>>>(d_A,n,nv);

    int lwork=0; SK(cusolverDnCpotrf_bufferSize(sol,CUBLAS_FILL_MODE_LOWER,n,d_A,n,&lwork));
    cf* d_work; int* d_info; CK(cudaMalloc(&d_work,lwork*sizeof(cf))); CK(cudaMalloc(&d_info,sizeof(int)));
    SK(cusolverDnCpotrf(sol,CUBLAS_FILL_MODE_LOWER,n,d_A,n,d_work,lwork,d_info));

    std::vector<cf> h_Hls((size_t)n*B);
    if(!in_path.empty()){ FILE*f=std::fopen(in_path.c_str(),"rb"); if(!f){fprintf(stderr,"open %s\n",in_path.c_str());return 1;} std::fread(h_Hls.data(),sizeof(cf),(size_t)n*B,f); std::fclose(f); }
    else { std::mt19937 g(0); std::normal_distribution<float> nd(0,1); for(auto&z:h_Hls)z=make_cuFloatComplex(nd(g),nd(g)); }
    CK(cudaMemcpy(d_Hls,h_Hls.data(),(size_t)n*B*sizeof(cf),cudaMemcpyHostToDevice));

    cf one=make_cuFloatComplex(1,0), zero=make_cuFloatComplex(0,0);
    auto estimate=[&]{
        CK(cudaMemcpyAsync(d_Z,d_Hls,(size_t)n*B*sizeof(cf),cudaMemcpyDeviceToDevice,st));
        SK(cusolverDnCpotrs(sol,CUBLAS_FILL_MODE_LOWER,n,B,d_A,n,d_Z,n,d_info)); // Z = A^{-1} Hls
        BK(cublasCgemm(blas,CUBLAS_OP_N,CUBLAS_OP_N,n,B,n,&one,d_R,n,d_Z,n,&zero,d_Hmmse,n)); // Hmmse = R Z
    };

    if(bench){
        for(int w=0;w<5;++w)estimate(); CK(cudaStreamSynchronize(st));
        cudaEvent_t t0,t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
        CK(cudaEventRecord(t0,st)); for(int i=0;i<iters;++i)estimate(); CK(cudaEventRecord(t1,st));
        CK(cudaEventSynchronize(t1)); float ms=0; CK(cudaEventElapsedTime(&ms,t0,t1));
        printf("CUDA  n=%d B=%d : %.4f ms/batch (%.0f symbols/ms)\n",n,B,ms/iters,B/(ms/iters));
    } else { estimate(); CK(cudaStreamSynchronize(st)); }

    if(!out_path.empty()){ std::vector<cf> o((size_t)n*B);
        CK(cudaMemcpy(o.data(),d_Hmmse,(size_t)n*B*sizeof(cf),cudaMemcpyDeviceToHost));
        FILE*f=std::fopen(out_path.c_str(),"wb"); std::fwrite(o.data(),sizeof(cf),(size_t)n*B,f); std::fclose(f); }
    return 0;
}

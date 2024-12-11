#include "cudaCoreGemm.h"

#include <cuda_fp16.h>
float run_cuda_kernel(Params& params, int warmup, int iter)
{
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    for (int i = 0; i < warmup; ++i)
    {
        cudaCoreGemmLauncher<half, half>(params, s);
        //cudaCoreGemmDispatcher(params, s);
    }
    cudaEventRecord(begin, s);
    for (int i = 0; i < iter; ++i)
    {
        cudaCoreGemmLauncher<half, half>(params, s);
        //cudaCoreGemmDispatcher(params, s);
    }
    cudaEventRecord(end, s);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, begin, end);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaStreamDestroy(s);
    return time / iter;
}


int main(int argc, char* argv[]) {
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    int warmup = 20;
    int iter = 100;

    
    printf("fp16   bs : %d    IC = %d   OC = %d    ", m, k, n);

    half*    h_act       = (half   *)malloc(m * k * sizeof(half));
    half*    h_weight    = (half   *)malloc(k * n * sizeof(half));
    half*    h_out       = (half   *)malloc(m * n * sizeof(half));

    
    half*    d_act;
    half*    d_weight;
    half*    d_out;

    cudaMalloc(&d_act,        m * k             * sizeof(half));
    cudaMalloc(&d_weight,     k * n             * sizeof(half));
    cudaMalloc(&d_out,        m * n             * sizeof(half));

    for (int i = 0; i < m * k; ++i) {
        h_act[i] = 1;
    }
    for (int i = 0; i < k * n; ++i) {
        h_weight[i] = 1;
    }
    
    cudaMemcpy(d_act,        h_act,        m * k             * sizeof(half),     cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight,     h_weight,     k * n             * sizeof(half),     cudaMemcpyHostToDevice);


    Params params{
        d_act, d_weight, 1.0, d_out, m, n, k};
    float time = run_cuda_kernel(params, warmup, iter);
    double gflops = 2.0 * double(m * n * k) / double(1.0e9) / time;
    printf("time : %f   gflops : %f\n", time, gflops);
    cudaMemcpy(h_out,        d_out,        m * n             * sizeof(half),     cudaMemcpyDeviceToHost);

    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < 10; ++j) {
            printf("%.1lf ", __half2float(h_out[i*n+j]));
        }
        printf("\n");
    }
    
    return 0;
}
#include "int8SQ.h"

#include <cuda_fp16.h>
float run_cuda_kernel(Params& params, int warmup, int iter)
{
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    //int8_sq_launcher<half>(params, s);
    for (int i = 0; i < warmup; ++i)
    {
        int8_sq_launcher<half>(params, s);
        //cudaCoreGemmDispatcher(params, s);
    }
    cudaEventRecord(begin, s);
    for (int i = 0; i < iter; ++i)
    {
        int8_sq_launcher<half>(params, s);
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
    int m = 1; //atoi(argv[1]);
    int n = 4096; //atoi(argv[2]);
    int k = 11008; //atoi(argv[3]);
    int warmup = 20;
    int iter = 100;

    bool per_token = true;
    bool per_channel = true;
    
    int scale_token_size = per_token ? m * sizeof(float) : sizeof(float);
    int scale_channel_size = per_channel ? n * sizeof(float) : sizeof(float);

    printf("int8   bs : %d    IC = %d   OC = %d   per_token = %d   per_channel = %d  ", m, k, n, scale_token_size, scale_channel_size);
    
    int8_t*  h_act            = (int8_t *)malloc(m * k * sizeof(int8_t));
    float*   h_scale_tokens   = (float  *)malloc(scale_token_size);
    float*   h_scale_channels = (float  *)malloc(scale_channel_size);
    int8_t*  h_weight         = (int8_t *)malloc(k * n * sizeof(int8_t));
    half*    h_out            = (half   *)malloc(m * n * sizeof(half));

    int*     h_idx            = (int    *)malloc(4 * sizeof(int));
    int*     h_scale          = (int    *)malloc(4 * sizeof(int));

    
    int8_t*  d_act;
    float*   d_scale_tokens;
    float*   d_scale_channels;
    int8_t*  d_weight;
    half*    d_out;

    int*     d_idx;
    int*     d_scale; 

    cudaMalloc(&d_act,            m * k             * sizeof(int8_t));
    cudaMalloc(&d_scale_tokens,   scale_token_size);
    cudaMalloc(&d_scale_channels, scale_channel_size);
    cudaMalloc(&d_weight,         k * n             * sizeof(int8_t));
    cudaMalloc(&d_out,            m * n             * sizeof(half));

    cudaMalloc(&d_idx,            4                 * sizeof(int32_t));
    cudaMalloc(&d_scale,          4                 * sizeof(int32_t));
    
    for (int i = 0; i < m * k; ++i) {
        h_act[i] = 1;
    }
    for (int i = 0; i < k * n; ++i) {
        h_weight[i] = 1;
    }

    for (int i = 0; i < scale_token_size / 4; ++i) {
        h_scale_tokens[i] = 1;
    }
    for (int i = 0; i < scale_channel_size / 4; ++i) {
        h_scale_channels[i] = 1;
    }

    h_idx[0] = 1;
    h_idx[1] = 2;
    h_idx[2] = 3;
    h_idx[3] = 4;
    h_scale[0] = 2;
    h_scale[1] = 1;
    h_scale[2] = 1;
    h_scale[3] = 2;
    for (int i = 0; i < 4; ++i) {
        printf("idx scale : %d %d\n", h_idx[i], h_scale[i]);
    }

    cudaMemcpy(d_act,            h_act,            m * k * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_tokens,   h_scale_tokens,   scale_token_size,       cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_channels, h_scale_channels, scale_channel_size,     cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight,         h_weight,         k * n * sizeof(int8_t), cudaMemcpyHostToDevice);

    cudaMemcpy(d_idx,            h_idx,            4 * sizeof(int32_t),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale,          h_scale,          4 * sizeof(int32_t),        cudaMemcpyHostToDevice);


    Params params{
        d_act, d_weight, d_scale_tokens, d_scale_channels, d_out, d_idx, d_scale, m, n, k, per_channel, per_token};
    float time = run_cuda_kernel(params, warmup, iter);
    double gflops = 2.0 * double(m * n * k) / double(1.0e9) / time;
    printf("time : %f   gflops : %f\n", time, gflops);
    cudaMemcpy(h_out,        d_out,        m * n             * sizeof(half),     cudaMemcpyDeviceToHost);



    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < 100; ++j) {
            printf("%.1lf ", __half2float(h_out[i*n+j]));
            //printf("%d ", h_out[i*n+j]);
        }
        printf("\n");
    }
    
    return 0;
}
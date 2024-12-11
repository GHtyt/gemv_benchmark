#include "common.h"
//#include "converter.h"
//#include "details.h"
//#include "utility.h"
#include "kernel.h"
#include "kernelDispatcher.h"
#include "kernelLauncher.h"
#include <iostream>


float run_cuda_kernel(Params& params, int warmup, int iter)
{
    
    int32_t major{};
    int32_t minor{};
    int32_t deviceIndex{};
    cudaGetDevice(&deviceIndex);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, deviceIndex);
    int arch =  ((major << 8) | minor);

    //int arch = tensorrt_llm::common::getSMVersion();
    //simple_assert(is_supported(arch, params.type));
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    for (int i = 0; i < warmup; ++i)
    {
        kernel_launcher(arch, params, s);
    }
    cudaEventRecord(begin, s);
    for (int i = 0; i < iter; ++i)
    {
        kernel_launcher(arch, params, s);
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
    //void * a;
    //Params p(a, a, a, a, a, a, a, 0, 0, 0, 0, 0, KernelType::FP16Int4Groupwise);

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    int warmup = 20;
    int iter = 100;
    int groupsize = 0;
    printf("w8a16  bs : %d    IC = %d   OC = %d    ", m, k, n);

    auto KV = KernelType::FP16Int8PerChannel;
    using AType = half;
    using WType = uint8_t;
    static constexpr int ASizeInBits = sizeof(AType) * 8;
    static constexpr int WSizeInBits = 8;
    int gs_factor = groupsize == 0 ? 1 : groupsize;

    half*    h_act       = (half   *)malloc(m * k * sizeof(half));
    half*    h_act_scale = (half   *)malloc(k * sizeof(half));
    uint8_t* h_weight    = (uint8_t*)malloc(k * n * sizeof(uint8_t));
    half*    h_scales    = (half   *)malloc(n * k / gs_factor * sizeof(half));
    half*    h_zeros     = (half   *)malloc(n * k / gs_factor * sizeof(half));
    half*    h_bias      = (half   *)malloc(n * sizeof(half));
    half*    h_out       = (half   *)malloc(m * n * sizeof(half));

    half*    d_act;
    half*    d_act_scale;
    uint8_t* d_weight;
    half*    d_scales;
    half*    d_zeros;
    half*    d_bias;
    half*    d_out;

    
    cudaMalloc(&d_act,        m * k             * sizeof(half));
    cudaMalloc(&d_act_scale,  k                 * sizeof(half));
    cudaMalloc(&d_weight,     k * n             * sizeof(uint8_t));
    cudaMalloc(&d_scales,     n * k / gs_factor * sizeof(half));
    cudaMalloc(&d_zeros,      n * k / gs_factor * sizeof(half));
    cudaMalloc(&d_bias,       n                 * sizeof(half));
    cudaMalloc(&d_out,        m * n             * sizeof(half));

    for (int i = 0; i < m * k; ++i) {
        h_act[i] = 1;
    }
    for (int i = 0; i < k; ++i) {
        h_act_scale[i] = 1;
    }
    for (int i = 0; i < k * n; ++i) {
        h_weight[i] = 1;
    }
    for (int i = 0; i < k * n / gs_factor; ++i) {
        h_scales[i] = 1;
    }
    for (int i = 0; i < k * n / gs_factor; ++i) {
        h_zeros[i] = 1;
    }
    for (int i = 0; i < n; ++i) {
        h_bias[i] = 1;
    }

    
    cudaMemcpy(d_act,        h_act,        m * k             * sizeof(half),     cudaMemcpyHostToDevice);
    cudaMemcpy(d_act_scale,  h_act_scale,  k                 * sizeof(half),     cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight,     h_weight,     k * n             * sizeof(uint8_t),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales,     h_scales,     k * n / gs_factor * sizeof(half),     cudaMemcpyHostToDevice);
    cudaMemcpy(d_zeros,      h_zeros,      k * n / gs_factor * sizeof(half),     cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias,       h_bias,       n                 * sizeof(half),     cudaMemcpyHostToDevice);


    void* p_act_scale = nullptr;
    void* p_zeros = nullptr;
    void* p_bias = nullptr;

    if (groupsize != 0)
    {
        p_zeros = d_zeros;
        p_bias = d_bias;
        p_act_scale = d_act_scale;
    }

    Params params(d_act, p_act_scale, d_weight, d_scales, p_zeros, p_bias, d_out, 1.f,
        m, n, k, groupsize, KV);
    float time = run_cuda_kernel(params, warmup, iter);
    double gflops = 2.0 * double(m * n * k) / double(1.0e9) / time;
    printf("time : %f   gflops : %f\n", time, gflops);
    cudaMemcpy(h_out,        d_out,        m * n             * sizeof(half),     cudaMemcpyDeviceToHost);
    /*
    CudaBuffer d_act(m * k * ASizeInBits / 8);
    CudaBuffer d_act_scale(k * ASizeInBits / 8);
    CudaBuffer d_weight(k * n * WSizeInBits / 8);
    CudaBuffer d_scales(n * k / gs_factor * ASizeInBits / 8);
    CudaBuffer d_zeros(n * k / gs_factor * ASizeInBits / 8);
    CudaBuffer d_bias(n * ASizeInBits / 8);
    CudaBuffer d_out(m * n * ASizeInBits / 8);
    std::vector<AType> h_act(m * k), h_act_scale(k);
    std::vector<uint8_t> h_weight(k * n);
    std::vector<AType> h_scales(n * k), h_zeros(n * k), h_bias(n);
    std::vector<AType> h_out1(m * n), h_out2(m * n);

    random_fill(h_act, -1.f, 1.f);
    random_fill(h_act_scale, -1.f, 1.f);
    random_fill(h_scales, -1.f, 1.f);
    random_fill(h_zeros, -1.f, 1.f);
    random_fill(h_bias, -1.f, 1.f);

    for (uint8_t& v : h_weight)
    {
        v = rand() % 256;
    }

    d_act.copy_from(h_act.data());
    d_act_scale.copy_from(h_act_scale.data());
    d_weight.copy_from(h_weight.data());
    d_scales.copy_from(h_scales.data());
    d_zeros.copy_from(h_zeros.data());
    d_bias.copy_from(h_bias.data());

    void* p_act_scale = nullptr;
    void* p_zeros = nullptr;
    void* p_bias = nullptr;

    if (groupsize != 0)
    {
        p_zeros = d_zeros.data();
        p_bias = d_bias.data();
        p_act_scale = d_act_scale.data();
    }
    wo::Params params(d_act.data(), p_act_scale, d_weight.data(), d_scales.data(), p_zeros, p_bias, d_out.data(), 1.f,
        m, n, k, groupsize, KT);
    float time1, time2;
    time1 = run_cuda_kernel(params, warmup, iter);
    d_out.copy_to(h_out1.data());
    time2 = run_cutlass_kernel<KT>(params, warmup, iter);
    d_out.copy_to(h_out2.data());
    float quant_scale = 1.f / (1 << (WSizeInBits - 1));
    bool pass = compare<AType>(h_out1.data(), h_out2.data(), m * n, quant_scale);
    printf(
        "cuda kernel cost time %.6f, cutlass kernel cost time %.6f, cuda speedup %.3f\n", time1, time2, time2 / time1);
    return pass;
    */
    return 0;
}
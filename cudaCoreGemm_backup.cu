/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cutlass/numeric_conversion.h"
#include "cudaCoreGemm.h"
#include <cub/cub.cuh>

template <typename InputType, typename OutputType, int TILE_M, int TILE_N, int BLOCK_SIZE>
__global__ void cudaCoreGemm(InputType const* __restrict__ act, InputType const* __restrict__ weight, float alpha,
    OutputType* __restrict__ output, int m, int n, int k)
{
    using VecType = int4;
    static constexpr int kStepK = static_cast<int>(128 / (8 * sizeof(InputType)));
    static constexpr int kTileK = kStepK * BLOCK_SIZE;
    auto tileIdM = static_cast<int>(blockIdx.x * TILE_M);
    auto tileIdN = static_cast<int>(blockIdx.y * TILE_N);
    auto tid = static_cast<int>(threadIdx.x);
    float tile_a[kStepK], tile_w[TILE_N * kStepK];
    float acc[TILE_M * TILE_N];

    static_assert(kStepK % 4 == 0);
    using CvtInputType = half;
    using Converter = cutlass::NumericArrayConverter<float, CvtInputType, 4>;
    using CvtSrcType = typename Converter::source_type;
    using CvtResType = typename Converter::result_type;
    static constexpr int kCvtCount = static_cast<int>(sizeof(VecType) / sizeof(CvtSrcType));

#pragma unroll
    for (int i = 0; i < TILE_M * TILE_N; ++i)
    {
        acc[i] = 0;
    }
    act += tileIdM * k;
    weight += tileIdN * k;
    output += tileIdM * n + tileIdN;
    for (int idxK = tid * kStepK; idxK < k; idxK += kTileK)
    {
        for (int i = 0; i < TILE_N; ++i)
        {
            auto tile_w_quantized = reinterpret_cast<VecType const*>(weight + i * k + idxK)[0];
#pragma unroll
            for (int cvtIdx = 0; cvtIdx < kCvtCount; ++cvtIdx)
            {
                reinterpret_cast<CvtResType*>(tile_w)[i * kCvtCount + cvtIdx]
                    = Converter::convert(reinterpret_cast<CvtSrcType*>(&tile_w_quantized)[cvtIdx]);
            }
        }
#pragma unroll
        for (int i = 0; i < TILE_M; ++i)
        {
            auto tile_a_quantized = reinterpret_cast<VecType const*>(act + i * k + idxK)[0];
#pragma unroll
            for (int cvtIdx = 0; cvtIdx < kCvtCount; ++cvtIdx)
            {
                reinterpret_cast<CvtResType*>(tile_a)[cvtIdx]
                    = Converter::convert(reinterpret_cast<CvtSrcType*>(&tile_a_quantized)[cvtIdx]);
            }
#pragma unroll
            for (int j = 0; j < TILE_N; ++j)
            {
#pragma unroll
                for (int l = 0; l < kStepK; ++l)
                {
                    acc[i * TILE_N + j] = fma(tile_a[l], tile_w[j * kStepK + l], acc[i * TILE_N + j]);
                }
            }
        }
    }

    typedef cub::WarpReduce<float> WarpReduce;

    static constexpr int kWarpSize = 32;
    static constexpr int kWarpNum = BLOCK_SIZE / kWarpSize;
    int warpId = tid / kWarpSize, laneId = tid % kWarpSize;
    __shared__ float shmem[TILE_M * TILE_N * kWarpNum];
    __shared__ typename WarpReduce::TempStorage tempStorage[kWarpNum];
#pragma unroll
    for (int mi = 0; mi < TILE_M; ++mi)
    {
#pragma unroll
        for (int ni = 0; ni < TILE_N; ++ni)
        {
            float val = WarpReduce(tempStorage[warpId]).Sum(acc[mi * TILE_N + ni]);
            if (laneId == 0)
            {
                shmem[mi * TILE_N + ni + warpId * TILE_M * TILE_N] = val;
            }
        }
    }
    __syncthreads();
    for (int ii = tid; ii < TILE_M * TILE_N; ii += BLOCK_SIZE)
    {
        int mid = ii / TILE_N, nid = ii % TILE_N;
        float val = 0;
#pragma unroll
        for (int jj = 0; jj < kWarpNum; ++jj)
        {
            val += shmem[jj * TILE_M * TILE_N + ii];
        }
        output[mid * n + nid] = static_cast<OutputType>(val * alpha);
    }
}

template <typename InputType, typename OutputType, int TILE_M, int TILE_N, int BLOCK_SIZE>
void cudaCoreGemmKernel(Params const& params, cudaStream_t stream)
{
    dim3 block(BLOCK_SIZE);
    dim3 grid(params.m / TILE_M, params.n / TILE_N);
    cudaCoreGemm<InputType, OutputType, TILE_M, TILE_N, BLOCK_SIZE><<<grid, block, 0, stream>>>(
        reinterpret_cast<InputType const*>(params.act), reinterpret_cast<InputType const*>(params.weight), params.alpha,
        reinterpret_cast<OutputType*>(params.output), params.m, params.n, params.k);
}

template <typename InputType, typename OutputType, int TILE_M, int TILE_N, int BLOCK_SIZE>
bool cudaCoreGemmTemplateCaller(Params const& params, cudaStream_t stream)
{
    constexpr int cudaCoreGemmTemplateMaxM = 16;
    if (params.m == TILE_M)
    {
        cudaCoreGemmKernel<InputType, OutputType, TILE_M, TILE_N, BLOCK_SIZE>(params, stream);
        return true;
    }
    if constexpr (TILE_M < cudaCoreGemmTemplateMaxM)
    {
        return cudaCoreGemmTemplateCaller<InputType, OutputType, TILE_M + 1, TILE_N, BLOCK_SIZE>(params, stream);
    }
    return false;
}

template <typename InputType, typename OutputType>
bool cudaCoreGemmLauncher(Params const& params, cudaStream_t stream)
{
    return cudaCoreGemmTemplateCaller<InputType, OutputType, 1, 2, 128>(params, stream);
}
/*
bool cudaCoreGemmDispatcher(Params const& params, cudaStream_t stream)
{
    bool dispatched = true;
    if (params.n % 2 != 0)
    {
        dispatched = false;
    }
    else if (params.inputType == nvinfer1::DataType::kFP8)
    {
        if (params.k % 16 != 0)
        {
            // Expect k % 16 == 0 for 128 bits alignment
            dispatched = false;
        }
        else if (params.outputType == nvinfer1::DataType::kHALF)
        {
            dispatched = cudaCoreGemmLauncher<__nv_fp8_e4m3, half>(params, stream);
        }
        else if (params.outputType == nvinfer1::DataType::kBF16)
        {
            dispatched = cudaCoreGemmLauncher<__nv_fp8_e4m3, __nv_bfloat16>(params, stream);
        }
        else if (params.outputType == nvinfer1::DataType::kFLOAT)
        {
            dispatched = cudaCoreGemmLauncher<__nv_fp8_e4m3, float>(params, stream);
        }
        else
        {
            dispatched = false;
        }
    }
    else if (params.inputType == nvinfer1::DataType::kHALF)
    {
        if (params.k % 8 != 0)
        {
            // Expect k % 8 == 0 for 128 bits alignment
            dispatched = false;
        }
        else if (params.outputType == nvinfer1::DataType::kHALF)
        {
            dispatched = cudaCoreGemmLauncher<half, half>(params, stream);
        }
        else if (params.outputType == nvinfer1::DataType::kFLOAT)
        {
            dispatched = cudaCoreGemmLauncher<half, float>(params, stream);
        }
        else
        {
            dispatched = false;
        }
    }
    else if (params.inputType == nvinfer1::DataType::kBF16)
    {
        if (params.k % 8 != 0)
        {
            // Expect k % 8 == 0 for 128 bits alignment
            dispatched = false;
        }
        else if (params.outputType == nvinfer1::DataType::kBF16)
        {
            dispatched = cudaCoreGemmLauncher<__nv_bfloat16, __nv_bfloat16>(params, stream);
        }
        else if (params.outputType == nvinfer1::DataType::kFLOAT)
        {
            dispatched = cudaCoreGemmLauncher<__nv_bfloat16, float>(params, stream);
        }
        else
        {
            dispatched = false;
        }
    }
    else if (params.inputType == nvinfer1::DataType::kFLOAT)
    {
        if (params.k % 4 != 0)
        {
            // Expect k % 4 == 0 for 128 bits alignment
            dispatched = false;
        }
        else if (params.outputType == nvinfer1::DataType::kFLOAT)
        {
            dispatched = cudaCoreGemmLauncher<float, float>(params, stream);
        }
        else
        {
            dispatched = false;
        }
    }
    else
    {
        dispatched = false;
    }

    if (!dispatched)
    {
        TLLM_LOG_DEBUG(
            "tensorrt_llm::kernels::cuda_core_gemm::cudaCoreGemmDispatcher failed to dispatch: inputType=%d, "
            "outputType=%d, "
            "m=%d, "
            "n=%d, k=%d",
            params.inputType, params.outputType, params.m, params.n, params.k);
    }
    return dispatched;
}
*/
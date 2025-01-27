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

#pragma once
#include "common.h"
#include "kernel.h"


// TODO:
// Using a mechanism similar to the gemm config profiler, dynamically search for the optimal configuration during the
// build engine process.
template <typename Details, int GroupSize, bool EnableActScale, bool EnableZero, bool EnableBias,
    bool ApplyAlphaInAdvance>
void dispatcher(Params& params, cudaStream_t s)
{
#define DISPATCHER_FOR_M(target_m, CtaM, CtaN, Threads)                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        if (params.m == target_m)                                                                                      \
        {                                                                                                              \
            exec_kernel<Details, CtaM, CtaN, Threads, GroupSize, EnableActScale, EnableZero, EnableBias,               \
                ApplyAlphaInAdvance>(params, s);                                                                       \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0);
    if constexpr (EnableZero)
    {
        // clang-format off
        DISPATCHER_FOR_M(1, 1, 4, 128);
        DISPATCHER_FOR_M(2, 2, 4, 128);
        DISPATCHER_FOR_M(3, 3, 4, 128);
        DISPATCHER_FOR_M(4, 4, 4, 128);
        // clang-format on
    }
    else
    {
        // clang-format off
        DISPATCHER_FOR_M(1, 1, 8, 128);
        DISPATCHER_FOR_M(2, 2, 8, 128);
        DISPATCHER_FOR_M(3, 3, 8, 128);
        DISPATCHER_FOR_M(4, 4, 8, 128);
        // clang-format on
    }
    throw std::runtime_error("unsupported m");
#undef DISPATCHER_FOR_M
}

template <typename Details, int GroupSize, bool EnableActScale, bool EnableZero, bool EnableBias>
void check_alpha(Params& params, cudaStream_t s)
{
    if (params.apply_alpha_in_advance && params.alpha != 1.f)
    {
        dispatcher<Details, GroupSize, EnableActScale, EnableZero, EnableBias, true>(params, s);
    }
    else
    {
        dispatcher<Details, GroupSize, EnableActScale, EnableZero, EnableBias, false>(params, s);
    }
}

template <typename Details, int GroupSize>
void check_pointer(Params& params, cudaStream_t s)
{
    if constexpr (GroupSize == 0)
    {
        check_alpha<Details, GroupSize, false, false, false>(params, s);
    }
    else
    {
        if (params.act_scale && params.zeros && params.bias)
        {
            check_alpha<Details, GroupSize, true, true, true>(params, s);
        }
        else if (params.act_scale && params.zeros && !params.bias)
        {
            check_alpha<Details, GroupSize, true, true, false>(params, s);
        }
        else if (params.act_scale && !params.zeros && params.bias)
        {
            check_alpha<Details, GroupSize, true, false, true>(params, s);
        }
        else if (!params.act_scale && params.zeros && params.bias)
        {
            check_alpha<Details, GroupSize, false, true, true>(params, s);
        }
        else if (!params.act_scale && !params.zeros && params.bias)
        {
            check_alpha<Details, GroupSize, false, false, true>(params, s);
        }
        else if (params.act_scale && !params.zeros && !params.bias)
        {
            check_alpha<Details, GroupSize, true, false, false>(params, s);
        }
        else if (!params.act_scale && params.zeros && !params.bias)
        {
            check_alpha<Details, GroupSize, false, true, false>(params, s);
        }
        else
        {
            check_alpha<Details, GroupSize, false, false, false>(params, s);
        }
    }
}

template <bool isGroupwise, typename Details>
void select_gs(Params& params, cudaStream_t s)
{
    if constexpr (isGroupwise)
    {
        if (params.groupsize == 64)
        {
            check_pointer<Details, 64>(params, s);
        }
        else if (params.groupsize == 128)
        {
            check_pointer<Details, 128>(params, s);
        }
    }
    else
    {
        if (params.groupsize == 0)
        {
            check_pointer<Details, 0>(params, s);
        }
    }
}

#define INSTANTIATE_WEIGHT_ONLY_CUDA_DISPATCHERS(KType, A, B, Layout, ConverterInterleave, KTile)                      \
    template void select_gs<kernel_type_traits<KType>::isGroupwise,                                                    \
        KernelDetails<A, B, Layout, ConverterInterleave, KTile>>(Params & params, cudaStream_t s);


INSTANTIATE_WEIGHT_ONLY_CUDA_DISPATCHERS(
    KernelType::BF16Int4Groupwise, BF16DetailsA, Int4DetailsW, ColumnMajor, false, 64);
INSTANTIATE_WEIGHT_ONLY_CUDA_DISPATCHERS(
    KernelType::BF16Int8PerChannel, BF16DetailsA, Int8DetailsW, ColumnMajor, false, 64);
INSTANTIATE_WEIGHT_ONLY_CUDA_DISPATCHERS(
    KernelType::FP16Int4Groupwise, FP16DetailsA, Int4DetailsW, ColumnMajor, false, 64);
INSTANTIATE_WEIGHT_ONLY_CUDA_DISPATCHERS(
    KernelType::FP16Int8PerChannel, FP16DetailsA, Int8DetailsW, ColumnMajor, false, 64);

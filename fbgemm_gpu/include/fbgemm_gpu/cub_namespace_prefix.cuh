/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef FBGEMM_CUB_USE_NAMESPACE

#undef CUB_NS_PREFIX
#undef CUB_NS_POSTFIX

#include <cuda.h> // for CUDA_VERSION
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
#include <cub/version.cuh>
#else
#define CUB_VERSION 0
#error "Error"
#endif

// PR https://github.com/NVIDIA/cub/pull/350 introduced breaking change.
// When the CUB_NS_[PRE|POST]FIX macros are set,
// CUB_NS_QUALIFIER must also be defined to the fully qualified CUB namespace
#if CUB_VERSION >= 101400
#undef CUB_NS_QUALIFIER
#else
#error "Error2"
#endif

#define CUB_NS_PREFIX namespace fbgemm_gpu {
#define CUB_NS_POSTFIX } // namespace fbgemm_gpu

// PR https://github.com/NVIDIA/cub/pull/350 introduced breaking change.
// When the CUB_NS_[PRE|POST]FIX macros are set,
// CUB_NS_QUALIFIER must also be defined to the fully qualified CUB namespace
#if CUB_VERSION >= 101400
#define CUB_NS_QUALIFIER ::fbgemm_gpu::cub
#endif

#endif

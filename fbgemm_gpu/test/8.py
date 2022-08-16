#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import random
import unittest
from itertools import accumulate
from typing import Any, Callable, List, Optional, Tuple, Type, Union

import hypothesis.strategies as st
import numpy as np
import torch
from hypothesis import given, settings, Verbosity

try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401

    # pyre-ignore[21]
    from test_utils import gpu_available, gpu_unavailable
except Exception:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
    from fbgemm_gpu.test.test_utils import gpu_available, gpu_unavailable

class SparseOpsTest(unittest.TestCase):

    # pyre-ignore [56]
    @given(
        N=st.integers(1, 32),
        shape=st.lists(st.integers(1, 32), min_size=1, max_size=2),
        dtype=st.sampled_from([torch.float, torch.half, torch.double]),
        use_cpu=st.booleans() if gpu_available else st.just(True),
        consecutive_indices=st.booleans(),
    )
    @settings(max_examples=20, deadline=None)
    def test_index_select_dim0(
        self,
        N: int,
        shape: List[int],
        dtype: torch.dtype,
        use_cpu: bool,
        consecutive_indices: bool,
    ) -> None:
        device = torch.device("cpu" if use_cpu else "cuda")
        U = random.randint(0, N + 1)
        if consecutive_indices:
            start = np.random.randint(0, U)
            length = np.random.randint(1, U - start + 1)
            indices = list(range(start, start + length))
            np_arr = np.array(indices)
            for _ in range(N - U):
                indices.append(np.random.randint(start, start + length))
                np_arr = np.array(indices)
                np.random.shuffle(np_arr)
            indices = torch.from_numpy(np_arr).to(torch.int).to(device)
            kwargs = {
                "consecutive_range_start": start,
                "consecutive_range_length": length,
            }
        else:
            indices = torch.randint(U, (N,), device=device)
            kwargs = {}
        print("indices = {}".format(sys.getsizeof(indices.storage())))
        input = torch.rand((U,) + tuple(shape), dtype=dtype, device=device)
        print("input = {}".format(sys.getsizeof(input.storage())))

        output_ref = torch.ops.fbgemm.index_select_dim0(input, indices, **kwargs)
        print("output_ref = {}".format(sys.getsizeof(output_ref.storage())))
        output = torch.index_select(input, 0, indices)
        print("output = {}".format(sys.getsizeof(output.storage())))

        torch.testing.assert_close(output, output_ref)

        gradcheck_args = [input.clone().detach().double().requires_grad_(True), indices]
        print("gradcheck_args[0], [1] = {}, {}".format(sys.getsizeof(gradcheck_args[0].storage()), sys.getsizeof(gradcheck_args[1].storage())))
        for k in kwargs:
            gradcheck_args.append(kwargs[k])

        torch.autograd.gradcheck(torch.ops.fbgemm.index_select_dim0, gradcheck_args)


if __name__ == "__main__":

    unittest.main()

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


def unbucketize_indices_value(
    bucketized_indices: torch.Tensor,
    bucketized_lengths: torch.Tensor,
    block_sizes: torch.Tensor,
    W: int,
    B: int,
) -> torch.Tensor:
    block_size_expand = torch.empty_like(bucketized_indices)
    bucket_expand = torch.empty_like(bucketized_indices)
    T = block_sizes.size()[0]
    offset = 0
    for w in range(W):
        for t in range(T):
            for b in range(B):
                seg_length = bucketized_lengths[w * T * B + t * B + b]
                for i in range(offset, offset + seg_length):
                    block_size_expand[i] = block_sizes[t]
                    bucket_expand[i] = w
                offset += seg_length
    return bucket_expand * block_size_expand + bucketized_indices


def get_n_rand_num_summing_to_k(n: int, k: int) -> np.ndarray:
    """Get a list of `n` integers which collectively sum to `k`, drawn
    uniformly from the set of all such lists.

    Args:
        n - The number of integers in the result list
        k - The value they should sum to
    """
    # There are a lot of ways to do this wrong, probably including
    # the ones you've just thought of. I think the following does
    # it correctly, though.
    if n == 0:
        return np.array([])
    return np.random.multinomial(k, np.ones(n) / n, size=1)[0]


class SparseOpsTest(unittest.TestCase):
    @staticmethod
    def permute_indices_ref_(
        lengths: torch.Tensor,
        indices: torch.Tensor,
        weights: Optional[torch.Tensor],
        permute: torch.LongTensor,
        is_1D: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        T = lengths.size(0)
        if is_1D:
            permuted_lengths = torch.index_select(lengths.view(-1), 0, permute).view(-1)
            original_segment_lengths = lengths.view(-1)
            original_segment_start = [0] + list(accumulate(lengths.view(-1)))

            permuted_indices = []
            permuted_weights = []
            for i in range(permute.numel()):
                start = original_segment_start[permute[i]]
                end = start + original_segment_lengths[permute[i]]
                permuted_indices.append(indices[start:end])
                if weights is not None:
                    permuted_weights.append(weights[start:end])

            permuted_indices = torch.cat(permuted_indices, dim=0).flatten()

            if weights is None:
                permuted_weights = None
            else:
                permuted_weights = torch.cat(permuted_weights, dim=0).flatten()
        else:
            permuted_lengths = torch.index_select(lengths.view(T, -1), 0, permute)
            original_segment_lengths = lengths.view(T, -1).sum(dim=1, dtype=torch.int32)
            original_segment_start = [0] + list(
                accumulate(original_segment_lengths.view(-1))
            )

            permuted_indices = []
            permuted_weights = []
            for i in range(permute.size(0)):
                start = original_segment_start[permute[i]]
                end = start + original_segment_lengths[permute[i]]
                permuted_indices.append(indices[start:end])
                if weights is not None:
                    permuted_weights.append(weights[start:end])

            permuted_indices = torch.cat(permuted_indices, dim=0).flatten()

            if weights is None:
                permuted_weights = None
            else:
                permuted_weights = torch.cat(permuted_weights, dim=0).flatten()

        return permuted_lengths, permuted_indices, permuted_weights

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        n=st.integers(2, 10),
        k=st.integers(2, 10),
        batch_size=st.integers(1, 30),
        divisions=st.integers(1, 10),
        max_length=st.integers(1, 20),
    )
    @settings(deadline=None)
    def test_pack_segments_smaller_max_len(
        self,
        n: int,
        k: int,
        batch_size: int,
        divisions: int,
        max_length: int,
    ) -> None:
        input_data = torch.tensor(np.random.rand(batch_size, n, k), dtype=torch.float32)
        lengths = torch.tensor(
            get_n_rand_num_summing_to_k(divisions, batch_size), dtype=torch.int
        )

        packed_tensor = torch.ops.fbgemm.pack_segments(
            t_in=input_data,
            lengths=lengths,
            max_length=max_length,
        )
        self.assertEqual(packed_tensor.shape, (divisions, max_length, n, k))

        packed_ref = self._pack_segments_ref(
            lengths,
            input_data,
            max_length=max_length,
        )
        # pyre-fixme[6]: For 2nd param expected `Tensor` but got `ndarray`.
        self.assertTrue(torch.equal(packed_tensor, packed_ref))

        if gpu_available:
            packed_cuda = torch.ops.fbgemm.pack_segments(
                t_in=input_data.cuda(),
                lengths=lengths.cuda(),
                max_length=max_length,
            )
            self.assertTrue(torch.equal(packed_tensor, packed_cuda.cpu()))

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
        input = torch.rand((U,) + tuple(shape), dtype=dtype, device=device)

        output_ref = torch.ops.fbgemm.index_select_dim0(input, indices, **kwargs)
        output = torch.index_select(input, 0, indices)

        torch.testing.assert_close(output, output_ref)

        gradcheck_args = [input.clone().detach().double().requires_grad_(True), indices]
        for k in kwargs:
            gradcheck_args.append(kwargs[k])

        torch.autograd.gradcheck(torch.ops.fbgemm.index_select_dim0, gradcheck_args)

    # pyre-ignore [56]
    @given(
        T=st.integers(1, 5),
        B=st.integers(1, 5),
        L=st.integers(1, 5),
    )
    @settings(max_examples=20, deadline=None)
    def test_bottom_unique_k_per_row(
        self,
        T: int,
        B: int,
        L: int,
    ) -> None:
        E = 1000000
        all_indices = (np.random.zipf(a=1.15, size=(T, B, 3 * L)) - 1) % E
        all_indices_deduped = torch.ops.fbgemm.bottom_unique_k_per_row(
            torch.as_tensor(all_indices), L
        )
        for index_tuple in itertools.product(range(T), range(B)):
            # sample without replacement from
            # https://stats.stackexchange.com/questions/20590/how-do-i-sample-without-replacement-using-a-sampling-with-replacement-function
            r = set()
            for x in all_indices[index_tuple]:
                if x not in r:
                    r.add(x)
                    if len(r) == L:
                        break
            assert (len(r)) == L, "too skewed distribution (alpha too big)"
            all_indices[index_tuple][:L] = sorted(r)
        all_indices_deduped_ref = torch.as_tensor(all_indices[:, :, :L])
        torch.testing.assert_close(all_indices_deduped, all_indices_deduped_ref)


if __name__ == "__main__":

    unittest.main()

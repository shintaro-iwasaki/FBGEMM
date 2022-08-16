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

    def _pack_segments_ref(
        self,
        lengths: torch.Tensor,
        tensor: torch.Tensor,
        max_length: Optional[int] = None,
    ) -> np.ndarray:
        lengths = lengths.numpy()
        sections = np.split(tensor, np.cumsum(lengths))
        max_length = np.max(lengths, initial=0) if max_length is None else max_length
        padded_arrs = []
        for arr in sections[:-1]:  # Last section is always a blank
            arr = arr[: min(max_length, len(arr)), ...]
            padded_arr = np.pad(
                arr,
                [(0, max(max_length - arr.shape[0], 0))]
                + ([(0, 0)] * (len(arr.shape) - 1)),
                constant_values=0,
            )
            padded_arrs.append(padded_arr)

        if len(padded_arrs) == 0:
            padded_arrs = torch.empty((0, 0) + tuple(tensor.shape[1:]))
        else:
            padded_arrs = torch.Tensor(np.stack(padded_arrs))

        # pyre-fixme[7]: Expected `ndarray` but got `Tensor`.
        return padded_arrs

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

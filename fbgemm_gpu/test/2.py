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

    # pyre-ignore [56]
    @given(
        N=st.integers(min_value=1, max_value=20),
        offsets_type=st.sampled_from([torch.int32, torch.int64]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_offsets_range(
        self,
        N: int,
        # pyre-fixme[11]: Annotation `int32` is not defined as a type.
        # pyre-fixme[11]: Annotation `int64` is not defined as a type.
        offsets_type: "Union[Type[torch.int32], Type[torch.int64]]",
    ) -> None:
        lengths = np.array([np.random.randint(low=0, high=20) for _ in range(N)])
        offsets = np.cumsum(np.concatenate(([0], lengths)))[:-1]
        range_ref = torch.from_numpy(
            np.concatenate([np.arange(size) for size in lengths])
        )
        output_size = np.sum(lengths)

        offsets_cpu = torch.tensor(offsets, dtype=offsets_type)
        range_cpu = torch.ops.fbgemm.offsets_range(offsets_cpu, output_size)
        range_ref = torch.tensor(range_ref, dtype=range_cpu.dtype)
        torch.testing.assert_close(range_cpu, range_ref, rtol=0, atol=0)

        if gpu_available:
            range_gpu = torch.ops.fbgemm.offsets_range(offsets_cpu.cuda(), output_size)
            range_ref = torch.tensor(range_ref, dtype=range_gpu.dtype)
            torch.testing.assert_close(range_gpu.cpu(), range_ref, rtol=0, atol=0)

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        data_type=st.sampled_from([torch.half, torch.float32]),
        segment_value_type=st.sampled_from([torch.int, torch.long]),
        segment_length_type=st.sampled_from([torch.int, torch.long]),
    )
    @settings(verbosity=Verbosity.verbose, deadline=None)
    def test_histogram_binning_calibration_by_feature(
        self,
        data_type: torch.dtype,
        segment_value_type: torch.dtype,
        segment_length_type: torch.dtype,
    ) -> None:
        num_bins = 5000
        num_segments = 42

        logit = torch.tensor([-0.0018, 0.0085, 0.0090, 0.0003, 0.0029]).type(data_type)

        segment_value = torch.tensor([40, 31, 32, 13, 31]).type(segment_value_type)
        lengths = torch.tensor([[1], [1], [1], [1], [1]]).type(segment_length_type)

        num_interval = num_bins * (num_segments + 1)
        bin_num_examples = torch.empty([num_interval], dtype=torch.float64).fill_(0.0)
        bin_num_positives = torch.empty([num_interval], dtype=torch.float64).fill_(0.0)

        (
            calibrated_prediction,
            bin_ids,
        ) = torch.ops.fbgemm.histogram_binning_calibration_by_feature(
            logit=logit,
            segment_value=segment_value,
            segment_lengths=lengths,
            num_segments=num_segments,
            bin_num_examples=bin_num_examples,
            bin_num_positives=bin_num_positives,
            num_bins=num_bins,
            positive_weight=0.4,
            lower_bound=0.0,
            upper_bound=1.0,
            bin_ctr_in_use_after=10000,
            bin_ctr_weight_value=0.9995,
        )

        expected_calibrated_prediction = torch.tensor(
            [0.2853, 0.2875, 0.2876, 0.2858, 0.2863]
        ).type(data_type)
        expected_bin_ids = torch.tensor(
            [206426, 161437, 166437, 71428, 161431], dtype=torch.long
        )

        torch.testing.assert_close(
            calibrated_prediction,
            expected_calibrated_prediction,
            rtol=1e-03,
            atol=1e-03,
        )

        self.assertTrue(
            torch.equal(
                bin_ids.long(),
                expected_bin_ids,
            )
        )

        if torch.cuda.is_available():
            (
                calibrated_prediction_gpu,
                bin_ids_gpu,
            ) = torch.ops.fbgemm.histogram_binning_calibration_by_feature(
                logit=logit.cuda(),
                segment_value=segment_value.cuda(),
                segment_lengths=lengths.cuda(),
                num_segments=num_segments,
                bin_num_examples=bin_num_examples.cuda(),
                bin_num_positives=bin_num_positives.cuda(),
                num_bins=num_bins,
                positive_weight=0.4,
                lower_bound=0.0,
                upper_bound=1.0,
                bin_ctr_in_use_after=10000,
                bin_ctr_weight_value=0.9995,
            )

            torch.testing.assert_close(
                calibrated_prediction_gpu,
                expected_calibrated_prediction.cuda(),
                rtol=1e-03,
                atol=1e-03,
            )

            self.assertTrue(
                torch.equal(
                    bin_ids_gpu.long(),
                    expected_bin_ids.cuda(),
                )
            )

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        data_type=st.sampled_from([torch.half, torch.float32]),
        segment_value_type=st.sampled_from([torch.int, torch.long]),
        segment_length_type=st.sampled_from([torch.int, torch.long]),
    )
    @settings(verbosity=Verbosity.verbose, deadline=None)
    def test_generic_histogram_binning_calibration_by_feature(
        self,
        data_type: torch.dtype,
        segment_value_type: torch.dtype,
        segment_length_type: torch.dtype,
    ) -> None:
        num_bins = 5000
        num_segments = 42

        logit = torch.tensor([-0.0018, 0.0085, 0.0090, 0.0003, 0.0029]).type(data_type)

        segment_value = torch.tensor([40, 31, 32, 13, 31]).type(segment_value_type)
        lengths = torch.tensor([[1], [1], [1], [1], [1]]).type(segment_length_type)

        num_interval = num_bins * (num_segments + 1)
        bin_num_examples = torch.empty([num_interval], dtype=torch.float64).fill_(0.0)
        bin_num_positives = torch.empty([num_interval], dtype=torch.float64).fill_(0.0)

        lower_bound = 0.0
        upper_bound = 1.0
        w = (upper_bound - lower_bound) / num_bins
        bin_boundaries = torch.arange(
            lower_bound + w, upper_bound - w / 2, w, dtype=torch.float64
        )

        (
            calibrated_prediction,
            bin_ids,
        ) = torch.ops.fbgemm.generic_histogram_binning_calibration_by_feature(
            logit=logit,
            segment_value=segment_value,
            segment_lengths=lengths,
            num_segments=num_segments,
            bin_num_examples=bin_num_examples,
            bin_num_positives=bin_num_positives,
            bin_boundaries=bin_boundaries,
            positive_weight=0.4,
            bin_ctr_in_use_after=10000,
            bin_ctr_weight_value=0.9995,
        )

        expected_calibrated_prediction = torch.tensor(
            [0.2853, 0.2875, 0.2876, 0.2858, 0.2863]
        ).type(data_type)
        expected_bin_ids = torch.tensor(
            [206426, 161437, 166437, 71428, 161431], dtype=torch.long
        )

        torch.testing.assert_close(
            calibrated_prediction,
            expected_calibrated_prediction,
            rtol=1e-03,
            atol=1e-03,
        )

        self.assertTrue(
            torch.equal(
                bin_ids.long(),
                expected_bin_ids,
            )
        )

        if torch.cuda.is_available():
            (
                calibrated_prediction_gpu,
                bin_ids_gpu,
            ) = torch.ops.fbgemm.generic_histogram_binning_calibration_by_feature(
                logit=logit.cuda(),
                segment_value=segment_value.cuda(),
                segment_lengths=lengths.cuda(),
                num_segments=num_segments,
                bin_num_examples=bin_num_examples.cuda(),
                bin_num_positives=bin_num_positives.cuda(),
                bin_boundaries=bin_boundaries.cuda(),
                positive_weight=0.4,
                bin_ctr_in_use_after=10000,
                bin_ctr_weight_value=0.9995,
            )

            torch.testing.assert_close(
                calibrated_prediction_gpu,
                expected_calibrated_prediction.cuda(),
                rtol=1e-03,
                atol=1e-03,
            )

            self.assertTrue(
                torch.equal(
                    bin_ids_gpu.long(),
                    expected_bin_ids.cuda(),
                )
            )

if __name__ == "__main__":

    unittest.main()

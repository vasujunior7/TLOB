import torch
import pytest
from utils.ofi import compute_ofi_from_lob, compute_ofi_bias_matrix

def test_compute_ofi_from_lob_basic():
    # Simulate LOB data for 1 batch, 2 timesteps, 2 levels (4 features/level = 8 total LOB features)
    # Features: [BP1, BV1, AP1, AV1, BP2, BV2, AP2, AV2]
    lob_data = torch.tensor([
        [ # Batch 1
            [100, 10, 101, 12, 99, 8, 102, 15], # t=0
            [100, 15, 101, 10, 99, 9, 102, 18], # t=1
            [100, 12, 101, 11, 99, 7, 102, 16], # t=2
        ]
    ], dtype=torch.float32)

    # Expected OFI calculation:
    # t=0 (prepend logic, diff with itself is 0, so OFI is 0)
    #   L1: d_BV=0, d_AV=0 => OFI=0
    #   L2: d_BV=0, d_AV=0 => OFI=0
    # t=1
    #   L1: d_BV=(15-10)=5, d_AV=(10-12)=-2 => OFI=5 - (-2) = 7
    #   L2: d_BV=(9-8)=1, d_AV=(18-15)=3 => OFI=1 - 3 = -2
    # t=2
    #   L1: d_BV=(12-15)=-3, d_AV=(11-10)=1 => OFI=-3 - 1 = -4
    #   L2: d_BV=(7-9)=-2, d_AV=(16-18)=-2 => OFI=-2 - (-2) = 0

    expected_ofi = torch.tensor([
        [
            [0.0, 0.0],
            [7.0, -2.0],
            [-4.0, 0.0],
        ]
    ], dtype=torch.float32)

    ofi = compute_ofi_from_lob(lob_data)
    assert torch.allclose(ofi, expected_ofi, atol=1e-6)

def test_compute_ofi_from_lob_multiple_batches():
    lob_data = torch.tensor([
        [ # Batch 1
            [100, 10, 101, 12, 99, 8, 102, 15],
            [100, 15, 101, 10, 99, 9, 102, 18],
        ],
        [ # Batch 2
            [200, 20, 201, 22, 199, 18, 202, 25],
            [200, 25, 201, 20, 199, 19, 202, 28],
        ]
    ], dtype=torch.float32)

    expected_ofi_b1 = torch.tensor([
        [0.0, 0.0],
        [7.0, -2.0],
    ], dtype=torch.float32)

    expected_ofi_b2 = torch.tensor([
        [0.0, 0.0],
        [7.0, -2.0],
    ], dtype=torch.float32)

    ofi = compute_ofi_from_lob(lob_data)
    assert ofi.shape == (2, 2, 2) # batch, seq_len, num_levels
    assert torch.allclose(ofi[0], expected_ofi_b1, atol=1e-6)
    assert torch.allclose(ofi[1], expected_ofi_b2, atol=1e-6)

def test_compute_ofi_bias_matrix_basic():
    ofi_input = torch.tensor([
        [ # Batch 1, t=0
            [1.0, 2.0], # OFI for level 1, level 2
        ],
        [ # Batch 1, t=1
            [3.0, -4.0], # OFI for level 1, level 2
        ]
    ], dtype=torch.float32)

    # t=0: OFI = [1, 2]
    # Norm sq = 1^2 + 2^2 = 5
    # Outer product = [[1*1, 1*2], [2*1, 2*2]] = [[1, 2], [2, 4]]
    # Bias matrix = [[1/5, 2/5], [2/5, 4/5]] = [[0.2, 0.4], [0.4, 0.8]]

    # t=1: OFI = [3, -4]
    # Norm sq = 3^2 + (-4)^2 = 9 + 16 = 25
    # Outer product = [[3*3, 3*-4], [-4*3, -4*-4]] = [[9, -12], [-12, 16]]
    # Bias matrix = [[9/25, -12/25], [-12/25, 16/25]] = [[0.36, -0.48], [-0.48, 0.64]]

    expected_bias_matrix = torch.tensor([
        [
            [[0.2, 0.4], [0.4, 0.8]],
            [[0.36, -0.48], [-0.48, 0.64]],
        ]
    ], dtype=torch.float32)

    bias_matrix = compute_ofi_bias_matrix(ofi_input)
    assert torch.allclose(bias_matrix, expected_bias_matrix, atol=1e-6)

def test_compute_ofi_bias_matrix_zero_ofi():
    ofi_input = torch.tensor([
        [[0.0, 0.0]], # Zero OFI
    ], dtype=torch.float32)

    # With eps=1e-8, norm_sq will be eps.
    # Outer product will be [[0,0],[0,0]]
    # Bias matrix should be all zeros.
    expected_bias_matrix = torch.tensor([
        [[[0.0, 0.0], [0.0, 0.0]]]
    ], dtype=torch.float32)

    bias_matrix = compute_ofi_bias_matrix(ofi_input)
    assert torch.allclose(bias_matrix, expected_bias_matrix, atol=1e-6)

def test_compute_ofi_bias_matrix_single_level():
    ofi_input = torch.tensor([
        [[5.0]],
    ], dtype=torch.float32)

    # Norm sq = 25
    # Outer product = [[25]]
    # Bias matrix = [[1.0]]
    expected_bias_matrix = torch.tensor([
        [[[1.0]]]
    ], dtype=torch.float32)

    bias_matrix = compute_ofi_bias_matrix(ofi_input)
    assert torch.allclose(bias_matrix, expected_bias_matrix, atol=1e-6)

import torch

def compute_ofi_from_lob(lob_data: torch.Tensor) -> torch.Tensor:
    """
    Computes Order Flow Imbalance (OFI) from LOB data.

    Args:
        lob_data (torch.Tensor): LOB input tensor of shape [batch_size, seq_len, num_features].
                                 Assumes features are ordered as:
                                 [bid_price_1, bid_vol_1, ..., bid_price_N, bid_vol_N,
                                  ask_price_1, ask_vol_1, ..., ask_price_N, ask_vol_N]
                                 where N is the number of LOB levels. Each level has 4 features (BP, BV, AP, AV).
    Returns:
        torch.Tensor: OFI tensor of shape [batch_size, seq_len, num_levels].
    """
    # Assuming 10 LOB levels, each with 4 features (BP, BV, AP, AV)
    # Total features = 10 * 4 = 40 (if only LOB features)
    # Or, if all_features is true, it's 144, but OFI only uses LOB part.
    # We need to extract bid_vol and ask_vol for each level.
    # For FI-2010, num_features = 40 (10 levels * 4 features/level)

    # Reshape to easily access levels: [batch, seq_len, num_levels, 4_features_per_level]
    num_lob_features = 40 # Assuming 10 levels * 4 features/level
    if lob_data.shape[2] < num_lob_features:
        raise ValueError(f"Expected at least {num_lob_features} features for LOB data, but got {lob_data.shape[2]}")

    # Extract only the LOB part for OFI calculation
    lob_features_only = lob_data[:, :, :num_lob_features]

    # Reshape to [batch_size, seq_len, N_LOB_LEVELS, LEN_LEVEL]
    reshaped_lob = lob_features_only.reshape(lob_features_only.shape[0], lob_features_only.shape[1], -1, 4)

    # bid_vol_l is at index 1 within each level's 4 features
    # ask_vol_l is at index 3 within each level's 4 features
    bid_volumes = reshaped_lob[:, :, :, 1] # [batch_size, seq_len, num_levels]
    ask_volumes = reshaped_lob[:, :, :, 3] # [batch_size, seq_len, num_levels]

    # Calculate change in bid/ask volumes
    # ΔBid_vol(l,t) = bid_vol(l,t) - bid_vol(l,t-1)
    # ΔAsk_vol(l,t) = ask_vol(l,t) - ask_vol(l,t-1)
    delta_bid_volumes = torch.diff(bid_volumes, dim=1, prepend=bid_volumes[:, :1, :])
    delta_ask_volumes = torch.diff(ask_volumes, dim=1, prepend=ask_volumes[:, :1, :])

    # OFI(l, t) = ΔBid_vol(l,t) - ΔAsk_vol(l,t)
    ofi = delta_bid_volumes - delta_ask_volumes

    return ofi

def compute_ofi_bias_matrix(ofi: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes the OFI attention bias matrix B(l, l', t).

    Args:
        ofi (torch.Tensor): OFI tensor of shape [batch_size, seq_len, num_levels].
        eps (float): Small epsilon for numerical stability.

    Returns:
        torch.Tensor: OFI bias matrix of shape [batch_size, seq_len, num_levels, num_levels].
    """
    # OFI(l,t) · OFI(l',t) / (‖OFI(t)‖² + ε)
    # ofi shape: [B, S, L]

    # L2 norm of OFI at each timestep: ‖OFI(t)‖²
    # Keepdims for broadcasting: [B, S, 1]
    ofi_norm_sq = torch.sum(ofi**2, dim=-1, keepdim=True) + eps # [B, S, 1]

    # Outer product: OFI(l,t) * OFI(l',t)
    # [B, S, L, 1] * [B, S, 1, L] -> [B, S, L, L]
    ofi_outer_product = ofi.unsqueeze(-1) * ofi.unsqueeze(-2)

    # Divide by norm
    bias_matrix = ofi_outer_product / ofi_norm_sq.unsqueeze(-1) # [B, S, L, L]

    return bias_matrix

# TLOB Improvement Spec
## Project: TLOB (arXiv 2502.15757) — FI-2010 Benchmark Improvements

---

## Context

Base paper: "TLOB: A New Transformer Model and Dataset for Limit Order Book Forecasting"
Official repo: https://github.com/LeonardoBerti00/TLOB
Benchmark dataset: FI-2010 (10-level NASDAQ LOB, 5 stocks, 10 days)
Task: Mid-price movement classification → up / stationary / down at horizons k = 10, 20, 50, 100

TLOB architecture (from diagram):
- Input: LOB snapshot sequence shape (T, 40) — 10 bid prices + 10 bid vols + 10 ask prices + 10 ask vols
- Bilinear Normalization Layer → handles non-stationarity + price/volume magnitude mismatch
- Sinusoidal Positional Encoding → added to temporal dimension
- Nx TLOB Blocks, each containing:
  - Layer Norm (pre-norm style)
  - Temporal Self-Attention → attention across T (timestep) dimension
  - Skip connection
  - Layer Norm
  - Spatial Self-Attention → attention across F (feature/level) dimension  
  - Skip connection
  - MLPLOB Block (position-wise FFN)
- Classification head → FC → softmax over 3 classes

The plan.md should describe exactly what to implement, where in the codebase, 
and what the expected ablation table looks like.

---

## Improvement 1 (Deferred) — Volatility-Adaptive Labeling Threshold

### What TLOB currently does

Label construction in the paper uses a fixed threshold α:

```
m(t) = smoothed mid-price over next k events
l(t) = up         if  [m(t) - mid(t)] / mid(t)  >  α
l(t) = stationary if  |[m(t) - mid(t)] / mid(t)| ≤  α
l(t) = down        if  [m(t) - mid(t)] / mid(t)  < -α
```

α is a single fixed scalar for the entire dataset.

### The problem mathematically

α is constant but market noise floor is not. On high-volatility snapshots,
a 0.2% mid-price move is noise. On low-volatility snapshots, it is signal.
Using the same α produces inconsistent label SNR (signal-to-noise ratio)
across volatility regimes. The model sees contradictory training signal —
the same price move magnitude gets labeled "up" sometimes and "stationary"
other times depending on the ambient volatility.

The paper itself acknowledges TLOB degrades under high-volatility conditions.
This is the root cause.

### The fix — vol-adaptive threshold

```
log_return(t) = log(mid(t)) - log(mid(t-1))
σ(t) = std(log_return(t-N) ... log_return(t))   # rolling realized vol, N=50 snapshots

α(t) = β × σ(t)      # β is a scalar hyperparameter, tune via grid search on val set

l(t) = up         if  [m(t) - mid(t)] / mid(t)  >  α(t)
l(t) = stationary if  |[m(t) - mid(t)] / mid(t)| ≤  α(t)
l(t) = down        if  [m(t) - mid(t)] / mid(t)  < -α(t)
```

β search space: [0.5, 0.75, 1.0, 1.25, 1.5] — grid search on validation set F1_macro.
β=1.0 means threshold equals realized vol exactly.

### Mathematical property this gives you

Label SNR becomes approximately stationary across volatility regimes.
The threshold scales with the noise floor so "up" carries consistent
informational content whether vol is high or low.

### Where to implement (deferred)

- Deferred for now. This plan focuses only on CB-Focal loss and OFI attention bias.

### What to measure (deferred)

Deferred. Keep fixed-label pipeline while implementing the two selected changes below.

---

## Improvement 2 — Class-Balanced Focal Loss (Implement Now)

### What TLOB currently does

Standard cross-entropy loss, likely with inverse-frequency class weighting:

```
w_c = 1 / n_c
L = -Σ_c w_c · y_c · log(p_c)
```

### The problem mathematically

At k=10, stationary class is ~70-80% of labels on FI-2010.
Inverse frequency weighting treats the 1000th stationary sample as equally
informative as the 1st. That is wrong — marginal information from additional
samples of an already well-represented class diminishes.

Consequence: gradients are dominated by stationary class even with weighting.
The model learns a strong stationary prior and weak up/down signal.
F1_up and F1_down suffer most at short horizons.

### The fix — Class-Balanced Loss + Focal modulation

Class-Balanced (CB) Loss (Cui et al., CVPR 2019):

```
# Effective number of samples — geometric series sum
E_c = (1 - β^n_c) / (1 - β)      β ∈ [0, 1),  n_c = sample count for class c

# As n_c → ∞:  E_c → 1/(1-β)    saturates — diminishing returns captured
# As n_c → 0:  E_c → 1           single sample gets full weight

CB weight:
w_c = (1 - β) / (1 - β^n_c)

Combined with Focal modulation:
L_CB_focal = -Σ_c  w_c · (1 - p_c)^γ · y_c · log(p_c)

γ = 2  (standard focal parameter — suppresses easy well-classified examples)
β = 0.9999  (start here, tune if needed)
```

### Mathematical effect

Two suppression mechanisms acting together:
1. CB weighting: stationary class gets lower weight because its effective sample
   size saturates — each additional stationary sample contributes less.
2. Focal term: easy stationary predictions (high p_stationary) contribute
   near-zero gradient. Model forced to allocate capacity to hard up/down cases.

### Where to implement

- New file: `losses/cb_focal_loss.py`
- Class: `CBFocalLoss(nn.Module)` with params `cb_beta`, `focal_gamma`, `num_classes`
- Compute class counts n_c from FI-2010 train labels for each horizon
- Use in `models/engine.py` as a selectable loss function
- Add to Hydra config in `config/config.py` experiment section:
   - `loss_type: "ce" | "cb_focal"`
   - `cb_beta: 0.9999`
   - `focal_gamma: 2.0`

### What to measure

Ablation specifically on F1_up and F1_down at k=10 (worst imbalance horizon):
- Baseline: no weighting
- Inverse frequency weighting
- CB Loss only (γ=0)
- CB + Focal (γ=2)

---

## Improvement 3 — OFI Attention Bias on Spatial Attention (Implement After CB-Focal)

### What TLOB currently does

Spatial self-attention across LOB feature dimension F:

```
Q = X W_Q,   K = X W_K,   V = X W_V      # learned linear projections
A = softmax( QKᵀ / √d_k )                # attention weight matrix (L × L)
output = A · V
```

Learns which LOB levels attend to which purely from raw price/volume patterns.
No inductive bias — level 1 bid is initialized as equally likely to attend to
level 8 ask as to level 1 ask. Model must discover microstructure from scratch.

### The fix — OFI attention bias

Order Flow Imbalance (OFI) at each level captures buying/selling pressure:

```
OFI(l, t) = ΔBid_vol(l,t) - ΔAsk_vol(l,t)

ΔBid_vol(l,t) = bid_vol(l,t) - bid_vol(l,t-1)
ΔAsk_vol(l,t) = ask_vol(l,t) - ask_vol(l,t-1)

OFI > 0: buying pressure at level l
OFI < 0: selling pressure at level l
```

OFI bias matrix between level pairs:

```
B(l, l', t) = OFI(l,t) · OFI(l',t) / (‖OFI(t)‖² + ε)

B positive: levels l and l' share same-sign pressure (both buying or both selling)
B negative: levels l and l' have opposing pressure (tension between levels)
```

Inject into spatial attention as an additive bias:

```
A = softmax( QKᵀ/√d_k  +  λ·B )

λ: learned scalar, initialized to 0.0
```

Critical: λ initialized to 0 means at training start the model behaves
exactly like the original TLOB. λ grows only if OFI bias is actually useful.
Graceful degradation — if OFI is not informative, λ stays near 0.

### Mathematical justification

Levels with correlated OFI are reacting to the same latent order flow event.
Making them attend to each other more strongly is informationally justified by
microstructure theory — not a heuristic. The bias is a learned soft prior,
not a hard constraint.

This is the same mechanism as pair representation bias in AlphaFold and
relative position biases in modern transformers — injecting known structural
relationships into attention.

### Where to implement

- New file: `utils/ofi.py`
   - `compute_ofi_from_lob(x)` for LOB tensors
   - `compute_ofi_bias_matrix(ofi, eps=1e-8)`
- Model changes in `models/tlob.py`
   - Add optional OFI bias injection inside spatial attention only
   - Keep temporal attention unchanged
   - Add `self.ofi_lambda = nn.Parameter(torch.zeros(1))` initialized exactly to 0.0
- Wire OFI tensors from preprocessing/data module into model forward path
- Add Hydra config flag in `config/config.py`:
   - `use_ofi_bias: bool = False`

### What to measure

Compare TLOB vs TLOB+OFI on:
- F1_down specifically (OFI selling pressure signal should help most here)
- Attention weight visualization: do levels with correlated OFI attend to each
  other more after training? Plot attention heatmaps with and without OFI bias.
- λ value after training — if λ ≈ 0, OFI is not helping. Report this honestly.

---

## Implementation Order (Updated Two-Problem Plan)

Do in this exact order. Do not implement the next until the previous is verified.

1. Reproduce TLOB baseline on FI-2010 in this repo.
   If numbers don't match within ±0.5% F1 from expected local baseline, stop and debug before proceeding.

2. Implement Improvement 2 (CB-Focal Loss) first.
   Verify F1_up and F1_down improve at k=10 before moving on.

3. Implement Improvement 3 (OFI bias) next.
   Keep core TLOB block structure unchanged and inject bias only in spatial attention.

4. Run combined model (CB-Focal + OFI bias) and compare to single-change runs.

5. Adaptive labeling remains deferred and should not be implemented in this phase.

---

## Ablation Table to Produce (Updated)

| Model variant                   | Labeling | Loss      | use_ofi_bias | k=10 F1↑ | k=50 F1↑ | Hi-vol F1↑ |
|---------------------------------|----------|-----------|--------------|-----------|-----------|------------|
| TLOB baseline                   | fixed α  | CE        | false        | —         | —         | —          |
| + CB Focal Loss                 | fixed α  | CB+Focal  | false        | +Δ₁       | +Δ₁       | +Δ₁        |
| + OFI Attention Bias            | fixed α  | CE        | true         | +Δ₂       | +Δ₂       | +Δ₂        |
| + CB Focal Loss + OFI Bias      | fixed α  | CB+Focal  | true         | +Δ₃       | +Δ₃       | +Δ₃        |

Each row = one experiment run. Every run logged with full params.
Champion model = highest macro-F1 on high-vol test quartile.

---

## MLflow Logging Requirements

Every experiment must log:
- Params: model_type, horizon_k, labeling_method, beta (if adaptive),
  loss_type, loss_beta, loss_gamma, use_ofi_bias, hidden_dim, n_blocks
- Metrics: f1_up, f1_stationary, f1_down, f1_macro, cohen_kappa,
  f1_vol_q1, f1_vol_q2, f1_vol_q3, f1_vol_q4 (per volatility quartile)
- Artifacts: confusion matrix PNG, per-vol-quartile F1 bar chart,
  attention heatmap (for OFI experiment), model checkpoint, ONNX export

---

## DVC Pipeline Stages Required (Updated)

```
raw_data → [download_fi2010]
         → [compute_ofi]          # Improvement 3 preprocessing
         → [normalize]            # Bilinear normalization (existing)
         → [train]                # parameterized by model config
         → [evaluate]             # per-class + per-vol-quartile metrics
```

Each stage versioned. Any change to OFI computation triggers recomputation
of OFI, train, and evaluate.

---

## Files to Create / Modify (Updated to this repo)

### New files
- `losses/cb_focal_loss.py` — CBFocalLoss class
- `utils/ofi.py` — compute_ofi(), compute_ofi_bias_matrix()
- `tests/test_losses.py` — unit tests for CB-Focal Loss
- `tests/test_ofi.py` — unit tests for OFI computation

### Files to modify
- `models/tlob.py` — add OFI bias injection to spatial attention block
- `models/engine.py` — integrate CB-Focal loss selection and logging
- `config/config.py` — add `loss_type`, `cb_beta`, `focal_gamma`, `use_ofi_bias`
- `run.py` — pass new config fields into Engine
- `dvc.yaml` — add/update OFI stage if DVC is used

---

## Notes for Claude Code

- Do not modify the core TLOB block structure (Bilinear Norm → Temporal Attn → Spatial Attn → MLPLOB)
- The OFI bias is injected ONLY into spatial attention, not temporal attention
- λ for OFI bias must be initialized to exactly 0.0 — not a small value, exactly 0
- Use unambiguous names: `cb_beta` for CB-Focal and `focal_gamma` for focal modulation
- All experiments must be reproducible from params.yaml + dvc repro
- Every new function needs a unit test before integration

---

## Immediate Execution Checklist (Two Problems Only)

1. Baseline run with current TLOB and CE.
2. Add CB-Focal loss and rerun k=10,20,50,100.
3. Add OFI bias (λ starts at 0.0) and rerun.
4. Run combined CB-Focal + OFI.
5. Compare using: F1_up, F1_stationary, F1_down, F1_macro, and high-vol quartile F1.

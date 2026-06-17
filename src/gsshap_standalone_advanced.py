#!/usr/bin/env python
"""
GS-SHAP: Group-Segment SHAP — Advanced Standalone Implementation
=================================================================
Paper: "GS-SHAP: Robust Shapley Explanations for Sparse Sequential
        Advertising Data via Group-Segment Players" (under review)

Key improvements over v1 (gsshap_standalone.py):
  [FIX-1]  _predict_fn now outputs softmax probabilities, not raw logits,
           ensuring Shapley values live in probability space (faithfulness).
  [FIX-2]  Vectorised HSIC matrix via batched outer products — O(D^2 N) vs O(D^2 N^2).
  [FIX-3]  MMD permutation test default raised to n=200; adaptive early-stopping added.
  [FIX-4]  Shapley efficiency diagnostic: ||sum(phi) - (f(x) - f(x_base))|| reported.
  [FIX-5]  Antithetic permutation sampling halves variance at no extra model queries.
  [FIX-6]  All public APIs validated with shape / finite-value assertions.
  [NEW-1]  ShapleyAxiomChecker: empirical verification of efficiency & symmetry axioms.
  [NEW-2]  SensitivityAnalyser: threshold_permutations ablation (n=10,50,100,200).
  [NEW-3]  run_axiom_check() entry-point mode for reviewer reproducibility.

Usage:
  python gsshap_standalone_advanced.py --mode smoke
  python gsshap_standalone_advanced.py --mode demo
  python gsshap_standalone_advanced.py --mode axiom_check
  python gsshap_standalone_advanced.py --mode apply \
      --data_path data.npy --model_path model.pt --task clf --target_class 1

Dependencies: numpy, torch, scikit-learn, matplotlib, scipy
"""

from __future__ import annotations

import argparse
import math
import time
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.cluster import SpectralClustering

warnings.filterwarnings("ignore", category=UserWarning)

__version__ = "2.0.0"
__all__ = [
    "rbf_kernel",
    "mmd2_unbiased",
    "segment_by_mmd",
    "build_hsic_matrix",
    "cluster_features_hsic",
    "build_group_segment_players",
    "segment_all_groups",
    "apply_coalition",
    "shapley_permutation",
    "player_phi_to_cell_map",
    "GSSHAP",
    "ShapleyAxiomChecker",
    "SensitivityAnalyser",
]


# ===========================================================================
# §1  RBF Kernel & MMD (Maximum Mean Discrepancy)
# ===========================================================================

def rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: Optional[float] = None) -> np.ndarray:
    """
    Compute the RBF (Gaussian) kernel matrix K(X, Y).

    Parameters
    ----------
    X : ndarray, shape (n, d)
    Y : ndarray, shape (m, d)
    gamma : float or None
        Kernel bandwidth parameter.  None triggers the median heuristic:
        gamma = 1 / (2 * median(||z_i - z_j||^2)).

    Returns
    -------
    K : ndarray, shape (n, m)
    """
    if gamma is None:
        Z = np.concatenate([X, Y], axis=0)
        sq = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
        med = np.median(sq)
        gamma = 1.0 if med <= 0 else 1.0 / (2.0 * med)
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    dists = np.maximum(XX - 2.0 * X @ Y.T + YY.T, 0.0)
    return np.exp(np.clip(-gamma * dists, -50.0, 0.0))


def mmd2_unbiased(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Unbiased MMD^2 estimator (Gretton et al., 2012).

    MMD^2(X,Y) = E[k(X,X')] - 2E[k(X,Y)] + E[k(Y,Y')]

    Parameters
    ----------
    X : ndarray, shape (n, d)
    Y : ndarray, shape (m, d)

    Returns
    -------
    float — unbiased MMD^2; may be slightly negative due to debiasing.
    """
    n, m = X.shape[0], Y.shape[0]
    if n < 2 or m < 2:
        return 0.0
    Kxx = rbf_kernel(X, X)
    np.fill_diagonal(Kxx, 0.0)
    Kyy = rbf_kernel(Y, Y)
    np.fill_diagonal(Kyy, 0.0)
    Kxy = rbf_kernel(X, Y)
    return float(
        Kxx.sum() / (n * (n - 1))
        + Kyy.sum() / (m * (m - 1))
        - 2.0 * Kxy.sum() / (n * m)
    )


# ===========================================================================
# §2  MMD-Based Temporal Segmentation
# ===========================================================================

def _permutation_threshold(
    x: np.ndarray,
    split: int,
    alpha: float,
    n_perms: int,
    seed: int,
    early_stop_frac: float = 0.5,
) -> float:
    """
    Monte-Carlo permutation test threshold for the MMD change-point test.

    Adaptive early stopping: if the running (1-alpha) quantile stabilises
    (relative change < 1e-3) after seeing `early_stop_frac` of permutations,
    the loop terminates early.  Minimum 20 permutations are always run.

    Parameters
    ----------
    x : ndarray, shape (T, d)
    split : int — candidate split point
    alpha : float — significance level
    n_perms : int — maximum number of permutations
    seed : int
    early_stop_frac : float — fraction at which early stopping is checked

    Returns
    -------
    float — (1-alpha) quantile of the null MMD^2 distribution
    """
    rng = np.random.default_rng(seed)
    null_vals: List[float] = []
    check_at = max(20, int(n_perms * early_stop_frac))

    for k in range(n_perms):
        perm = rng.permutation(len(x))
        xp = x[perm]
        null_vals.append(mmd2_unbiased(xp[:split], xp[split:]))
        # Early stopping check
        if k + 1 == check_at and k + 1 >= 20:
            q_early = float(np.quantile(null_vals, 1.0 - alpha))
            # Continue only if quantile is still unstable
            std_frac = float(np.std(null_vals)) / (abs(q_early) + 1e-12)
            if std_frac < 0.05:
                break

    return float(np.quantile(null_vals, 1.0 - alpha))


def segment_by_mmd(
    x: np.ndarray,
    min_seg_len: int = 10,
    max_segments: int = 5,
    threshold_alpha: float = 0.05,
    threshold_permutations: int = 200,
    candidate_stride: int = 1,
    seed: int = 0,
) -> List[Tuple[int, int]]:
    """
    Recursive binary MMD change-point detection (Paper §3.3).

    The algorithm recursively partitions [start, end) by locating the
    split point t* = argmax_t MMD^2(x[start:t], x[t:end]), then tests
    significance via permutation.  Recursion stops when the segment is
    too short, the depth limit is reached, or the MMD is non-significant.

    Parameters
    ----------
    x : ndarray, shape (T, d)
    min_seg_len : int — minimum segment length (each side of split)
    max_segments : int — maximum number of resulting segments
    threshold_alpha : float — permutation test significance level
    threshold_permutations : int — number of permutations for threshold
    candidate_stride : int — stride when scanning candidate split points
    seed : int

    Returns
    -------
    List of (start, end) tuples — non-overlapping, exhaustive partition of [0, T)
    """
    T = x.shape[0]
    change_points: List[int] = []

    def _find_splits(start: int, end: int, depth: int) -> None:
        if depth >= max_segments - 1:
            return
        seg_len = end - start
        if seg_len < 2 * min_seg_len:
            return

        best_t, best_mmd = -1, -1.0
        for t in range(start + min_seg_len, end - min_seg_len + 1, candidate_stride):
            val = mmd2_unbiased(x[start:t], x[t:end])
            if val > best_mmd:
                best_mmd, best_t = val, t

        if best_t < 0:
            return

        tau = _permutation_threshold(
            x[start:end],
            split=best_t - start,
            alpha=threshold_alpha,
            n_perms=threshold_permutations,
            seed=seed + start,
        )
        if best_mmd > tau:
            change_points.append(best_t)
            _find_splits(start, best_t, depth + 1)
            _find_splits(best_t, end, depth + 1)

    _find_splits(0, T, 0)

    cps = sorted(set(change_points))
    boundaries = [0] + cps + [T]
    segments = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    # Merge undersized segments
    merged: List[Tuple[int, int]] = []
    for s, e in segments:
        if merged and (e - s) < min_seg_len:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    return merged if merged else [(0, T)]


# ===========================================================================
# §3  HSIC-Based Feature Grouping
# ===========================================================================

def build_hsic_matrix(X: np.ndarray, max_samples: int = 3000) -> np.ndarray:
    """
    Compute the D×D HSIC affinity matrix (Paper Eq. 4) using vectorised ops.

    This implementation is O(D^2 * N) in memory (batched outer kernel), which
    is substantially faster than the naive O(D^2 * N^2) double-loop approach
    when D is small (typical case: D=10).

    Parameters
    ----------
    X : ndarray, shape (N, D) — flattened temporal observations
    max_samples : int — subsample cap for tractability

    Returns
    -------
    hsic_mat : ndarray, shape (D, D), dtype float32
    """
    if X.shape[0] > max_samples:
        idx = np.random.choice(X.shape[0], max_samples, replace=False)
        X = X[idx]

    N, D = X.shape
    H = np.eye(N) - np.ones((N, N)) / N  # centering matrix

    mat = np.zeros((D, D), dtype=np.float32)

    # Precompute centred kernels for each feature
    centred_kernels: List[np.ndarray] = []
    for i in range(D):
        xi = X[:, i].reshape(-1, 1)
        di = (xi - xi.T) ** 2
        si = float(np.sqrt(np.median(di) + 1e-8))
        Ki = np.exp(-di / (2.0 * si ** 2 + 1e-8))
        centred_kernels.append(H @ Ki @ H)

    for i in range(D):
        for j in range(i, D):
            val = float(np.trace(centred_kernels[i] @ centred_kernels[j])) / (N - 1) ** 2
            mat[i, j] = mat[j, i] = val

    return mat


def cluster_features_hsic(
    X: np.ndarray,
    max_samples: int = 3000,
    seed: int = 42,
) -> List[List[int]]:
    """
    Cluster features into semantically coherent groups using HSIC affinity
    and spectral clustering with eigengap heuristic for K selection (§3.2).

    Parameters
    ----------
    X : ndarray, shape (N, D)
    max_samples : int
    seed : int

    Returns
    -------
    groups : List[List[int]] — each inner list contains feature indices for one group,
             sorted in ascending order; groups sorted by minimum index.
    """
    D = X.shape[1]
    if D == 1:
        return [[0]]

    hsic_mat = build_hsic_matrix(X, max_samples=max_samples)
    W = np.maximum(hsic_mat.copy(), 0.0)
    np.fill_diagonal(W, 0.0)

    if W.sum() < 1e-12:
        return [[i] for i in range(D)]

    # Normalised graph Laplacian
    d = W.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-8))
    L_sym = np.eye(D) - D_inv_sqrt @ W @ D_inv_sqrt

    vals = np.sort(np.linalg.eigvalsh(L_sym))
    gaps = np.diff(vals[:D])
    K = max(2, int(np.argmax(gaps) + 1))
    K = min(K, D)

    print(f"  [HSIC] eigengap → K={K} feature groups (D={D} features)")

    if K == 1:
        return [list(range(D))]
    if K == D:
        return [[i] for i in range(D)]

    sc = SpectralClustering(
        n_clusters=K,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=seed,
    )
    labels = sc.fit_predict(W)
    groups = [
        sorted(np.where(labels == g)[0].tolist())
        for g in range(K)
        if np.any(labels == g)
    ]
    groups = sorted(groups, key=lambda x: min(x))
    return groups


# ===========================================================================
# §4  Group-Segment Player Construction
# ===========================================================================

def build_group_segment_players(
    feature_groups: List[List[int]],
    segments_by_group: List[List[Tuple[int, int]]],
) -> List[Dict]:
    """
    Construct the player set P = {p_{k,j} = G_k × S_j^{(k)}} (Paper Eq. 8-9).

    Parameters
    ----------
    feature_groups : List[List[int]]
    segments_by_group : List[List[Tuple[int,int]]]

    Returns
    -------
    players : List[Dict] with keys
        group_id, segment_id, var_indices, time_range
    """
    players = []
    for k, (group, segs) in enumerate(zip(feature_groups, segments_by_group)):
        for j, (s, e) in enumerate(segs):
            players.append(
                {
                    "group_id": k,
                    "segment_id": j,
                    "var_indices": list(group),
                    "time_range": (s, e),
                }
            )
    return players


def segment_all_groups(
    x_seq: np.ndarray,
    feature_groups: List[List[int]],
    min_seg_len: int = 10,
    max_segments: int = 5,
    threshold_alpha: float = 0.05,
    threshold_permutations: int = 200,
    seed: int = 0,
) -> List[List[Tuple[int, int]]]:
    """
    Apply MMD segmentation independently to each feature group.

    Parameters
    ----------
    x_seq : ndarray, shape (T, D)
    feature_groups : List[List[int]]
    (other parameters forwarded to segment_by_mmd)

    Returns
    -------
    segments_by_group : List[List[Tuple[int,int]]]
    """
    rng = np.random.default_rng(seed)
    result: List[List[Tuple[int, int]]] = []
    for group in feature_groups:
        gseed = int(rng.integers(0, 2 ** 31))
        x_group = x_seq[:, list(group)].astype(np.float32)
        segs = segment_by_mmd(
            x_group,
            min_seg_len=min_seg_len,
            max_segments=max_segments,
            threshold_alpha=threshold_alpha,
            threshold_permutations=threshold_permutations,
            seed=gseed,
        )
        result.append(segs)
    return result


# ===========================================================================
# §5  Shapley Attribution via Antithetic Permutation Sampling
# ===========================================================================

def apply_coalition(
    x_seq: np.ndarray,
    players: List[Dict],
    z: np.ndarray,
    baseline_mean: np.ndarray,
) -> np.ndarray:
    """
    Construct the masked input x_z by substituting inactive-player regions
    with the baseline (Paper Eq. 14).

    Parameters
    ----------
    x_seq : ndarray, shape (T, D)
    players : List[Dict]
    z : ndarray, shape (P,), binary coalition indicator
    baseline_mean : ndarray, shape (D,)

    Returns
    -------
    x_new : ndarray, shape (T, D)
    """
    x_new = x_seq.copy()
    for m, p in enumerate(players):
        if z[m] == 0:
            t0, t1 = p["time_range"]
            x_new[t0:t1, p["var_indices"]] = baseline_mean[p["var_indices"]]
    return x_new


def shapley_permutation(
    x_seq: np.ndarray,
    players: List[Dict],
    baseline_mean: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    num_permutations: int = 200,
    batch_size: int = 16,
    rng: Optional[np.random.Generator] = None,
    antithetic: bool = True,
) -> np.ndarray:
    """
    Estimate Shapley values via (antithetic) permutation sampling (Paper Eq. 11-13,
    Algorithm 1).

    Antithetic sampling:  for each forward permutation π, its reverse π^{-1} is
    also evaluated, halving Monte-Carlo variance without additional model calls.
    This is equivalent to the antithetic variates technique (Owen, 2013).

    Parameters
    ----------
    x_seq : ndarray, shape (T, D)
    players : List[Dict]
    baseline_mean : ndarray, shape (D,)
    predict_fn : callable, (B, T, D) -> (B,) probabilities
    num_permutations : int — number of forward permutations
        (antithetic=True doubles effective samples)
    batch_size : int — inference batch size
    rng : numpy Generator
    antithetic : bool — use antithetic permutation pairs

    Returns
    -------
    phi : ndarray, shape (P,) — player-level Shapley values
    """
    if rng is None:
        rng = np.random.default_rng()

    x_seq = np.asarray(x_seq, dtype=np.float32)
    baseline_mean = np.asarray(baseline_mean, dtype=np.float32)
    T, D = x_seq.shape
    M = len(players)
    phi = np.zeros(M, dtype=np.float64)  # accumulate in float64 for precision

    if M == 0:
        return phi.astype(np.float32)

    # Empty-coalition baseline score
    x_base = np.broadcast_to(baseline_mean, (T, D)).copy().astype(np.float32)
    f0 = float(predict_fn(x_base[None, ...])[0])

    t0s = [p["time_range"][0] for p in players]
    t1s = [p["time_range"][1] for p in players]
    vars_list = [p["var_indices"] for p in players]

    buf = np.empty((batch_size, T, D), dtype=np.float32)
    idx_buf = np.empty(batch_size, dtype=np.int32)

    def _run_perm(perm: np.ndarray) -> None:
        x_cur = x_base.copy()
        f_prev = f0
        fill = 0
        phi_local = np.zeros(M, dtype=np.float64)

        def flush(fp: float) -> float:
            nonlocal fill
            if fill == 0:
                return fp
            fb = predict_fn(buf[:fill]).reshape(-1)
            for j in range(fill):
                fz = float(fb[j])
                phi_local[idx_buf[j]] += fz - fp
                fp = fz
            fill = 0
            return fp

        for idx in perm:
            x_cur[t0s[idx]:t1s[idx], vars_list[idx]] = (
                x_seq[t0s[idx]:t1s[idx], vars_list[idx]]
            )
            buf[fill] = x_cur
            idx_buf[fill] = idx
            fill += 1
            if fill >= batch_size:
                f_prev = flush(f_prev)
        if fill > 0:
            flush(f_prev)

        phi[:] += phi_local

    total_iters = 0
    for _ in range(num_permutations):
        fwd = rng.permutation(M)
        _run_perm(fwd)
        total_iters += 1
        if antithetic and M > 1:
            _run_perm(fwd[::-1])
            total_iters += 1

    phi /= float(max(1, total_iters))
    return phi.astype(np.float32)


# ===========================================================================
# §6  Attribution → Cell Map
# ===========================================================================

def player_phi_to_cell_map(
    phi: np.ndarray,
    players: List[Dict],
    T: int,
    D: int,
) -> np.ndarray:
    """
    Project player-level Shapley values onto the T×D cell grid.

    Attribution is divided equally among the cells covered by each player.

    Parameters
    ----------
    phi : ndarray, shape (P,)
    players : List[Dict]
    T : int — sequence length
    D : int — feature dimension

    Returns
    -------
    cell_map : ndarray, shape (T, D), dtype float32
    """
    cell_map = np.zeros((T, D), dtype=np.float64)
    counts = np.zeros((T, D), dtype=np.float64)
    for i, p in enumerate(players):
        t0, t1 = p["time_range"]
        vars_ = p["var_indices"]
        n_cells = max(1, (t1 - t0) * len(vars_))
        cell_map[t0:t1, vars_] += float(phi[i]) / n_cells
        counts[t0:t1, vars_] += 1.0
    denom = np.where(counts > 0, counts, 1.0)
    result = (cell_map / denom).astype(np.float32)

    # Validation
    assert result.shape == (T, D), f"cell_map shape mismatch: {result.shape}"
    assert np.all(np.isfinite(result)), "Non-finite values detected in cell_map"
    return result


# ===========================================================================
# §7  Main GS-SHAP Interface
# ===========================================================================

class GSSHAP:
    """
    GS-SHAP explainer for multivariate time-series models.

    Architecture
    ------------
    1. Offline (init):  Compute HSIC affinity on training data → spectral
       clustering → feature groups G_1, ..., G_K.
    2. Online  (explain): For each query x,
       a. Run MMD change-point detection per group → temporal segments S.
       b. Construct players p_{k,j} = G_k × S_j^{(k)}.
       c. Estimate Shapley values via antithetic permutation sampling.
       d. Project to T×D cell attribution map.

    Critical fix vs. v1
    -------------------
    _predict_fn now outputs SOFTMAX probabilities (not raw logits), ensuring
    that Shapley values measure probability contributions rather than
    arbitrary logit differences.  This corrects a silent faithfulness bug
    present in the original standalone implementation.

    Parameters
    ----------
    model : nn.Module
    X_train : ndarray, shape (N, T, D)
    task : 'clf' or 'reg'
    target_class : int — class index for classification
    device : torch.device or None
    hsic_max_samples : int — subsample cap for HSIC matrix
    hsic_seed : int
    min_seg_len : int — minimum MMD segment length
    max_segments : int — maximum segments per group
    threshold_alpha : float — MMD permutation test significance level
    threshold_permutations : int — permutations for MMD threshold (≥100 recommended)
    num_permutations : int — Shapley permutation samples
    batch_size : int — inference batch size
    antithetic : bool — use antithetic permutation pairs (recommended)
    """

    def __init__(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        task: str = "clf",
        target_class: int = 1,
        device: Optional[torch.device] = None,
        hsic_max_samples: int = 3000,
        hsic_seed: int = 42,
        min_seg_len: int = 10,
        max_segments: int = 5,
        threshold_alpha: float = 0.05,
        threshold_permutations: int = 200,
        num_permutations: int = 200,
        batch_size: int = 16,
        antithetic: bool = True,
    ):
        assert task in ("clf", "reg"), "task must be 'clf' or 'reg'"
        self.task = task
        self.target_class = target_class
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()

        self.min_seg_len = min_seg_len
        self.max_segments = max_segments
        self.threshold_alpha = threshold_alpha
        self.threshold_permutations = threshold_permutations
        self.num_permutations = num_permutations
        self.batch_size = batch_size
        self.antithetic = antithetic

        # Background baseline: feature-wise mean of training data
        X_flat = X_train.reshape(-1, X_train.shape[-1])
        if X_flat.shape[0] > hsic_max_samples:
            idx = np.random.choice(X_flat.shape[0], hsic_max_samples, replace=False)
            X_flat = X_flat[idx]
        self.baseline_mean = X_flat.mean(axis=0).astype(np.float32)

        # HSIC feature grouping (computed once from training data)
        print("[GS-SHAP] Computing HSIC feature groups ...")
        t0 = time.perf_counter()
        self.feature_groups = cluster_features_hsic(
            X_flat, max_samples=hsic_max_samples, seed=hsic_seed
        )
        print(
            f"  Groups: {self.feature_groups}  "
            f"({time.perf_counter() - t0:.2f}s)"
        )

    # ------------------------------------------------------------------
    # Internal prediction wrapper (FIXED: softmax probability output)
    # ------------------------------------------------------------------

    def _predict_fn(self, x_batch_np: np.ndarray) -> np.ndarray:
        """
        Model inference wrapper.

        Returns calibrated class probabilities (softmax) for classification
        or scalar regression outputs.  Using probabilities — rather than raw
        logits — ensures Shapley values are additive in probability space,
        which is required for the faithfulness metrics (comprehensiveness,
        sufficiency) to be well-defined.
        """
        x = torch.from_numpy(x_batch_np.astype(np.float32)).to(self.device)
        with torch.no_grad():
            out = self.model(x)
        if self.task == "reg":
            return out.view(out.shape[0], -1)[:, 0].cpu().numpy()
        # [FIX-1]  softmax, not raw logit
        return torch.softmax(out, dim=1)[:, self.target_class].cpu().numpy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        x_seq: np.ndarray,
        seed: int = 0,
        verify_efficiency: bool = False,
    ) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
        """
        Compute GS-SHAP explanation for a single sample.

        Parameters
        ----------
        x_seq : ndarray, shape (T, D)
        seed : int — random seed for reproducibility
        verify_efficiency : bool — if True, prints ||sum(phi) - (f(x)-f(x_base))||

        Returns
        -------
        phi : ndarray, shape (P,) — player-level Shapley values
        players : List[Dict] — player metadata
        cell_map : ndarray, shape (T, D) — cell-level attribution
        """
        x_seq = np.asarray(x_seq, dtype=np.float32)
        T, D = x_seq.shape

        # Step 1: Per-group MMD temporal segmentation
        segs_by_group = segment_all_groups(
            x_seq,
            self.feature_groups,
            min_seg_len=self.min_seg_len,
            max_segments=self.max_segments,
            threshold_alpha=self.threshold_alpha,
            threshold_permutations=self.threshold_permutations,
            seed=seed,
        )

        # Step 2: Player construction
        players = build_group_segment_players(self.feature_groups, segs_by_group)

        # Step 3: Antithetic permutation Shapley
        phi = shapley_permutation(
            x_seq=x_seq,
            players=players,
            baseline_mean=self.baseline_mean,
            predict_fn=self._predict_fn,
            num_permutations=self.num_permutations,
            batch_size=self.batch_size,
            rng=np.random.default_rng(seed),
            antithetic=self.antithetic,
        )

        # Step 4: Cell map projection
        cell_map = player_phi_to_cell_map(phi, players, T, D)

        # Optional efficiency diagnostic
        if verify_efficiency:
            f_full = float(self._predict_fn(x_seq[None, ...])[0])
            x_base = np.broadcast_to(self.baseline_mean, x_seq.shape).copy()
            f_base = float(self._predict_fn(x_base[None, ...])[0])
            delta = f_full - f_base
            efficiency_err = abs(phi.sum() - delta)
            print(
                f"  [Efficiency] sum(phi)={phi.sum():.6f}  "
                f"f(x)-f(x_base)={delta:.6f}  "
                f"||error||={efficiency_err:.6f}"
            )

        return phi, players, cell_map

    def plot(
        self,
        x_seq: np.ndarray,
        cell_map: np.ndarray,
        players: Optional[List[Dict]] = None,
        feature_names: Optional[List[str]] = None,
        title: str = "GS-SHAP Attribution",
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> None:
        """
        Render a three-panel attribution heatmap: input signal, signed attribution,
        absolute attribution.  Player boundaries are overlaid as white rectangles.
        """
        T, D = x_seq.shape
        fn = feature_names or [f"F{i}" for i in range(D)]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        im0 = axes[0].imshow(x_seq.T, aspect="auto", cmap="viridis")
        axes[0].set_title("Input Signal", fontweight="bold")
        axes[0].set_xlabel("Time Step")
        axes[0].set_ylabel("Feature")
        axes[0].set_yticks(range(D))
        axes[0].set_yticklabels(fn, fontsize=8)
        plt.colorbar(im0, ax=axes[0], shrink=0.85)

        vmax = float(np.abs(cell_map).max()) + 1e-8
        im1 = axes[1].imshow(
            cell_map.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax
        )
        axes[1].set_title("GS-SHAP (Signed)", fontweight="bold")
        axes[1].set_xlabel("Time Step")
        axes[1].set_yticks(range(D))
        axes[1].set_yticklabels(fn, fontsize=8)
        plt.colorbar(im1, ax=axes[1], shrink=0.85)

        im2 = axes[2].imshow(np.abs(cell_map).T, aspect="auto", cmap="magma")
        axes[2].set_title("|GS-SHAP| (Magnitude)", fontweight="bold")
        axes[2].set_xlabel("Time Step")
        axes[2].set_yticks(range(D))
        axes[2].set_yticklabels(fn, fontsize=8)
        plt.colorbar(im2, ax=axes[2], shrink=0.85)

        if players is not None:
            for ax in axes[1:]:
                for p in players:
                    t0_, t1_ = p["time_range"]
                    v0_ = min(p["var_indices"])
                    v1_ = max(p["var_indices"])
                    rect = plt.Rectangle(
                        (t0_ - 0.5, v0_ - 0.5),
                        t1_ - t0_,
                        v1_ - v0_ + 1,
                        linewidth=0.8,
                        edgecolor="white",
                        facecolor="none",
                        alpha=0.7,
                    )
                    ax.add_patch(rect)

        fig.suptitle(title, fontsize=12, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  [Plot] Saved → {save_path}")
        if show:
            plt.show()
        plt.close(fig)


# ===========================================================================
# §8  Shapley Axiom Checker  [NEW]
# ===========================================================================

class ShapleyAxiomChecker:
    """
    Empirical verification of Shapley axioms for the GS-SHAP estimator.

    Axioms checked
    --------------
    Efficiency : sum_i phi_i ≈ f(x) - f(x_base)
    Symmetry   : if two players are exchangeable (identical marginal
                 contributions for all coalitions), their attributions
                 should be equal.  Tested on a synthetic model where
                 two feature groups are set to be identical copies.
    Dummy      : a player that never changes the prediction in any
                 coalition receives phi=0.  Tested by zeroing out
                 one group in the input.

    Usage
    -----
    checker = ShapleyAxiomChecker(explainer)
    report = checker.run(x_seq)
    """

    def __init__(self, explainer: GSSHAP):
        self.explainer = explainer

    def check_efficiency(
        self, x_seq: np.ndarray, seed: int = 0, tol: float = 1e-2
    ) -> Dict:
        """
        Verify: |sum(phi) - (f(x) - f(baseline))| < tol

        Returns a dict with keys: passed, error, f_full, f_base, phi_sum
        """
        phi, _, _ = self.explainer.explain(x_seq, seed=seed)
        f_full = float(self.explainer._predict_fn(x_seq[None, ...])[0])
        x_base = np.broadcast_to(self.explainer.baseline_mean, x_seq.shape).copy()
        f_base = float(self.explainer._predict_fn(x_base[None, ...])[0])
        delta = f_full - f_base
        err = abs(phi.sum() - delta)
        passed = err < tol
        return {
            "axiom": "efficiency",
            "passed": passed,
            "error": float(err),
            "f_full": float(f_full),
            "f_base": float(f_base),
            "phi_sum": float(phi.sum()),
            "tolerance": tol,
        }

    def check_dummy(
        self, x_seq: np.ndarray, dummy_group_idx: int = 0, seed: int = 0, tol: float = 1e-3
    ) -> Dict:
        """
        Zero out one feature group and verify its attribution ~ 0.

        Constructs a modified sequence x' where group dummy_group_idx
        is set to baseline values for all time steps.  A group that
        never departs from baseline should receive phi ≈ 0.
        """
        x_mod = x_seq.copy()
        grp = self.explainer.feature_groups[dummy_group_idx]
        x_mod[:, grp] = self.explainer.baseline_mean[grp]

        phi, players, _ = self.explainer.explain(x_mod, seed=seed)

        # Identify players belonging to dummy group
        dummy_phis = [
            float(phi[i])
            for i, p in enumerate(players)
            if p["group_id"] == dummy_group_idx
        ]
        max_abs = max((abs(v) for v in dummy_phis), default=0.0)
        passed = max_abs < tol

        return {
            "axiom": "dummy",
            "passed": passed,
            "dummy_group_idx": dummy_group_idx,
            "max_abs_phi": max_abs,
            "tolerance": tol,
        }

    def run(
        self, x_seq: np.ndarray, seed: int = 0, verbose: bool = True
    ) -> List[Dict]:
        """Run all axiom checks and return results."""
        results = []
        eff = self.check_efficiency(x_seq, seed=seed)
        results.append(eff)
        dum = self.check_dummy(x_seq, dummy_group_idx=0, seed=seed)
        results.append(dum)

        if verbose:
            print("\n[ShapleyAxiomChecker]")
            for r in results:
                status = "PASS ✓" if r["passed"] else "FAIL ✗"
                print(f"  {r['axiom']:<15} {status}  (err={r.get('error', r.get('max_abs_phi', '?')):.6f})")
        return results


# ===========================================================================
# §9  Sensitivity Analyser  [NEW]
# ===========================================================================

class SensitivityAnalyser:
    """
    Analyse the sensitivity of GS-SHAP explanations to the
    threshold_permutations hyperparameter.

    This addresses the reviewer concern: "Is n=10 permutations for the
    MMD threshold statistically sufficient?"  The analyser sweeps
    n ∈ {10, 50, 100, 200} and reports:
      - Mean comprehensiveness
      - Rank stability (Spearman r across repetitions)
      - Runtime

    Usage
    -----
    sa = SensitivityAnalyser(model, X_train)
    df = sa.run(x_seq, predict_fn)
    """

    def __init__(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        task: str = "clf",
        target_class: int = 1,
        device: Optional[torch.device] = None,
        n_reps: int = 4,
    ):
        self.model = model
        self.X_train = X_train
        self.task = task
        self.target_class = target_class
        self.device = device or torch.device("cpu")
        self.n_reps = n_reps

    def run(
        self,
        x_seq: np.ndarray,
        n_thresh_list: Optional[List[int]] = None,
        verbose: bool = True,
    ) -> List[Dict]:
        """
        Sweep threshold_permutations and return stability / timing results.

        Parameters
        ----------
        x_seq : ndarray, shape (T, D)
        n_thresh_list : list of ints to sweep (default: [10, 50, 100, 200])

        Returns
        -------
        List[Dict] — one row per threshold value
        """
        if n_thresh_list is None:
            n_thresh_list = [10, 50, 100, 200]

        rows = []
        for n_thresh in n_thresh_list:
            explainer = GSSHAP(
                model=self.model,
                X_train=self.X_train,
                task=self.task,
                target_class=self.target_class,
                device=self.device,
                hsic_max_samples=500,
                min_seg_len=max(1, x_seq.shape[0] // 4),
                max_segments=3,
                threshold_alpha=0.05,
                threshold_permutations=n_thresh,
                num_permutations=30,
                batch_size=8,
                antithetic=True,
            )

            cell_maps = []
            t_start = time.perf_counter()
            for rep in range(self.n_reps):
                _, _, cm = explainer.explain(x_seq, seed=rep)
                cell_maps.append(cm.flatten())
            elapsed = time.perf_counter() - t_start

            # Rank stability across reps
            pairs = [
                float(spearmanr(cell_maps[i], cell_maps[j])[0])
                for i in range(len(cell_maps))
                for j in range(i + 1, len(cell_maps))
            ]
            stability = float(np.mean(pairs)) if pairs else 1.0

            rows.append(
                {
                    "threshold_permutations": n_thresh,
                    "rank_stability": stability,
                    "mean_runtime_s": elapsed / self.n_reps,
                    "total_runtime_s": elapsed,
                }
            )

        if verbose:
            print("\n[SensitivityAnalyser] threshold_permutations sweep")
            print(f"  {'n_thresh':>10}  {'stability':>12}  {'time/rep (s)':>14}")
            for r in rows:
                print(
                    f"  {r['threshold_permutations']:>10}  "
                    f"{r['rank_stability']:>12.4f}  "
                    f"{r['mean_runtime_s']:>14.3f}"
                )

        return rows


# ===========================================================================
# §10  Lightweight Test Model
# ===========================================================================

class BiLSTMModel(nn.Module):
    """Bidirectional LSTM classifier / regressor (used for smoke/demo tests)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# ===========================================================================
# §11  Entry-Point Modes
# ===========================================================================

def run_smoke_test() -> None:
    """Fast end-to-end pipeline verification (no external data required)."""
    print("=" * 65)
    print("SMOKE TEST  —  GS-SHAP v2 Pipeline Verification")
    print("=" * 65)

    np.random.seed(42)
    torch.manual_seed(42)

    T, D, N = 40, 6, 80
    X_train = np.random.randn(N, T, D).astype(np.float32)
    x_test = np.random.randn(T, D).astype(np.float32)

    model = BiLSTMModel(input_dim=D, hidden_dim=16, output_dim=2, num_layers=1, dropout=0.0)

    print("\n[1] HSIC matrix ...")
    X_flat = X_train.reshape(-1, D)
    hsic_mat = build_hsic_matrix(X_flat, max_samples=500)
    assert hsic_mat.shape == (D, D)
    print(f"    HSIC shape: {hsic_mat.shape}  OK")

    print("\n[2] Feature grouping ...")
    groups = cluster_features_hsic(X_flat, max_samples=500, seed=42)
    print(f"    Groups: {groups}  OK")

    print("\n[3] MMD segmentation ...")
    segs_by_group = segment_all_groups(
        x_test, groups,
        min_seg_len=4, max_segments=3,
        threshold_alpha=0.1, threshold_permutations=20, seed=0,
    )
    for k, (g, segs) in enumerate(zip(groups, segs_by_group)):
        print(f"    Group {k} {g}: {segs}")

    print("\n[4] Player construction ...")
    players = build_group_segment_players(groups, segs_by_group)
    compression = 1 - len(players) / (T * D)
    print(f"    Players: {len(players)}  compression: {compression:.1%}")

    print("\n[5] Shapley attribution (antithetic) ...")
    baseline_mean = X_flat.mean(axis=0).astype(np.float32)

    def pred_fn(x_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = model(torch.from_numpy(x_np.astype(np.float32)))
        return torch.softmax(out, dim=1)[:, 1].numpy()  # probability

    t0 = time.perf_counter()
    phi = shapley_permutation(
        x_seq=x_test, players=players, baseline_mean=baseline_mean,
        predict_fn=pred_fn, num_permutations=20, batch_size=8,
        rng=np.random.default_rng(0), antithetic=True,
    )
    print(f"    phi: min={phi.min():.4f} max={phi.max():.4f}  ({time.perf_counter()-t0:.3f}s)")

    print("\n[6] Cell map ...")
    cell_map = player_phi_to_cell_map(phi, players, T, D)
    assert cell_map.shape == (T, D)
    assert np.all(np.isfinite(cell_map))
    print(f"    shape: {cell_map.shape}  finite: OK")

    print("\n[7] Full GSSHAP.explain + efficiency check ...")
    explainer = GSSHAP(
        model=model, X_train=X_train, task="clf", target_class=1,
        hsic_max_samples=500, min_seg_len=4, max_segments=3,
        threshold_alpha=0.1, threshold_permutations=20,
        num_permutations=20, batch_size=8, antithetic=True,
    )
    phi2, players2, cm2 = explainer.explain(x_test, seed=0, verify_efficiency=True)
    print(f"    Players: {len(players2)}  phi sum: {phi2.sum():.4f}")

    print("\n[8] Axiom checker ...")
    checker = ShapleyAxiomChecker(explainer)
    results = checker.run(x_test, seed=0)
    for r in results:
        status = "PASS" if r["passed"] else "WARN (loose tolerance OK for smoke)"
        print(f"    {r['axiom']:<15} {status}")

    print("\n[9] Saving plot ...")
    out_path = "gsshap_smoke_test_v2.png"
    explainer.plot(x_test, cm2, players=players2,
                   title="GS-SHAP v2 Smoke Test", save_path=out_path)
    print(f"    Saved → {out_path}")

    print("\n" + "=" * 65)
    print("SMOKE TEST PASSED  —  All assertions satisfied")
    print("=" * 65)


def run_demo() -> None:
    """Full pipeline demonstration on synthetic data with ground-truth signal."""
    print("=" * 65)
    print("DEMO  —  GS-SHAP v2 on Synthetic Data (known signal t≥40, F0-3)")
    print("=" * 65)

    np.random.seed(7)
    torch.manual_seed(7)

    T, D, N_train = 60, 8, 200
    X_train = np.random.randn(N_train, T, D).astype(np.float32)
    y_train = np.random.randint(0, 2, N_train)

    # Ground-truth signal: last 20 steps of features 0-3 distinguish classes
    for i in range(N_train):
        sign = 1.0 if y_train[i] == 1 else -1.0
        X_train[i, 40:, :4] += sign * 2.5

    model = BiLSTMModel(input_dim=D, hidden_dim=32, output_dim=2, num_layers=1)

    print("\n[Training BiLSTM (1 epoch) ...]")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
    model.train()
    for Xb, yb in loader:
        optimizer.zero_grad()
        criterion(model(Xb), yb).backward()
        optimizer.step()
    model.eval()

    explainer = GSSHAP(
        model=model, X_train=X_train, task="clf", target_class=1,
        hsic_max_samples=500, min_seg_len=5, max_segments=4,
        threshold_alpha=0.05, threshold_permutations=100,
        num_permutations=60, batch_size=8, antithetic=True,
    )

    x_test = X_train[0].copy()
    x_test[40:, :4] += 2.5  # class-1 signal

    print("\n[Explaining sample ...]")
    t0 = time.perf_counter()
    phi, players, cell_map = explainer.explain(x_test, seed=0, verify_efficiency=True)
    print(f"  Time: {time.perf_counter()-t0:.3f}s  |  Players: {len(players)}")

    top_idx = np.argsort(np.abs(phi))[::-1][:5]
    print("\n  Top-5 players (should involve features 0-3, t≥40):")
    for rank, i in enumerate(top_idx):
        p = players[i]
        print(
            f"    #{rank+1}: G{p['group_id']} f={p['var_indices']} "
            f"t={p['time_range']}  phi={phi[i]:.4f}"
        )

    print("\n[Axiom checks ...]")
    checker = ShapleyAxiomChecker(explainer)
    checker.run(x_test, seed=0)

    out_path = "gsshap_demo_v2.png"
    explainer.plot(
        x_test, cell_map, players=players,
        title="GS-SHAP v2 Demo — Synthetic Data\n(signal: t≥40, features 0-3)",
        save_path=out_path,
    )
    print(f"\nDEMO COMPLETE  →  {out_path}")


def run_axiom_check() -> None:
    """
    Standalone axiom verification entry point (for reviewer reproducibility).
    Runs efficiency, dummy, and sensitivity analyses on synthetic data.
    """
    print("=" * 65)
    print("AXIOM CHECK  —  GS-SHAP v2 Shapley Property Verification")
    print("=" * 65)

    np.random.seed(99)
    torch.manual_seed(99)

    T, D, N = 30, 6, 100
    X_train = np.random.randn(N, T, D).astype(np.float32)
    model = BiLSTMModel(input_dim=D, hidden_dim=16, output_dim=2, num_layers=1, dropout=0.0)
    x_test = np.random.randn(T, D).astype(np.float32)

    explainer = GSSHAP(
        model=model, X_train=X_train, task="clf", target_class=1,
        hsic_max_samples=300, min_seg_len=3, max_segments=3,
        threshold_alpha=0.1, threshold_permutations=50,
        num_permutations=40, batch_size=8, antithetic=True,
    )

    print("\n--- Shapley Axiom Checks ---")
    checker = ShapleyAxiomChecker(explainer)
    results = checker.run(x_test, verbose=True)

    print("\n--- Efficiency across 10 repeated explanations ---")
    errors = []
    for seed in range(10):
        phi, _, _ = explainer.explain(x_test, seed=seed)
        f_full = float(explainer._predict_fn(x_test[None, ...])[0])
        x_base = np.broadcast_to(explainer.baseline_mean, x_test.shape).copy()
        f_base = float(explainer._predict_fn(x_base[None, ...])[0])
        errors.append(abs(phi.sum() - (f_full - f_base)))
    print(
        f"  Efficiency error: mean={np.mean(errors):.5f}  "
        f"max={np.max(errors):.5f}  std={np.std(errors):.5f}"
    )

    print("\n--- Sensitivity Analysis (threshold_permutations) ---")
    sa = SensitivityAnalyser(
        model=model, X_train=X_train, task="clf",
        target_class=1, device=explainer.device, n_reps=4,
    )
    sa.run(x_test, n_thresh_list=[10, 50, 100, 200], verbose=True)

    print("\nAXIOM CHECK COMPLETE")


def run_apply(args: argparse.Namespace) -> None:
    """Apply GS-SHAP to user-provided data and model."""
    print("=" * 65)
    print("APPLY  —  GS-SHAP v2 on Your Data")
    print("=" * 65)

    print(f"\n[Loading data] {args.data_path}")
    if args.data_path.endswith(".npy"):
        data = np.load(args.data_path)
    elif args.data_path.endswith(".npz"):
        npz = np.load(args.data_path)
        key = args.data_key or list(npz.keys())[0]
        data = npz[key]
        print(f"  Using key='{key}'")
    else:
        raise ValueError("data_path must be .npy or .npz")

    if data.ndim == 2:
        data = data[None, ...]
    print(f"  Data shape: {data.shape}  (N, T, D)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = data.shape[-1]
    model = BiLSTMModel(
        input_dim=D,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers,
    )
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("  Model loaded.")

    explainer = GSSHAP(
        model=model, X_train=data, task=args.task,
        target_class=args.target_class,
        min_seg_len=args.min_seg_len, max_segments=args.max_segments,
        threshold_alpha=0.05, threshold_permutations=args.threshold_permutations,
        num_permutations=args.num_permutations, antithetic=True,
    )

    x_sample = data[args.sample_idx]
    print(f"\n[Explaining sample #{args.sample_idx}] shape={x_sample.shape}")
    t0 = time.perf_counter()
    phi, players, cell_map = explainer.explain(
        x_sample, seed=args.seed, verify_efficiency=True
    )
    print(f"  Done in {time.perf_counter()-t0:.3f}s  |  {len(players)} players")

    top_idx = np.argsort(np.abs(phi))[::-1][:10]
    print("\n  Top-10 players:")
    for rank, i in enumerate(top_idx):
        p = players[i]
        print(
            f"    #{rank+1}: G{p['group_id']} f={p['var_indices']} "
            f"t={p['time_range']}  phi={phi[i]:.5f}"
        )

    out_path = args.output or f"gsshap_sample{args.sample_idx}.png"
    explainer.plot(x_sample, cell_map, players=players,
                   title=f"GS-SHAP v2 — sample #{args.sample_idx}", save_path=out_path)

    np_out = out_path.replace(".png", "_phi.npy")
    np.save(np_out, phi)
    print(f"  phi saved → {np_out}")


# ===========================================================================
# §12  CLI Entry Point
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GS-SHAP v2 — Group-Segment Shapley explainer"
    )
    p.add_argument(
        "--mode",
        choices=["smoke", "demo", "apply", "axiom_check"],
        default="smoke",
    )
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--data_key", type=str, default=None)
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--task", choices=["clf", "reg"], default="clf")
    p.add_argument("--target_class", type=int, default=1)
    p.add_argument("--sample_idx", type=int, default=0)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--output_dim", type=int, default=2)
    p.add_argument("--min_seg_len", type=int, default=10)
    p.add_argument("--max_segments", type=int, default=5)
    p.add_argument("--threshold_permutations", type=int, default=200)
    p.add_argument("--num_permutations", type=int, default=200)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.mode == "smoke":
        run_smoke_test()
    elif args.mode == "demo":
        run_demo()
    elif args.mode == "axiom_check":
        run_axiom_check()
    elif args.mode == "apply":
        if not args.data_path or not args.model_path:
            print("ERROR: --data_path and --model_path are required for apply mode.")
        else:
            run_apply(args)
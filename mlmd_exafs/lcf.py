"""Linear combination fitting (LCF) of simulated EXAFS spectra to experiment.

Given an experimental chi(k) and a library of simulated chi(k) "standards"
(e.g. ``<savefile>-chi_avg.dat`` files from several candidate structures), the
fit proceeds in two steps:

1. For every standard, determine an individual delta E0 by shifting it against
   the experimental spectrum and minimizing the selected metric on
   k^n-weighted chi(k).
2. For every combination of 1..``max_components`` standards, fit the relative
   weights of the E0-shifted standards (weights >= 0, sum(weights) = 1; a
   single standard has weight 1). How delta E0 is treated here depends on
   ``e0_mode``:

   * ``"joint"`` (default): the n weights and n delta E0 values (one per
     standard) are refined together in one SLSQP optimization.
   * ``"joint_shared"``: the n weights and ONE delta E0 shared by all
     standards of the combination are refined together (like Athena's single
     E0 shift).
   * ``"fixed"``: delta E0 values stay at their step-1 values and only the
     weights are fitted (the original procedure).

   Combinations are ranked by the metric.

Joint modes start from the step-1 delta E0 values plus the fixed-mode weights,
and from ``n_starts - 1`` further starts with every delta E0 offset by
+-``e0_start_step``, +-2*``e0_start_step``, ... eV (the metric is multimodal in
E0). The best result is kept. In ``"joint"`` mode the fixed-mode solution is
the first start, so the joint result is never worse than it on the same k
points.

Within one joint optimization the metric is evaluated on a fixed set of k
points: those where every standard of the combination stays inside its k range
at both delta E0 bounds (the shift is monotonic in delta E0, so they stay
valid over the whole range). If that set fails the ``min_valid_frac`` check,
the points valid at the step-1 values are used instead and each standard's
delta E0 bounds are narrowed so those points never leave its k range.

Reduced chi-square counts (n - 1) weights plus n (``"joint"``), 1
(``"joint_shared"``) or 0 (``"fixed"``) delta E0 values as free parameters.
1-sigma uncertainties of the weights and delta E0 values come from the
Jacobian of the k-weighted residual (covariance = redchi * (J^T J)^-1); they
are NaN for weights pinned at 0 or 1, for delta E0 values at a bound, and for
delta E0 values that were not fitted (``"fixed"``).

E0 convention: a standard shifted by delta E0 is evaluated on the
experimental k grid at

    k_query^2 = k_exp^2 + e0_sign * K2_PER_EV * delta_E0

with K2_PER_EV = 0.262468426 (k in A^-1, E in eV). The default
``e0_sign=-1`` follows the Artemis/IFEFFIT convention (theory evaluated at
k^2 = k_exp^2 - K2_PER_EV * delta_E0, so a positive delta E0 moves the theory
edge up in energy) and matches :mod:`mlmd_exafs.fitting_E0`. ``e0_sign=+1``
flips the sign of every reported delta E0 (weights and metrics are
unchanged).

Experimental and standard files may be ``.dat`` (whitespace-delimited) or
``.csv`` (comma-delimited); the first two numeric columns are read as k and
chi(k).

Used by ``mlmd-exafs lcf``, the MCP server and the SciLink plug-in.
"""

from __future__ import annotations

import csv
import glob
import itertools
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .fitting_E0 import read_experimental
from .lcf_summary import write_top_candidates, write_top_fits

# k^2 = 0.262468426 * E, with k in Angstrom^-1 and E in eV
K2_PER_EV = 0.262468426

METRIC_ALIASES = {
    "redchi": "redchi",
    "reduced_chi_square": "redchi",
    "reduced_chi2": "redchi",
    "chi2red": "redchi",
    "chi2": "chi2",
    "chi_square": "chi2",
    "chi_squared": "chi2",
    "rmsd": "rmsd",
    "rmse": "rmsd",
    "rfactor": "rfactor",
    "r-factor": "rfactor",
    "r_factor": "rfactor",
}

METRICS = ("redchi", "chi2", "rmsd", "rfactor")


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def canonical_metric(metric: str) -> str:
    key = metric.lower().strip()
    if key not in METRIC_ALIASES:
        raise ValueError(
            f"Unknown metric {metric!r}. Use one of: "
            f"{', '.join(sorted(set(METRIC_ALIASES)))}"
        )
    return METRIC_ALIASES[key]


def load_spectrum(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load k and chi(k) from a .dat or .csv file (sorted, duplicate k removed)."""
    k, chi = read_experimental(path)
    k, idx = np.unique(k, return_index=True)
    return k, chi[idx]


def resolve_standards(standards: str | Iterable[str]) -> list[str]:
    """Expand a glob pattern or a list of paths/patterns into sorted file paths."""
    patterns = [standards] if isinstance(standards, str) else list(standards)
    files: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        files.extend(matches if matches else [pattern])
    missing = [f for f in files if not os.path.isfile(f)]
    if missing:
        raise FileNotFoundError(f"Standard file(s) not found: {', '.join(missing)}")
    if not files:
        raise ValueError("No standard files given.")
    return list(dict.fromkeys(files))


def _standard_names(files: list[str]) -> list[str]:
    """Label standards by basename, falling back to the path when not unique."""
    basenames = [Path(f).name for f in files]
    return [
        b if basenames.count(b) == 1 else str(Path(f)) for b, f in zip(basenames, files)
    ]


def restrict_k_range(k, chi, kmin=None, kmax=None):
    mask = np.ones_like(k, dtype=bool)
    if kmin is not None:
        mask &= k >= kmin
    if kmax is not None:
        mask &= k <= kmax
    return k[mask], chi[mask]


def apply_k_weight(k, y, kweight):
    if kweight == 0:
        return y
    return y * k**kweight


def shifted_standard_on_exp_grid(k_exp, k_std, chi_std, delta_e0, e0_sign=-1.0):
    """Interpolate an E0-shifted standard onto the experimental k grid.

    Points that fall outside the standard's k range are NaN.
    """
    q2 = k_exp**2 + e0_sign * K2_PER_EV * delta_e0
    q = np.full_like(k_exp, np.nan, dtype=float)
    valid = q2 >= 0.0
    q[valid] = np.sqrt(q2[valid])
    return np.interp(q, k_std, chi_std, left=np.nan, right=np.nan)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def calculate_metrics(y, yfit, n_params: int = 0) -> dict[str, float]:
    """Chi-square, reduced chi-square, RMSD and R-factor of a fit.

    No experimental uncertainties are available, so chi2 = sum((y - yfit)^2).
    """
    y = np.asarray(y, dtype=float)
    yfit = np.asarray(yfit, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yfit)
    y = y[mask]
    yfit = yfit[mask]

    if y.size == 0:
        return {"n_points": 0, "chi2": np.inf, "redchi": np.inf, "rmsd": np.inf, "rfactor": np.inf}

    resid = y - yfit
    chi2 = float(np.sum(resid**2))
    dof = max(int(y.size - n_params), 1)
    denom = float(np.sum(y**2))
    return {
        "n_points": int(y.size),
        "chi2": chi2,
        "redchi": chi2 / dof,
        "rmsd": float(np.sqrt(np.mean(resid**2))),
        "rfactor": chi2 / denom if denom > 0 else np.inf,
    }


def metric_score(y, yfit, metric, n_params=0):
    return calculate_metrics(y, yfit, n_params=n_params)[canonical_metric(metric)]


# ---------------------------------------------------------------------------
# Step 1: delta E0 per standard
# ---------------------------------------------------------------------------


def determine_delta_e0_for_standard(
    k_exp,
    chi_exp,
    k_std,
    chi_std,
    metric="redchi",
    e0_bounds=(-20.0, 20.0),
    e0_sign=-1.0,
    kweight=2,
    scale_for_shift=True,
    min_valid_frac=0.95,
):
    """Determine delta E0 for one standard against the experimental spectrum.

    If ``scale_for_shift`` is True, a temporary non-negative amplitude is
    optimized during the E0 search so spectral shape, not amplitude, drives the
    shift. That amplitude is not used in the LCF.

    Returns ``(delta_e0, score, success)``.
    """
    from scipy.optimize import minimize_scalar

    metric = canonical_metric(metric)
    y_exp = apply_k_weight(k_exp, chi_exp, kweight)
    min_valid = max(5, int(np.ceil(min_valid_frac * k_exp.size)))

    def objective(delta_e0):
        shifted = shifted_standard_on_exp_grid(k_exp, k_std, chi_std, delta_e0, e0_sign)
        valid = np.isfinite(shifted)
        if valid.sum() < min_valid:
            return np.inf

        y = y_exp[valid]
        s = apply_k_weight(k_exp[valid], shifted[valid], kweight)

        if scale_for_shift:
            denom = float(np.dot(s, s))
            if denom <= 0:
                return np.inf
            amp = max(float(np.dot(y, s) / denom), 0.0)
        else:
            amp = 1.0

        return metric_score(y, amp * s, metric, n_params=1)

    result = minimize_scalar(
        objective, bounds=e0_bounds, method="bounded", options={"xatol": 1e-4}
    )
    return float(result.x), float(result.fun), bool(result.success)


# ---------------------------------------------------------------------------
# Step 2: LCF over combinations
# ---------------------------------------------------------------------------

E0_MODES = ("joint", "joint_shared", "fixed")

# Weights within this distance of 0 or 1, and delta E0 values within this
# distance (eV) of a bound, count as pinned: no uncertainty is reported.
_PIN_TOL = 1e-6


def canonical_e0_mode(e0_mode: str) -> str:
    key = e0_mode.lower().strip()
    if key not in E0_MODES:
        raise ValueError(f"Unknown e0_mode {e0_mode!r}. Use one of: {', '.join(E0_MODES)}")
    return key


def n_free_params(n: int, e0_mode: str) -> int:
    """Free parameters of an n-standard fit: (n - 1) weights plus the E0 terms."""
    n_e0 = {"joint": n, "joint_shared": 1, "fixed": 0}[canonical_e0_mode(e0_mode)]
    return max(n - 1, 0) + n_e0


def _shifted_matrix(k_exp, standards, combo, delta_e0_by_name, e0_sign, kweight):
    """Columns = each shifted (and k-weighted) standard on the experimental grid."""
    cols = []
    for name in combo:
        k_std, chi_std = standards[name]
        shifted = shifted_standard_on_exp_grid(
            k_exp, k_std, chi_std, delta_e0_by_name[name], e0_sign
        )
        cols.append(apply_k_weight(k_exp, shifted, kweight))
    return np.column_stack(cols)


def _e0_bounds_keeping_points(k, k_std, e0_sign, e0_bounds):
    """delta E0 interval (within ``e0_bounds``) keeping every k inside the standard's range."""
    # e0_sign * K2_PER_EV * delta_e0 must lie in [q_min^2 - min k^2, q_max^2 - max k^2]
    lo_q2 = max(float(np.min(k_std)), 0.0) ** 2 - float(np.min(k)) ** 2
    hi_q2 = float(np.max(k_std)) ** 2 - float(np.max(k)) ** 2
    a, b = sorted((lo_q2 / (e0_sign * K2_PER_EV), hi_q2 / (e0_sign * K2_PER_EV)))
    # small inward margin so rounding in sqrt() never pushes a point outside
    return max(a + 1e-8, e0_bounds[0]), min(b - 1e-8, e0_bounds[1])


def _uncertainties(residual, weights, delta_e0, e0_lo, e0_hi, n_fit_e0, redchi):
    """1-sigma errors of weights and delta E0 values from a finite-difference Jacobian.

    ``residual(w, e)`` returns the k-weighted residual for weights ``w`` and
    E0 parameters ``e`` (length ``n_fit_e0``: n, 1 or 0). The weights keep
    their sum constraint by treating the last unpinned weight as dependent.
    Pinned weights and E0 parameters at a bound get NaN.
    """
    n = len(weights)
    w_err = np.full(n, np.nan)
    e_err = np.full(n_fit_e0, np.nan)

    free_w = [i for i in range(n) if _PIN_TOL < weights[i] < 1.0 - _PIN_TOL] if n > 1 else []
    indep_w = free_w[:-1]
    free_e = [j for j in range(n_fit_e0)
              if e0_lo[j] + _PIN_TOL < delta_e0[j] < e0_hi[j] - _PIN_TOL]

    n_par = len(indep_w) + len(free_e)
    if n_par == 0 or not np.isfinite(redchi):
        return w_err, e_err

    def res_at(theta):
        w = np.array(weights, dtype=float)
        e = np.array(delta_e0, dtype=float)
        if indep_w:
            dw = theta[:len(indep_w)] - w[indep_w]
            w[indep_w] += dw
            w[free_w[-1]] -= np.sum(dw)
        e[free_e] = theta[len(indep_w):]
        return residual(w, e)

    theta0 = np.concatenate([np.asarray(weights, float)[indep_w],
                             np.asarray(delta_e0, float)[free_e]])
    steps = np.concatenate([np.full(len(indep_w), 1e-6), np.full(len(free_e), 1e-4)])
    jac = np.empty((res_at(theta0).size, n_par))
    for i, h in enumerate(steps):
        tp, tm = theta0.copy(), theta0.copy()
        tp[i] += h
        tm[i] -= h
        jac[:, i] = (res_at(tp) - res_at(tm)) / (2 * h)

    jtj = jac.T @ jac
    if not np.all(np.isfinite(jtj)) or np.linalg.cond(jtj) > 1e14:
        return w_err, e_err
    cov = redchi * np.linalg.inv(jtj)

    m = len(indep_w)
    for pos, i in enumerate(indep_w):
        w_err[i] = np.sqrt(cov[pos, pos])
    if free_w:
        # dependent weight = 1 - sum(independent) - sum(pinned)
        w_err[free_w[-1]] = np.sqrt(np.sum(cov[:m, :m]))
    for pos, j in enumerate(free_e):
        e_err[j] = np.sqrt(cov[m + pos, m + pos])
    return w_err, e_err


def fit_lcf_combo(
    k_exp,
    chi_exp,
    standards,
    combo,
    delta_e0_by_name,
    metric="redchi",
    e0_sign=-1.0,
    kweight=2,
    min_valid_frac=0.95,
    e0_mode="fixed",
    e0_bounds=(-20.0, 20.0),
    n_starts=5,
    e0_start_step=2.5,
):
    """Fit weights (and, in joint modes, delta E0 values) for one combination.

    Weights are >= 0 and sum to 1 (SLSQP); a single standard has weight 1.

    Parameters
    ----------
    delta_e0_by_name : dict
        Step-1 delta E0 per standard. Used as-is in ``"fixed"`` mode and as the
        initial guess in the joint modes.
    e0_mode : {"fixed", "joint", "joint_shared"}
        ``"fixed"`` (default here, for backward compatibility) fits only the
        weights. ``"joint"`` refines one delta E0 per standard together with
        the weights; ``"joint_shared"`` refines one delta E0 shared by all
        standards of the combination together with the weights.
    e0_bounds : tuple of float
        delta E0 bounds (eV) for the joint modes.
    n_starts : int
        Starting points per center in the joint modes: offsets 0,
        +step, -step, +2*step, ... applied to every delta E0. ``"joint"`` has
        one center (the step-1 values); ``"joint_shared"`` has the mean of the
        step-1 values plus each distinct step-1 value.
    e0_start_step : float
        Offset (eV) between starting points.

    Returns
    -------
    dict or None
        ``combo``, ``weights``, ``weight_err``, ``delta_e0`` (fitted, one per
        standard), ``delta_e0_err``, ``delta_e0_initial`` (step 1),
        ``metrics``, ``n_params``, ``e0_mode``, ``mask`` ("bounds" or
        "initial", the k points the metric was evaluated on), ``success`` and
        ``message``. None if too few valid k points overlap.
    """
    from scipy.optimize import minimize

    metric = canonical_metric(metric)
    e0_mode = canonical_e0_mode(e0_mode)
    n = len(combo)
    n_params = n_free_params(n, e0_mode)
    init_e0 = np.array([delta_e0_by_name[name] for name in combo], dtype=float)
    min_valid = max(5, int(np.ceil(min_valid_frac * k_exp.size)))
    y_all = apply_k_weight(k_exp, chi_exp, kweight)

    def design(k, e_vec):
        return np.column_stack([
            apply_k_weight(k, shifted_standard_on_exp_grid(k, *standards[name], e, e0_sign), kweight)
            for name, e in zip(combo, e_vec)
        ])

    if e0_mode == "fixed":
        X_all = _shifted_matrix(k_exp, standards, combo, delta_e0_by_name, e0_sign, kweight)

        valid = np.isfinite(y_all) & np.all(np.isfinite(X_all), axis=1)
        if valid.sum() < min_valid:
            return None

        y = y_all[valid]
        X = X_all[valid, :]
        k_m = k_exp[valid]

        if n == 1:
            weights = np.array([1.0])
            success, message = True, "single standard; weight fixed to 1"
        else:
            # The weights are optimized with the original parameter count
            # (n E0 + n - 1 weights) so they match earlier releases bit for
            # bit; only the reported metrics use the corrected count.
            n_params_opt = n + max(n - 1, 0)
            x0 = np.full(n, 1.0 / n)
            result = minimize(
                lambda w: metric_score(y, X @ w, metric, n_params=n_params_opt),
                x0,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * n,
                constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
                options={"ftol": 1e-12, "maxiter": 1000},
            )
            weights = np.clip(result.x, 0.0, 1.0)
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else x0
            success, message = bool(result.success), str(result.message)

        metrics = calculate_metrics(y, X @ weights, n_params=n_params)
        w_err, _ = _uncertainties(
            lambda w, e: y - X @ w, weights, [], [], [], 0, metrics["redchi"]
        )
        return {
            "combo": combo,
            "weights": weights,
            "weight_err": w_err,
            "delta_e0": init_e0.copy(),
            "delta_e0_err": np.full(n, np.nan),
            "delta_e0_initial": init_e0,
            "metrics": metrics,
            "n_params": n_params,
            "e0_mode": e0_mode,
            "mask": "initial",
            "success": success,
            "message": message,
        }

    # ---- joint modes: fixed set of k points for the whole optimization ----
    lo, hi = float(e0_bounds[0]), float(e0_bounds[1])
    valid = np.isfinite(y_all)
    for name in combo:
        for bound in (lo, hi):
            valid &= np.isfinite(
                shifted_standard_on_exp_grid(k_exp, *standards[name], bound, e0_sign)
            )
    if valid.sum() >= min_valid:
        mask_kind = "bounds"
        std_bounds = [(lo, hi)] * n
    else:
        # Too restrictive: use the points valid at the step-1 values and narrow
        # each standard's delta E0 range so none of them can become NaN.
        valid = np.isfinite(y_all) & np.all(np.isfinite(design(k_exp, init_e0)), axis=1)
        if valid.sum() < min_valid:
            return None
        mask_kind = "initial"
        std_bounds = [
            _e0_bounds_keeping_points(k_exp[valid], standards[name][0], e0_sign, (lo, hi))
            for name in combo
        ]

    if e0_mode == "joint":
        e_bounds = std_bounds
    else:
        e_bounds = [(max(b[0] for b in std_bounds), min(b[1] for b in std_bounds))]
        if e_bounds[0][0] > e_bounds[0][1]:
            return None
    n_e = len(e_bounds)
    e_lo = np.array([b[0] for b in e_bounds])
    e_hi = np.array([b[1] for b in e_bounds])

    y = y_all[valid]
    k_m = k_exp[valid]
    n_w = n if n > 1 else 0

    def expand(e):
        return np.asarray(e, dtype=float) if e0_mode == "joint" else np.full(n, float(e[0]))

    def split(p):
        w = np.asarray(p[:n_w], dtype=float) if n_w else np.array([1.0])
        return w, np.asarray(p[n_w:], dtype=float)

    def objective(p):
        w, e = split(p)
        yfit = design(k_m, expand(e)) @ w
        if not np.all(np.isfinite(yfit)):
            return 1e30
        return metric_score(y, yfit, metric, n_params=n_params)

    # Starting points: the fixed-mode solution first, then E0 offsets.
    fixed = fit_lcf_combo(
        k_exp, chi_exp, standards, combo, delta_e0_by_name, metric=metric,
        e0_sign=e0_sign, kweight=kweight, min_valid_frac=min_valid_frac,
        e0_mode="fixed",
    )
    w0 = fixed["weights"] if fixed is not None else np.full(n, 1.0 / n)
    offsets = [0.0]
    for i in range(1, max(int(n_starts), 1)):
        offsets.append(((i + 1) // 2) * e0_start_step * (1 if i % 2 else -1))
    if e0_mode == "joint":
        centers = [init_e0]
    else:
        centers = [np.array([init_e0.mean()])] + [
            np.array([v]) for v in dict.fromkeys(init_e0.tolist())
        ]
    starts = []
    for center in centers:
        for off in offsets:
            e = np.clip(center + off, e_lo, e_hi)
            starts.append(np.concatenate([w0[:n_w], e]))

    bounds = [(0.0, 1.0)] * n_w + list(e_bounds)
    constraints = (
        [{"type": "eq", "fun": lambda p: np.sum(p[:n_w]) - 1.0}] if n_w else []
    )

    best_val, best_p, success, message = objective(starts[0]), starts[0], True, (
        "kept initial values (no optimizer run improved on them)"
    )
    for x0 in starts:
        val = objective(x0)
        if val < best_val:
            best_val, best_p = val, x0
        result = minimize(
            objective, x0, method="SLSQP", bounds=bounds, constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        p = np.asarray(result.x, dtype=float).copy()
        if n_w:
            w = np.clip(p[:n_w], 0.0, 1.0)
            p[:n_w] = w / np.sum(w) if np.sum(w) > 0 else w0
        p[n_w:] = np.clip(p[n_w:], e_lo, e_hi)
        val = objective(p)
        if val < best_val:
            best_val, best_p = val, p
            success, message = bool(result.success), str(result.message)

    weights, e = split(best_p)
    delta_e0 = expand(e)
    metrics = calculate_metrics(y, design(k_m, delta_e0) @ weights, n_params=n_params)
    w_err, e_err = _uncertainties(
        lambda w, ee: y - design(k_m, expand(ee)) @ w,
        weights, e, e_lo, e_hi, n_e, metrics["redchi"],
    )
    return {
        "combo": combo,
        "weights": weights,
        "weight_err": w_err,
        "delta_e0": delta_e0,
        "delta_e0_err": expand(e_err) if e0_mode == "joint" else np.full(n, e_err[0]),
        "delta_e0_initial": init_e0,
        "metrics": metrics,
        "n_params": n_params,
        "e0_mode": e0_mode,
        "mask": mask_kind,
        "success": success,
        "message": message,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_R_MAX_PLOT = 6.0  # upper R limit (A) of the R-space plot


def _plot_exp_vs_fit(ax, x_exp, y_exp, x_fit, y_fit):
    """Experiment as white dots over a light-gray smooth curve, LCF in red."""
    from scipy.interpolate import make_interp_spline

    x_dense = np.linspace(x_exp.min(), x_exp.max(), 10 * len(x_exp))
    y_dense = make_interp_spline(x_exp, y_exp, k=3)(x_dense)
    ax.plot(x_dense, y_dense, color="0.8", lw=1.2, zorder=1)
    ax.plot(x_exp, y_exp, linestyle="none", marker="o", markersize=4,
            markerfacecolor="white", markeredgecolor="black", markeredgewidth=0.8,
            zorder=2, label="experiment")
    ax.plot(x_fit, y_fit, color="red", lw=2, zorder=3, label="LCF")
    ax.legend(frameon=False, fontsize=9)


def _plot_best_fit(k, chi_exp, chi_fit, k_path, r_path, kweight=2, dk_win=1.0):
    """Write the k-space (k^2 chi) and R-space (|chi(R)|) experiment-vs-LCF plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .analysis import xftf

    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_exp_vs_fit(ax, k, k**2 * chi_exp, k, k**2 * chi_fit)
    ax.set_xlabel(r"$k$ [$\AA^{-1}$]")
    ax.set_ylabel(r"$k^2\chi(k)$")
    fig.tight_layout()
    fig.savefig(k_path, dpi=300)
    plt.close(fig)

    # FT on a uniform k grid, Hanning window kept inside the fitted k range
    k_u = np.arange(k.min(), k.max(), 0.05)
    kmin_w, kmax_w = k.min() + dk_win / 2, k.max() - dk_win / 2
    if kmax_w <= kmin_w:
        kmin_w, kmax_w, dk_win = k.min(), k.max(), 0.0
    ft = {}
    for name, chi in (("exp", chi_exp), ("fit", chi_fit)):
        r, mag = xftf(k_u, np.interp(k_u, k, chi), kmin=kmin_w, kmax=kmax_w,
                      dk_win=dk_win, kweight=kweight)
        ft[name] = mag
    in_r = r <= _R_MAX_PLOT
    # dots every ~0.1 A; the dense FT stays as the fit line
    step = max(1, int(round(0.1 / (r[1] - r[0]))))
    r_dots = r[in_r][::step]
    mag_dots = ft["exp"][in_r][::step]

    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_exp_vs_fit(ax, r_dots, mag_dots, r[in_r], ft["fit"][in_r])
    ax.set_xlim(0, _R_MAX_PLOT)
    ax.set_xlabel(r"$R$ [$\AA$]")
    ax.set_ylabel(rf"$|\chi(R)|$ [$\AA^{{-{kweight + 1}}}$]")
    fig.tight_layout()
    fig.savefig(r_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_lcf(
    exp_file: str,
    standards: str | Iterable[str],
    outdir: str = "lcf_fit",
    metric: str = "redchi",
    max_components: int = 3,
    kmin: float | None = None,
    kmax: float | None = None,
    k_weight: int = 2,
    e0_min: float = -20.0,
    e0_max: float = 20.0,
    e0_sign: float = -1.0,
    scale_for_shift: bool = True,
    min_valid_frac: float = 0.95,
    n_top: int = 10,
    verbose: bool = True,
    e0_mode: str = "joint",
    n_starts: int = 5,
    e0_start_step: float = 2.5,
) -> dict[str, Any]:
    """Linear combination fit of simulated standards to an experimental chi(k).

    Parameters
    ----------
    exp_file : str
        Experimental chi(k), ``.dat`` or ``.csv`` (columns k, chi).
    standards : str or iterable of str
        Glob pattern(s) or paths of simulated chi(k) files (``.dat``/``.csv``),
        e.g. ``"sims/*-chi_avg.dat"``.
    outdir : str
        Output directory.
    metric : str
        Ranking/optimization metric: redchi, chi2, rmsd, or rfactor.
    max_components : int
        Largest number of standards combined in one fit.
    kmin, kmax : float, optional
        Experimental k range used in the fit (A^-1).
    k_weight : int
        k-weight n applied to chi(k) for the E0 search and LCF.
    e0_min, e0_max : float
        delta E0 bounds (eV) for the per-standard search and the joint fits.
    e0_sign : float
        E0 sign convention; -1 (default) is Artemis/IFEFFIT, see module
        docstring.
    scale_for_shift : bool
        Optimize a temporary amplitude during each E0 search.
    min_valid_frac : float
        Minimum fraction of experimental points the shifted standards must
        cover for a fit to count.
    n_top : int
        Number of top-ranked fits included in the returned dict.
    verbose : bool
        Print progress and the best fit (``fit.log`` is written either way).
    e0_mode : {"joint", "joint_shared", "fixed"}
        How delta E0 is treated in the combination fits: refined per standard
        together with the weights (``"joint"``), refined as one shift shared
        by the combination (``"joint_shared"``), or fixed at the per-standard
        values (``"fixed"``, the original procedure). See module docstring.
    n_starts : int
        Starting points per joint fit (E0 offsets 0, +-step, +-2*step, ...).
    e0_start_step : float
        Offset (eV) between joint-fit starting points.

    Writes ``lcf_results.csv`` (all fits, ranked), ``delta_e0.csv``,
    ``best_lcf_fit.dat``, ``best_lcf_fit.png`` (k^2 chi(k)),
    ``best_lcf_fit_R.png`` (|chi(R)|), ``top_fits.csv`` (best fit summary),
    ``top_<n_top>_candidates.csv`` (the ``n_top`` best fits, one row each) and
    ``fit.log`` (the progress output plus the returned dict as JSON) into
    ``outdir``.

    Returns
    -------
    dict
        ``best`` (standards, weights, weight_err, delta_e0_eV,
        delta_e0_err_eV, delta_e0_initial_eV, metrics, n_params, e0_mode),
        ``top`` (list of the ``n_top`` best fits), ``delta_e0`` (per-standard
        step-1 values), ``e0_mode``, ``n_standards``, ``n_fits``, and output
        file paths.
    """
    metric = canonical_metric(metric)
    e0_mode = canonical_e0_mode(e0_mode)
    if max_components < 1:
        raise ValueError("max_components must be >= 1")
    if e0_min >= e0_max:
        raise ValueError("e0_min must be smaller than e0_max")

    k_exp, chi_exp = load_spectrum(exp_file)
    k_exp, chi_exp = restrict_k_range(k_exp, chi_exp, kmin=kmin, kmax=kmax)
    if k_exp.size < 5:
        raise RuntimeError("Too few experimental k-points after k-range selection.")

    files = resolve_standards(standards)
    names = _standard_names(files)
    library = {name: load_spectrum(f) for name, f in zip(names, files)}

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # progress lines go to the terminal (if verbose) and always to fit.log
    log_lines: list[str] = []

    def log(line: str = "") -> None:
        log_lines.append(line)
        if verbose:
            print(line)

    log(f"Experimental file: {exp_file}")
    log(f"Standards loaded:  {len(library)}")
    log(f"k range:           {k_exp.min():.4f} to {k_exp.max():.4f}")
    log(f"k-weight:          {k_weight}")
    log(f"Optimization:      {metric}")
    log(f"E0 mode:           {e0_mode}")
    log()
    log("Determining delta E0 for each standard...")

    # Step 1: delta E0 per standard
    delta_e0_by_name: dict[str, float] = {}
    e0_rows = []
    for name, (k_std, chi_std) in library.items():
        de0, score, success = determine_delta_e0_for_standard(
            k_exp,
            chi_exp,
            k_std,
            chi_std,
            metric=metric,
            e0_bounds=(e0_min, e0_max),
            e0_sign=e0_sign,
            kweight=k_weight,
            scale_for_shift=scale_for_shift,
            min_valid_frac=min_valid_frac,
        )
        delta_e0_by_name[name] = de0
        e0_rows.append(
            {"standard": name, "delta_e0_eV": de0, f"e0_fit_{metric}": score, "success": success}
        )
        status = "ok" if success else "check"
        log(f"{name:45s} deltaE0 = {de0:9.4f} eV   {metric} = {score:.6g}   {status}")
    e0_rows.sort(key=lambda r: r[f"e0_fit_{metric}"])

    # Step 2: LCF over all 1..max_components combinations
    max_components = min(max_components, len(names))
    results = []
    for n in range(1, max_components + 1):
        for combo in itertools.combinations(names, n):
            fit = fit_lcf_combo(
                k_exp,
                chi_exp,
                library,
                combo,
                delta_e0_by_name,
                metric=metric,
                e0_sign=e0_sign,
                kweight=k_weight,
                min_valid_frac=min_valid_frac,
                e0_mode=e0_mode,
                e0_bounds=(e0_min, e0_max),
                n_starts=n_starts,
                e0_start_step=e0_start_step,
            )
            if fit is not None:
                results.append(fit)

    if not results:
        raise RuntimeError("No valid LCF fits were produced.")

    results.sort(key=lambda r: r["metrics"][metric])

    def summarize(rank, r):
        return {
            "rank": rank,
            "n_components": len(r["combo"]),
            "standards": list(r["combo"]),
            "weights": [float(w) for w in r["weights"]],
            "weight_err": [float(w) for w in r["weight_err"]],
            "delta_e0_eV": [float(e) for e in r["delta_e0"]],
            "delta_e0_err_eV": [float(e) for e in r["delta_e0_err"]],
            "delta_e0_initial_eV": [float(e) for e in r["delta_e0_initial"]],
            "e0_mode": r["e0_mode"],
            "optimized_metric": metric,
            **{m: float(r["metrics"][m]) for m in METRICS},
            "n_points": r["metrics"]["n_points"],
            "n_params": r["n_params"],
            "mask": r["mask"],
            "success": r["success"],
            "message": r["message"],
        }

    ranked = [summarize(i, r) for i, r in enumerate(results, start=1)]
    best = ranked[0]

    # All results, ranked
    results_path = out / "lcf_results.csv"
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["rank", "n_components", "optimized_metric", "optimized_score",
                  *METRICS, "n_points", "n_params", "e0_mode", "mask", "success",
                  "message", "standards", "weights", "delta_e0_eV"]
        for i in range(1, max_components + 1):
            header += [f"std_{i}", f"weight_{i}", f"weight_err_{i}",
                       f"delta_e0_eV_{i}", f"delta_e0_err_eV_{i}",
                       f"delta_e0_initial_eV_{i}"]
        writer.writerow(header)
        for row in ranked:
            line = [row["rank"], row["n_components"], metric, row[metric],
                    *(row[m] for m in METRICS), row["n_points"], row["n_params"],
                    row["e0_mode"], row["mask"], row["success"], row["message"],
                    " | ".join(row["standards"]),
                    " | ".join(f"{w:.10g}" for w in row["weights"]),
                    " | ".join(f"{e:.10g}" for e in row["delta_e0_eV"])]
            for i in range(max_components):
                if i < row["n_components"]:
                    line += [row["standards"][i], row["weights"][i],
                             row["weight_err"][i], row["delta_e0_eV"][i],
                             row["delta_e0_err_eV"][i], row["delta_e0_initial_eV"][i]]
                else:
                    line += [""] * 6
            writer.writerow(line)

    # Per-standard delta E0
    delta_e0_path = out / "delta_e0.csv"
    with open(delta_e0_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(e0_rows[0]))
        writer.writeheader()
        writer.writerows(e0_rows)

    # Best-fit curve (unweighted chi, plus k^2-weighted columns), built with the
    # best combination's own fitted delta E0 values
    best_e0 = dict(zip(best["standards"], best["delta_e0_eV"]))
    X = _shifted_matrix(k_exp, library, best["standards"], best_e0, e0_sign, 0)
    valid = np.isfinite(chi_exp) & np.all(np.isfinite(X), axis=1)
    fit_chi = X @ np.array(best["weights"])
    resid = chi_exp - fit_chi
    kv = k_exp[valid]
    curve = np.column_stack([
        kv,
        chi_exp[valid],
        fit_chi[valid],
        resid[valid],
        kv**2 * chi_exp[valid],
        kv**2 * fit_chi[valid],
        kv**2 * resid[valid],
    ])
    best_path = out / "best_lcf_fit.dat"
    np.savetxt(
        best_path,
        curve,
        header=(
            "k experimental_chi fitted_chi residual_chi "
            "experimental_k2chi fitted_k2chi residual_k2chi\n"
            f"Delta E0 and LCF optimizations used k^{k_weight}-weighted chi(k).\n"
            f"e0_mode: {e0_mode}\n"
            f"best standards: {' | '.join(best['standards'])}\n"
            f"weights: {' | '.join(f'{w:.10g}' for w in best['weights'])}\n"
            f"weight_err: {' | '.join(f'{w:.10g}' for w in best['weight_err'])}\n"
            f"delta_e0_eV: {' | '.join(f'{e:.10g}' for e in best['delta_e0_eV'])}\n"
            f"delta_e0_err_eV: {' | '.join(f'{e:.10g}' for e in best['delta_e0_err_eV'])}\n"
            f"delta_e0_initial_eV: "
            f"{' | '.join(f'{e:.10g}' for e in best['delta_e0_initial_eV'])}"
        ),
    )

    plot_path = out / "best_lcf_fit.png"
    plot_r_path = out / "best_lcf_fit_R.png"
    _plot_best_fit(kv, chi_exp[valid], fit_chi[valid], plot_path, plot_r_path)

    top_fits_path = write_top_fits([out], out / "top_fits.csv")
    top_candidates_path = write_top_candidates(
        [out], out / f"top_{n_top}_candidates.csv", n=n_top)

    log_path = out / "fit.log"
    log()
    log(f"Ran {len(ranked)} LCF fits.")
    log(f"Best fit ({metric} = {best[metric]:.6g}, e0_mode = {e0_mode}):")
    for s, w, e, e1 in zip(best["standards"], best["weights"],
                           best["delta_e0_eV"], best["delta_e0_initial_eV"]):
        log(f"  {w:8.4f}  {s}  (deltaE0 = {e:.4f} eV, step 1: {e1:.4f} eV)")
    for path in (results_path, delta_e0_path, best_path, plot_path, plot_r_path,
                 top_fits_path, top_candidates_path, log_path):
        log(f"Wrote: {path}")

    result = {
        "best": best,
        "top": ranked[:n_top],
        "delta_e0": e0_rows,
        "e0_mode": e0_mode,
        "n_standards": len(library),
        "n_fits": len(ranked),
        "outdir": str(out),
        "results_file": str(results_path),
        "delta_e0_file": str(delta_e0_path),
        "best_fit_file": str(best_path),
        "plot_file": str(plot_path),
        "plot_r_file": str(plot_r_path),
        "top_fits_file": top_fits_path,
        "top_candidates_file": top_candidates_path,
        "log_file": str(log_path),
    }
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n\nReturned result:\n")
        f.write(json.dumps(result, indent=2) + "\n")
    return result

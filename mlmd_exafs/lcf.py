"""Linear combination fitting (LCF) of simulated EXAFS spectra to experiment.

Given an experimental chi(k) and a library of simulated chi(k) "standards"
(e.g. ``<savefile>-chi_avg.dat`` files from several candidate structures), the
fit proceeds in two steps:

1. For every standard, determine an individual delta E0 by shifting it against
   the experimental spectrum and minimizing the selected metric on
   k^n-weighted chi(k).
2. For every combination of 1..``max_components`` standards, fit the relative
   weights of the E0-shifted standards (weights >= 0, sum(weights) = 1; a
   single standard has weight 1). Combinations are ranked by the metric.

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
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .fitting_E0 import read_experimental

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
):
    """Fit weights for a fixed combination of standards.

    One standard: weight fixed to 1. Two or more: weights >= 0 and
    sum(weights) = 1 (SLSQP). Returns None if too few valid k points overlap.
    """
    from scipy.optimize import minimize

    metric = canonical_metric(metric)
    n = len(combo)

    y_all = apply_k_weight(k_exp, chi_exp, kweight)
    X_all = _shifted_matrix(k_exp, standards, combo, delta_e0_by_name, e0_sign, kweight)

    valid = np.isfinite(y_all) & np.all(np.isfinite(X_all), axis=1)
    min_valid = max(5, int(np.ceil(min_valid_frac * k_exp.size)))
    if valid.sum() < min_valid:
        return None

    y = y_all[valid]
    X = X_all[valid, :]

    # E0 shifts were optimized first (n), then n-1 independent fractions.
    n_params = n + max(n - 1, 0)

    if n == 1:
        weights = np.array([1.0])
        return {
            "combo": combo,
            "weights": weights,
            "metrics": calculate_metrics(y, X[:, 0], n_params=n_params),
            "success": True,
            "message": "single standard; weight fixed to 1",
        }

    x0 = np.full(n, 1.0 / n)
    result = minimize(
        lambda w: metric_score(y, X @ w, metric, n_params=n_params),
        x0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    weights = np.clip(result.x, 0.0, 1.0)
    weights = weights / np.sum(weights) if np.sum(weights) > 0 else x0

    return {
        "combo": combo,
        "weights": weights,
        "metrics": calculate_metrics(y, X @ weights, n_params=n_params),
        "success": bool(result.success),
        "message": str(result.message),
    }


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
        Per-standard delta E0 search bounds (eV).
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

    Writes ``lcf_results.csv`` (all fits, ranked), ``delta_e0.csv``,
    ``best_lcf_fit.dat`` and ``best_lcf_fit.png`` into ``outdir``.

    Returns
    -------
    dict
        ``best`` (standards, weights, delta_e0_eV, metrics), ``top`` (list of
        the ``n_top`` best fits), ``delta_e0`` (per standard), ``n_standards``,
        ``n_fits``, and output file paths.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric = canonical_metric(metric)
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

    if verbose:
        print(f"Experimental file: {exp_file}")
        print(f"Standards loaded:  {len(library)}")
        print(f"k range:           {k_exp.min():.4f} to {k_exp.max():.4f}")
        print(f"k-weight:          {k_weight}")
        print(f"Optimization:      {metric}")
        print()
        print("Determining delta E0 for each standard...")

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
        if verbose:
            status = "ok" if success else "check"
            print(f"{name:45s} deltaE0 = {de0:9.4f} eV   {metric} = {score:.6g}   {status}")
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
            "delta_e0_eV": [delta_e0_by_name[n] for n in r["combo"]],
            "optimized_metric": metric,
            **{m: float(r["metrics"][m]) for m in METRICS},
            "n_points": r["metrics"]["n_points"],
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
                  *METRICS, "n_points", "success", "message",
                  "standards", "weights", "delta_e0_eV"]
        for i in range(1, max_components + 1):
            header += [f"std_{i}", f"weight_{i}", f"delta_e0_eV_{i}"]
        writer.writerow(header)
        for row in ranked:
            line = [row["rank"], row["n_components"], metric, row[metric],
                    *(row[m] for m in METRICS), row["n_points"], row["success"],
                    row["message"],
                    " | ".join(row["standards"]),
                    " | ".join(f"{w:.10g}" for w in row["weights"]),
                    " | ".join(f"{e:.10g}" for e in row["delta_e0_eV"])]
            for i in range(max_components):
                if i < row["n_components"]:
                    line += [row["standards"][i], row["weights"][i], row["delta_e0_eV"][i]]
                else:
                    line += ["", "", ""]
            writer.writerow(line)

    # Per-standard delta E0
    delta_e0_path = out / "delta_e0.csv"
    with open(delta_e0_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(e0_rows[0]))
        writer.writeheader()
        writer.writerows(e0_rows)

    # Best-fit curve (unweighted chi, plus k^2-weighted columns)
    X = _shifted_matrix(k_exp, library, best["standards"], delta_e0_by_name, e0_sign, 0)
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
            f"best standards: {' | '.join(best['standards'])}\n"
            f"weights: {' | '.join(f'{w:.10g}' for w in best['weights'])}\n"
            f"delta_e0_eV: {' | '.join(f'{e:.10g}' for e in best['delta_e0_eV'])}"
        ),
    )

    plot_path = out / "best_lcf_fit.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(kv, kv**2 * chi_exp[valid], linestyle="none", marker="o", markersize=3.5,
            color="black", label="experiment")
    label = " + ".join(
        f"{w:.2f} {Path(s).stem}" for s, w in zip(best["standards"], best["weights"])
    )
    ax.plot(kv, kv**2 * fit_chi[valid], color="magenta", lw=2, label=f"LCF: {label}")
    ax.axhline(0, color="0.6", lw=0.6, zorder=0)
    ax.set_xlabel(r"$k$ [$\AA^{-1}$]")
    ax.set_ylabel(r"$k^2\chi(k)$")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)

    if verbose:
        print()
        print(f"Ran {len(ranked)} LCF fits.")
        print(f"Best fit ({metric} = {best[metric]:.6g}):")
        for s, w, e in zip(best["standards"], best["weights"], best["delta_e0_eV"]):
            print(f"  {w:8.4f}  {s}  (deltaE0 = {e:.4f} eV)")
        for path in (results_path, delta_e0_path, best_path, plot_path):
            print(f"Wrote: {path}")

    return {
        "best": best,
        "top": ranked[:n_top],
        "delta_e0": e0_rows,
        "n_standards": len(library),
        "n_fits": len(ranked),
        "outdir": str(out),
        "results_file": str(results_path),
        "delta_e0_file": str(delta_e0_path),
        "best_fit_file": str(best_path),
        "plot_file": str(plot_path),
    }

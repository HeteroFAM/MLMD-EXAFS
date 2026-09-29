#!/usr/bin/env python
# coding: utf-8
"""Determine the E0 shift that best aligns simulated chi(k) with experiment.

The simulated averaged chi(k) (``<savefile>-chi_avg.dat`` from
:func:`mlmd_exafs.analysis.average_chi`) is shifted in energy,
k'^2 = k^2 + E0 / K2EV, and E0 is chosen to minimize the mean squared
deviation of k^2*chi against the experimental spectrum over [kmin, kmax].

Experimental data may be whitespace-delimited (``.dat``) or comma-separated
(``.csv``); the first two numeric columns are read as k and chi (or k^2*chi).

Used by ``mlmd-exafs average --exp-file`` and ``mlmd-exafs fit-e0``, and
available standalone as ``python -m mlmd_exafs.fitting_E0``.
"""

import os
import re
import argparse

import numpy as np


# ============================================================
# Defaults
# ============================================================

DEFAULT_OUTDIR = "aimd_exafs"

# Supported experimental file extensions -> column delimiter regex
EXP_FILE_DELIMITERS = {
    ".dat": r"\s+",
    ".csv": r"\s*,\s*",
}

# E0 search range in eV
E0_MIN_DEFAULT = -10.0
E0_MAX_DEFAULT = 10.0
E0_GRID_N_DEFAULT = 1201

# k fitting range
DEFAULT_KMIN = 2.0
DEFAULT_KMAX = 12.0

# hbar^2 / 2me in eV Angstrom^2
K2EV_DEFAULT = 3.81


# ============================================================
# Command-line input
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Determine E0 shift using averaged FEFF chi(k), usually <savefile>-chi_avg.dat."
        )
    )

    parser.add_argument(
        "--sim-file",
        required=True,
        help="Simulation averaged chi(k) file, e.g. exafs-chi_avg.dat.",
    )

    parser.add_argument(
        "--exp-file",
        required=True,
        help="Experimental data file (.dat whitespace- or .csv comma-delimited).",
    )

    parser.add_argument(
        "--outdir",
        default=DEFAULT_OUTDIR,
        help="Output folder. Default: aimd_exafs",
    )

    parser.add_argument(
        "--kmin",
        type=float,
        default=DEFAULT_KMIN,
        help="Minimum k value used for E0 fitting and plot. Default: 2.0",
    )

    parser.add_argument(
        "--kmax",
        type=float,
        default=DEFAULT_KMAX,
        help="Maximum k value used for E0 fitting and plot. Default: 12.0",
    )

    parser.add_argument(
        "--exp-col2-is-k2chi",
        action="store_true",
        help=(
            "Use this flag if column 2 of experimental file is already k^2*chi. "
            "Without this flag, column 2 is treated as chi."
        ),
    )

    parser.add_argument(
        "--sim-col2-is-k2chi",
        action="store_true",
        help=(
            "Use this only if column 2 of the simulation file is already k^2*chi. "
            "Default assumes column 2 is chi, which is correct for chi_avg.dat "
            "from the averaging script."
        ),
    )

    parser.add_argument(
        "--e0-min",
        type=float,
        default=E0_MIN_DEFAULT,
        help="Minimum E0 search value in eV. Default: -10.0",
    )

    parser.add_argument(
        "--e0-max",
        type=float,
        default=E0_MAX_DEFAULT,
        help="Maximum E0 search value in eV. Default: 10.0",
    )

    parser.add_argument(
        "--e0-grid-n",
        type=int,
        default=E0_GRID_N_DEFAULT,
        help="Number of E0 grid points. Default: 1201",
    )

    parser.add_argument(
        "--k2ev",
        type=float,
        default=K2EV_DEFAULT,
        help="hbar^2 / 2me in eV Angstrom^2. Default: 3.81",
    )

    return parser.parse_args()


# ============================================================
# Data reader
# ============================================================

def read_two_column_numeric(filename, delimiter=r"\s+"):
    """
    Read first two numeric columns from a file.

    Header lines are skipped automatically.
    Lines starting with # are ignored.
    ``delimiter`` is a regex used to split each line into columns.
    """

    xvals = []
    yvals = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            parts = re.split(delimiter, line)

            if len(parts) < 2:
                continue

            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue

            xvals.append(x)
            yvals.append(y)

    if len(xvals) == 0:
        raise RuntimeError(f"No valid numeric data found in {filename}")

    xvals = np.array(xvals, dtype=float)
    yvals = np.array(yvals, dtype=float)

    valid = np.isfinite(xvals) & np.isfinite(yvals)

    xvals = xvals[valid]
    yvals = yvals[valid]

    order = np.argsort(xvals)

    return xvals[order], yvals[order]


def read_experimental(filename):
    """
    Read experimental k and column-2 data from a .dat or .csv file.
    """

    ext = os.path.splitext(filename)[1].lower()

    if ext not in EXP_FILE_DELIMITERS:
        raise ValueError(
            f"Unsupported experimental file type {ext!r} for {filename}. "
            f"Use one of: {', '.join(sorted(EXP_FILE_DELIMITERS))}"
        )

    return read_two_column_numeric(filename, delimiter=EXP_FILE_DELIMITERS[ext])


# ============================================================
# Minimization functions
# ============================================================

def golden_section_minimize(func, a, b, tol=1.0e-6, maxiter=200):
    """
    Minimize func on [a, b] using golden-section search.
    """

    gr = (np.sqrt(5.0) - 1.0) / 2.0

    c = b - gr * (b - a)
    d = a + gr * (b - a)

    fc = func(c)
    fd = func(d)

    for _ in range(maxiter):
        if abs(b - a) < tol:
            break

        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = func(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = func(d)

    xbest = 0.5 * (a + b)
    fbest = func(xbest)

    return xbest, fbest


def grid_then_refine_minimize(func, xmin, xmax, ngrid=1201):
    """
    First do a grid search, then refine near the best grid point.
    """

    grid = np.linspace(xmin, xmax, ngrid)
    vals = np.array([func(x) for x in grid])

    if not np.any(np.isfinite(vals)):
        raise RuntimeError("All E0 RSMD values are non-finite.")

    ibest = np.nanargmin(vals)

    if ibest == 0:
        a = grid[0]
        b = grid[1]
    elif ibest == ngrid - 1:
        a = grid[-2]
        b = grid[-1]
    else:
        a = grid[ibest - 1]
        b = grid[ibest + 1]

    xbest, fbest = golden_section_minimize(func, a, b)

    return xbest, fbest


# ============================================================
# Interpolation and RSMD calculation
# ============================================================

def forecast_and_rsmd(
    exp_k,
    exp_k2chi,
    sim_kprime,
    sim_kprime2chi,
    kmin,
    kmax,
):
    """
    Interpolates simulation k'^2 chi'(k') onto experimental k values.

    Only experimental k values between kmin and kmax are used.

    RSMD here is:

        mean((exp_k2chi - interpolated_sim_k2chi)^2)

    No square root is applied.
    """

    order = np.argsort(sim_kprime)

    sim_kprime = sim_kprime[order]
    sim_kprime2chi = sim_kprime2chi[order]

    base_mask = (
        np.isfinite(exp_k)
        & np.isfinite(exp_k2chi)
        & (exp_k >= kmin)
        & (exp_k <= kmax)
    )

    x_exp_all = exp_k[base_mask]
    y_exp_all = exp_k2chi[base_mask]

    if len(x_exp_all) < 1:
        return np.inf, None

    lower_indices = np.searchsorted(sim_kprime, x_exp_all, side="right") - 1

    valid = (
        (lower_indices >= 0)
        & (lower_indices + 1 < len(sim_kprime))
    )

    if np.count_nonzero(valid) < 1:
        return np.inf, None

    x_exp = x_exp_all[valid]
    y_exp = y_exp_all[valid]
    lower_indices = lower_indices[valid]

    x1 = sim_kprime[lower_indices]
    x2 = sim_kprime[lower_indices + 1]

    y1 = sim_kprime2chi[lower_indices]
    y2 = sim_kprime2chi[lower_indices + 1]

    denom = x2 - x1

    nonzero = denom != 0.0

    if np.count_nonzero(nonzero) < 1:
        return np.inf, None

    x_exp = x_exp[nonzero]
    y_exp = y_exp[nonzero]
    x1 = x1[nonzero]
    x2 = x2[nonzero]
    y1 = y1[nonzero]
    y2 = y2[nonzero]

    omega = y1 + (x_exp - x1) * (y2 - y1) / (x2 - x1)

    error_sq = (y_exp - omega) ** 2

    positive = np.isfinite(error_sq) & (error_sq > 0.0)

    if np.count_nonzero(positive) < 1:
        return np.inf, None

    rsmd = np.mean(error_sq[positive])

    table = {
        "exp_k": x_exp,
        "exp_k2chi": y_exp,
        "omega_interp": omega,
        "squared_error": error_sq,
    }

    return rsmd, table


# ============================================================
# Main workflow
# ============================================================

def fit_e0(
    sim_file,
    exp_file,
    outdir=DEFAULT_OUTDIR,
    kmin=DEFAULT_KMIN,
    kmax=DEFAULT_KMAX,
    exp_col2_is_k2chi=False,
    sim_col2_is_k2chi=False,
    e0_min=E0_MIN_DEFAULT,
    e0_max=E0_MAX_DEFAULT,
    e0_grid_n=E0_GRID_N_DEFAULT,
    k2ev=K2EV_DEFAULT,
):
    """
    Fit the E0 shift of simulated chi(k) against an experimental spectrum.

    Writes avg_chi_E0_shifted.dat, best_E0_comparison.dat,
    E0_fit_summary.txt and calc_vs_exp.png into ``outdir``.

    Returns a dict with ``best_E0``, ``min_rsmd`` and the written file paths.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sim_file = os.path.abspath(sim_file)
    exp_file = os.path.abspath(exp_file)
    outdir = os.path.abspath(outdir)

    if kmin >= kmax:
        raise ValueError("--kmin must be smaller than --kmax")

    if e0_min >= e0_max:
        raise ValueError("--e0-min must be smaller than --e0-max")

    os.makedirs(outdir, exist_ok=True)

    print("============================================================")
    print("E0 determination from averaged chi(k)")
    print("============================================================")
    print(f"Simulation file: {sim_file}")
    print(f"Experimental file: {exp_file}")
    print(f"Output folder: {outdir}")
    print(f"E0 search range: {e0_min} to {e0_max} eV")
    print(f"k fitting range: {kmin} to {kmax} Å^-1")
    print()

    # ============================================================
    # Read simulation averaged chi(k)
    # ============================================================

    if not os.path.exists(sim_file):
        raise FileNotFoundError(f"Simulation file not found: {sim_file}")

    sim_k, sim_col2 = read_two_column_numeric(sim_file)

    if len(sim_k) < 5:
        raise RuntimeError(f"Not enough valid simulation data points in {sim_file}")

    if sim_col2_is_k2chi:
        valid_nonzero = sim_k != 0.0
        sim_k = sim_k[valid_nonzero]
        sim_col2 = sim_col2[valid_nonzero]

        sim_chi = sim_col2 / sim_k**2
        print("Simulation column 2 treated as k^2*chi and converted to chi.")
    else:
        sim_chi = sim_col2
        print("Simulation column 2 treated as chi.")
        print("This is correct for chi_avg.dat from mlmd-exafs average.")

    # ============================================================
    # Read experimental data
    # ============================================================

    if not os.path.exists(exp_file):
        raise FileNotFoundError(f"Experimental file not found: {exp_file}")

    exp_k, exp_col2 = read_experimental(exp_file)

    if len(exp_k) < 5:
        raise RuntimeError(f"Not enough valid experimental data points in {exp_file}")

    if exp_col2_is_k2chi:
        exp_k2chi = exp_col2
        print("Experimental column 2 treated as k^2*chi.")
    else:
        exp_k2chi = exp_k**2 * exp_col2
        print("Experimental column 2 treated as chi and converted to k^2*chi.")

    # Check that experimental data exists in selected k range
    exp_range_mask = (exp_k >= kmin) & (exp_k <= kmax)

    if np.count_nonzero(exp_range_mask) < 2:
        raise RuntimeError(
            f"Not enough experimental data points between "
            f"{kmin} and {kmax} Å^-1."
        )

    # ============================================================
    # Define RSMD as a function of E0
    # ============================================================

    def rsmd_for_E0(E0):
        sim_kprime_sq = (sim_k**2 * k2ev + E0) / k2ev

        valid = sim_kprime_sq > 0.0

        if np.count_nonzero(valid) < 2:
            return np.inf

        sim_kprime = np.sqrt(sim_kprime_sq[valid])
        sim_chi_valid = sim_chi[valid]

        sim_kprime2chi = sim_kprime**2 * sim_chi_valid

        rsmd, _ = forecast_and_rsmd(
            exp_k=exp_k,
            exp_k2chi=exp_k2chi,
            sim_kprime=sim_kprime,
            sim_kprime2chi=sim_kprime2chi,
            kmin=kmin,
            kmax=kmax,
        )

        return rsmd

    # ============================================================
    # Optimize E0
    # ============================================================

    best_E0, best_rsmd = grid_then_refine_minimize(
        rsmd_for_E0,
        e0_min,
        e0_max,
        ngrid=e0_grid_n,
    )

    print()
    print(f"Best E0 shift: {best_E0:.6f} eV")
    print(f"Minimum RSMD: {best_rsmd:.6e}")

    # ============================================================
    # Apply best E0 shift
    # ============================================================

    sim_kprime_sq_best = (sim_k**2 * k2ev + best_E0) / k2ev
    valid_best = sim_kprime_sq_best > 0.0

    sim_kprime_best = np.sqrt(sim_kprime_sq_best[valid_best])
    sim_chi_best = sim_chi[valid_best]
    sim_kprime2chi_best = sim_kprime_best**2 * sim_chi_best

    order_best = np.argsort(sim_kprime_best)

    sim_kprime_best = sim_kprime_best[order_best]
    sim_chi_best = sim_chi_best[order_best]
    sim_kprime2chi_best = sim_kprime2chi_best[order_best]

    # ============================================================
    # Build comparison table
    # ============================================================

    best_rsmd_check, best_table = forecast_and_rsmd(
        exp_k=exp_k,
        exp_k2chi=exp_k2chi,
        sim_kprime=sim_kprime_best,
        sim_kprime2chi=sim_kprime2chi_best,
        kmin=kmin,
        kmax=kmax,
    )

    if best_table is None:
        raise RuntimeError("Could not build comparison table for best E0.")

    # ============================================================
    # Save shifted simulation
    # ============================================================

    shifted_sim_path = os.path.join(outdir, "avg_chi_E0_shifted.dat")

    shifted_sim_data = np.column_stack([
        sim_kprime_best,
        sim_chi_best,
        sim_kprime2chi_best,
    ])

    np.savetxt(
        shifted_sim_path,
        shifted_sim_data,
        header="k chi k2_chi",
        comments="# ",
    )

    # ============================================================
    # Save comparison table
    # ============================================================

    comparison_path = os.path.join(outdir, "best_E0_comparison.dat")

    comparison_data = np.column_stack([
        best_table["exp_k"],
        best_table["exp_k2chi"],
        best_table["omega_interp"],
        best_table["squared_error"],
    ])

    np.savetxt(
        comparison_path,
        comparison_data,
        header="exp_k exp_k2chi sim_interp_k2chi squared_error",
        comments="# ",
    )

    # ============================================================
    # Save summary
    # ============================================================

    summary_path = os.path.join(outdir, "E0_fit_summary.txt")

    with open(summary_path, "w") as f:
        f.write("E0 fitting summary\n")
        f.write(f"SIM_FILE = {sim_file}\n")
        f.write(f"EXP_FILE = {exp_file}\n")
        f.write(f"EXP_COL2_IS_K2CHI = {exp_col2_is_k2chi}\n")
        f.write(f"SIM_COL2_IS_K2CHI = {sim_col2_is_k2chi}\n")
        f.write(f"KMIN = {kmin:.6f}\n")
        f.write(f"KMAX = {kmax:.6f}\n")
        f.write(f"E0_MIN = {e0_min:.6f} eV\n")
        f.write(f"E0_MAX = {e0_max:.6f} eV\n")
        f.write(f"E0_GRID_N = {e0_grid_n}\n")
        f.write(f"Best_E0 = {best_E0:.10f} eV\n")
        f.write(f"Minimum_RSMD = {best_rsmd:.10e}\n")
        f.write(f"Check_RSMD_from_saved_table = {best_rsmd_check:.10e}\n")
        f.write(f"K2EV = {k2ev:.8f} eV Angstrom^2\n")
        f.write("\n")
        f.write("RSMD definition:\n")
        f.write("    RSMD = mean((exp_k2chi - interpolated_sim_k2chi)^2)\n")
        f.write("No square root is applied.\n")

    # ============================================================
    # Plot experiment and shifted simulation
    # ============================================================

    plot_path = os.path.join(outdir, "calc_vs_exp.png")

    plt.figure(figsize=(7, 5))

    plt.plot(
        exp_k,
        exp_k2chi,
        linestyle="none",
        marker="o",
        markersize=3.5,
        markerfacecolor="black",
        markeredgecolor="black",
        label="experiment",
    )

    plt.plot(
        sim_kprime_best,
        sim_kprime2chi_best,
        color="blue",
        lw=1.5,
        label=f"{os.path.basename(sim_file)}, E0 = {best_E0:.3f} eV",
    )

    plt.xlim(kmin, kmax)

    plt.xlabel(r"$k$ [$\AA^{-1}$]")
    plt.ylabel(r"$k^2\chi(k)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # ============================================================
    # Done
    # ============================================================

    print()
    print("Done.")
    print(f"k fitting range: {kmin} to {kmax} Å^-1")
    print(f"Best E0: {best_E0:.6f} eV")
    print(f"Minimum RSMD: {best_rsmd:.6e}")
    print(f"Wrote: {shifted_sim_path}")
    print(f"Wrote: {comparison_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {plot_path}")

    return {
        "best_E0": float(best_E0),
        "min_rsmd": float(best_rsmd),
        "kmin": kmin,
        "kmax": kmax,
        "outdir": outdir,
        "shifted_sim_file": shifted_sim_path,
        "comparison_file": comparison_path,
        "summary_file": summary_path,
        "plot_file": plot_path,
    }


def main():
    """
    python -m mlmd_exafs.fitting_E0 \\
  --sim-file exafs-chi_avg.dat \\
  --exp-file exp_k.dat \\
  --kmin 2.0 \\
  --kmax 12.0 \\
  --exp-col2-is-k2chi
    """

    args = parse_arguments()

    fit_e0(
        sim_file=args.sim_file,
        exp_file=args.exp_file,
        outdir=args.outdir,
        kmin=args.kmin,
        kmax=args.kmax,
        exp_col2_is_k2chi=args.exp_col2_is_k2chi,
        sim_col2_is_k2chi=args.sim_col2_is_k2chi,
        e0_min=args.e0_min,
        e0_max=args.e0_max,
        e0_grid_n=args.e0_grid_n,
        k2ev=args.k2ev,
    )


if __name__ == "__main__":
    main()

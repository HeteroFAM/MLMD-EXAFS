"""Configurational averaging, Fourier transform, and plotting of chi(k).

After FEFF runs on every sampled snapshot, ``average_chi`` computes the mean
chi(k) across snapshots (with per-k sampling spread), ``xftf`` transforms
k-space to R-space with a Hanning window, and the plotting helpers render the
k^n*chi(k) spectrum plus k- and R-space convergence panels.

The spread band reported here is the snapshot-to-snapshot statistical
dispersion of the MD ensemble (a sampling band) — it reflects how converged
the average is, not a systematic force-field error bar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

CHI_HEADER = "#       k          chi          mag           phase @#\n"


# ---------------------------------------------------------------------------
# chi.dat reading + averaging
# ---------------------------------------------------------------------------


def _extract_chi(chi_path: Path) -> tuple[list[float], list[float]] | None:
    """Read k and chi columns from a FEFF chi.dat file, or None if malformed."""
    with open(chi_path) as f:
        lines = f.readlines()
    if CHI_HEADER not in lines:
        return None
    index = lines.index(CHI_HEADER)
    rows = [
        [tok for tok in line.split(" ") if tok != ""] for line in lines[index + 1:]
    ]
    k = [float(r[0]) for r in rows]
    chi = [float(r[1]) for r in rows]
    if k and k[0] == 0.05:
        chi = [np.nan] + chi
        k = [0.0] + k
    return k, chi


def average_chi(directory: str, savefile: str) -> dict[str, Any]:
    """Average all chi.dat files in a FEFF output directory.

    Scans each subdirectory of ``directory`` for a ``chi.dat``, reads its k and
    chi columns, and computes the mean chi(k) plus the per-k standard deviation
    and standard error of the mean across snapshots.

    Writes ``<savefile>-chi_avg.dat`` (columns: k chi_avg chi_std chi_sem).

    Returns
    -------
    dict
        ``k``, ``chi_avg``, ``chi_std``, ``chi_sem``, ``n_samples``,
        ``skipped``, ``output_file``.
    """
    feff_dir = Path(directory)
    k_all: list[list[float]] = []
    chi_all: list[list[float]] = []
    skipped = 0

    for sample_dir in sorted(feff_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        chi_path = sample_dir / "chi.dat"
        if not chi_path.is_file():
            skipped += 1
            continue
        parsed = _extract_chi(chi_path)
        if parsed is None:
            skipped += 1
            continue
        k, chi = parsed
        k_all.append(k)
        chi_all.append(chi)

    if not chi_all:
        return {
            "k": np.array([]),
            "chi_avg": np.array([]),
            "chi_std": np.array([]),
            "chi_sem": np.array([]),
            "n_samples": 0,
            "skipped": skipped,
            "output_file": "",
        }

    k_arr = np.array(k_all)
    chi_arr = np.array(chi_all)

    # sanity: all snapshots should share the same k grid
    if np.around(k_arr.std(axis=0).sum()) != 0:
        print("Warning: k grids differ across snapshots.")

    chi_mean = np.nanmean(chi_arr, axis=0)
    chi_std = np.nanstd(chi_arr, axis=0)
    n_valid = np.sum(~np.isnan(chi_arr), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        chi_sem = chi_std / np.sqrt(n_valid)
    chi_sem[n_valid == 0] = np.nan

    paired = np.vstack((k_arr[0], chi_mean, chi_std, chi_sem)).T
    output_file = f"{savefile}-chi_avg.dat"
    with open(output_file, "w") as f:
        f.write("# k chi_avg chi_std chi_sem\n")
        f.writelines([" ".join(row) + "\n" for row in paired.astype(str)])

    print(f"Averaged {len(chi_all)} snapshots ({skipped} skipped) -> {output_file}")
    return {
        "k": k_arr[0],
        "chi_avg": chi_mean,
        "chi_std": chi_std,
        "chi_sem": chi_sem,
        "n_samples": len(chi_all),
        "skipped": skipped,
        "output_file": output_file,
    }


# ---------------------------------------------------------------------------
# Fourier transform k-space -> R-space
# ---------------------------------------------------------------------------


def hanning_window(k, kmin: float, kmax: float, dk_win: float = 1.0):
    """Hanning (cosine-bell) window for the EXAFS Fourier transform.

    Rises 0->1 over [kmin-dk/2, kmin+dk/2], flat at 1 to kmax-dk/2, falls
    1->0 over [kmax-dk/2, kmax+dk/2].
    """
    k = np.asarray(k, dtype=float)
    w = np.zeros_like(k)
    lo_rise, hi_rise = kmin - dk_win / 2, kmin + dk_win / 2
    lo_fall, hi_fall = kmax - dk_win / 2, kmax + dk_win / 2
    for i, ki in enumerate(k):
        if hi_rise <= ki <= lo_fall:
            w[i] = 1.0
        elif lo_rise < ki < hi_rise:
            w[i] = 0.5 * (1 - np.cos(np.pi * (ki - lo_rise) / dk_win))
        elif lo_fall < ki < hi_fall:
            w[i] = 0.5 * (1 + np.cos(np.pi * (ki - lo_fall) / dk_win))
    return w


def xftf(
    k,
    chi,
    kmin: float = 2.0,
    kmax: float = 11.0,
    dk_win: float = 1.0,
    kweight: int = 2,
    nfft: int = 2048,
):
    """Forward EXAFS Fourier transform (k-space -> R-space).

    Computes ``chi_tilde(R) = (dk/sqrt(pi)) sum_j k^n chi(k) W(k) e^{2ikR}``
    with a Hanning window and a zero-padded FFT.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(r, |chi_tilde(R)|)``.
    """
    k = np.asarray(k, dtype=float)
    chi = np.where(np.isnan(chi), 0.0, np.asarray(chi, dtype=float))
    dk = k[1] - k[0]
    win = hanning_window(k, kmin, kmax, dk_win)
    chi_kw = k**kweight * chi * win
    fx = np.zeros(nfft, dtype=complex)
    fx[: len(chi_kw)] = chi_kw
    xft = (dk / np.sqrt(np.pi)) * np.fft.fft(fx)
    rstep = np.pi / (dk * nfft)
    r = rstep * np.arange(nfft // 2)
    return r, np.abs(xft[: nfft // 2])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_chi(
    chi_file: str,
    savefile: str,
    k_weight: int = 2,
    band: str = "sem",
    n_sigma: float = 1.0,
    n_samples: int | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Plot k-weighted chi(k) with the MD sampling band shaded around the mean.

    Reads the ``<savefile>-chi_avg.dat`` produced by :func:`average_chi`
    (columns k chi_avg chi_std chi_sem) and writes ``<savefile>.png``.

    Parameters
    ----------
    chi_file : str
        Averaged chi file from :func:`average_chi`.
    savefile : str
        Base path for the output PNG.
    k_weight : int
        k-weighting exponent n in k^n*chi(k) (1, 2, or 3).
    band : {"sem", "std", "none"}
        Which spread to shade: standard error of the mean, snapshot standard
        deviation, or no band. This is a sampling band, not a model-error bar.
    n_sigma : float
        Band half-width in multiples of the chosen spread.
    n_samples : int, optional
        Frame count, annotated on the plot title when given.
    title : str, optional
        Override the plot title.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if band not in ("sem", "std", "none"):
        raise ValueError("band must be 'sem', 'std', or 'none'")

    data = np.loadtxt(chi_file, comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{chi_file} needs >=2 columns to be a chi file.")

    k = data[:, 0]
    chi = data[:, 1]
    chi_std = data[:, 2] if data.shape[1] > 2 else None
    chi_sem = data[:, 3] if data.shape[1] > 3 else None

    kw = k**k_weight
    y = kw * chi

    spread, band_label = None, None
    if band == "sem" and chi_sem is not None:
        spread, band_label = kw * chi_sem, f"±{n_sigma:g} SEM"
    elif band == "std" and chi_std is not None:
        spread, band_label = kw * chi_std, f"±{n_sigma:g} SD (snapshot spread)"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if spread is not None:
        finite = np.isfinite(y) & np.isfinite(spread)
        ax.fill_between(
            k[finite],
            (y - n_sigma * spread)[finite],
            (y + n_sigma * spread)[finite],
            alpha=0.25,
            color="tab:blue",
            label=band_label,
            linewidth=0,
        )
    ax.plot(k, y, color="tab:blue", lw=1.5, label=r"$\langle\chi\rangle$")
    ax.set_xlabel(r"$k$ ($\AA^{-1}$)")
    ax.set_ylabel(rf"$k^{k_weight}\,\chi(k)$ ($\AA^{{-{k_weight}}}$)")
    if title is None:
        title = rf"EXAFS $k^{k_weight}\chi(k)$"
        if n_samples is not None:
            title += f" — {n_samples} frames"
    ax.set_title(title)
    ax.axhline(0.0, color="0.6", lw=0.6, zorder=0)
    if spread is not None:
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    output_file = f"{savefile}.png"
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_file}")
    return {"output_file": output_file, "k_weight": k_weight, "band": band}


def plot_convergence(
    directory: str,
    savefile: str,
    step: int = 10,
    k_weight: int = 2,
    kmin: float = 2.0,
    kmax: float = 11.0,
) -> dict[str, Any]:
    """Plot k-space and R-space convergence versus number of averaged snapshots.

    Reads all chi.dat files in ``directory`` and produces a two-panel figure:
    k^n*chi(k) (left) and |chi_tilde(R)| (right), each drawn for increasing
    snapshot counts so the averaging convergence is visible.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    feff_dir = Path(directory)
    k_all: list[list[float]] = []
    chi_all: list[list[float]] = []
    skipped = 0
    for sample_dir in sorted(feff_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        chi_path = sample_dir / "chi.dat"
        if not chi_path.is_file():
            skipped += 1
            continue
        parsed = _extract_chi(chi_path)
        if parsed is None:
            skipped += 1
            continue
        k, chi = parsed
        k_all.append(k)
        chi_all.append(chi)

    if not chi_all:
        raise ValueError(f"No usable chi.dat files found in {directory}.")

    k_arr = np.array(k_all)
    chi_arr = np.array(chi_all)
    n_finished = len(chi_all)
    kgrid = k_arr[0]

    cmap = mpl.colormaps.get_cmap("viridis_r")
    counts = list(range(step, n_finished + 1, step)) or [n_finished]
    colors = cmap(np.linspace(0, 1, len(counts)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for c, j in zip(colors, counts):
        mean_j = np.nanmean(chi_arr[:j], axis=0)
        axes[0].plot(
            kgrid, (kgrid**k_weight) * mean_j, lw=1, color=c, alpha=0.9, label=f"{j}"
        )
        r, mag = xftf(kgrid, mean_j, kmin=kmin, kmax=kmax, kweight=k_weight)
        axes[1].plot(r, mag, lw=1, color=c, alpha=0.9)

    axes[0].set_xlim(-0.1, kmax + 0.1)
    axes[0].set_xlabel(r"$k$ ($\AA^{-1}$)")
    axes[0].set_ylabel(rf"$k^{k_weight}\chi(k)$ ($\AA^{{-{k_weight}}}$)")
    axes[0].set_title("k-space convergence")
    axes[0].legend(title="snapshots", fontsize=8, ncol=2, frameon=False)

    axes[1].set_xlim(0, 6)
    axes[1].set_xlabel(r"$R$ ($\AA$)")
    axes[1].set_ylabel(r"$|\tilde\chi(R)|$ ($\AA^{-3}$)")
    axes[1].set_title("R-space convergence")

    fig.tight_layout()
    output_file = f"{savefile}.png"
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f"Averaged {n_finished} snapshots ({skipped} skipped) -> {output_file}")
    return {"output_file": output_file, "n_samples": n_finished, "skipped": skipped}

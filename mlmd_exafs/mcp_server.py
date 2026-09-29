"""Model Context Protocol (MCP) server for the MLMD-EXAFS workflow.

Exposes the MLIP-MD + FEFF EXAFS pipeline as MCP tools so any MCP-capable
orchestrator (Claude Code, Cline, Cursor, ...) can drive it natively. Each
pipeline stage is one MCP tool; the tool bodies call the same functions in the
``mlmd_exafs`` package that the CLI and the SciLink plug-in use.

Run it:

    mlmd-exafs-mcp            # stdio transport (default)

or register the command in your client's MCP config, e.g.:

    {
      "mcpServers": {
        "mlmd-exafs": { "command": "mlmd-exafs-mcp" }
      }
    }

Requires the MCP SDK: ``pip install "mlmd-exafs[mcp]"`` (installs ``mcp``).
FEFF9 is a separate binary; pass its path to ``run_feff`` via ``feff_bin``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# The server class was named FastMCP in mcp 1.x and renamed to MCPServer in
# mcp 2.x; both expose the same .tool() decorator and .run() entry point.
try:
    from mcp.server.fastmcp import FastMCP as _Server  # mcp 1.x
except ImportError:
    try:
        from mcp.server.mcpserver import MCPServer as _Server  # mcp 2.x
    except ImportError as exc:  # pragma: no cover - only hit without the extra
        raise ImportError(
            "The MCP server requires the 'mcp' package. Install it with "
            '`pip install "mlmd-exafs[mcp]"`.'
        ) from exc

from .analysis import average_chi, plot_chi, plot_convergence
from .calculators import build_calculator
from .cleanup_exafs import cleanup_exafs
from .feff import generate_feff_inputs_from_trajectory
from .fitting_E0 import fit_e0
from .lcf import run_lcf
from .md import relax, run_md
from .run_feff import DEFAULT_FEFF_BIN, run_feff_batch

mcp = _Server("mlmd-exafs")


@mcp.tool()
def mlmd_relax(
    structure_path: str,
    output: str,
    backend: str,
    device: str = "cpu",
    model: Optional[str] = None,
    head: str = "omat",
    modal: str = "mpa",
    fmax: float = 0.05,
) -> dict:
    """Relax a structure's atomic positions and cell with an MLIP.

    Required first step before molecular dynamics. Writes the relaxed structure
    to ``output``.

    Args:
        structure_path: ASE-readable input structure (CIF, extxyz, POSCAR, ...).
        output: Path to write the relaxed structure to.
        backend: MLIP backend (chgnet, mace, uma, orb, sevennet).
        device: cpu or cuda.
        model: Model name/checkpoint (backend default if omitted).
        head: UMA task head (oc20, omat, omol, odac, omc).
        modal: ORB / SevenNet dataset modality (mpa or omat24).
        fmax: Force convergence threshold in eV/A.
    """
    calc = build_calculator(backend, device=device, model=model, head=head, modal=modal)
    result = relax(structure_path, output, calc, fmax=fmax)
    result["status"] = "success"
    return result


@mcp.tool()
def mlmd_md(
    structure_path: str,
    directory: str,
    backend: str,
    device: str = "cpu",
    model: Optional[str] = None,
    head: str = "omat",
    modal: str = "mpa",
    temperature: float = 300.0,
    step_size: float = 10.0,
    n_steps: int = 11000,
) -> dict:
    """Run NVT molecular dynamics (Nose-Hoover chain) with an MLIP.

    Generates a thermally sampled extxyz trajectory for EXAFS. Point
    ``structure_path`` at the relaxed structure from ``mlmd_relax``.

    Args:
        structure_path: (Relaxed) structure file.
        directory: Output directory for the trajectory.
        backend: MLIP backend (chgnet, mace, uma, orb, sevennet).
        device: cpu or cuda.
        model: Model name/checkpoint.
        head: UMA task head.
        modal: ORB / SevenNet modality.
        temperature: Temperature in K.
        step_size: MD time step in atomic units (~0.02419 fs each).
        n_steps: Number of MD steps.
    """
    calc = build_calculator(backend, device=device, model=model, head=head, modal=modal)
    result = run_md(
        structure_path,
        calc,
        directory,
        temperature=temperature,
        step_size=step_size,
        n_steps=n_steps,
    )
    result["status"] = "success"
    return result


@mcp.tool()
def mlmd_feff_input(
    trajectory_path: str,
    target_atom: int,
    hole: int = 1,
    rmax: float = 6.0,
    scf: str = "6.0 0 30 0.2 1",
    s02: float = 1.0,
    control: str = "1 1 1 1 1 1",
    corrections: Optional[str] = None,
    step_size: int = 250,
    sampling_start: int = 0,
) -> dict:
    """Generate FEFF input files from an MD trajectory.

    Carves the local environment around the absorbing atom in sampled frames
    and writes a ``feff.inp`` per frame. Returns the output directory.

    Args:
        trajectory_path: MD trajectory (extxyz/traj).
        target_atom: Index of the absorbing atom.
        hole: HOLE card index (1=K, 2=L1, 3=L2, 4=L3).
        rmax: FEFF RMAX path cutoff in A (carve radius is rmax + 2.5 A).
        scf: SCF card parameters.
        s02: S0^2 amplitude reduction factor.
        control: CONTROL card.
        corrections: CORRECTIONS card "vrcorr vicorr", or None to omit.
        step_size: Sample every N-th frame.
        sampling_start: First frame index to sample (skip equilibration).
    """
    result = generate_feff_inputs_from_trajectory(
        trajectory_path=trajectory_path,
        target_atom=target_atom,
        hole=hole,
        rmax=rmax,
        scf=scf,
        s02=s02,
        control=control,
        corrections=corrections,
        step_size=step_size,
        sampling_start=sampling_start,
    )
    result["status"] = "success"
    return result


@mcp.tool()
def mlmd_run_feff(
    directory: str,
    feff_bin: str = DEFAULT_FEFF_BIN,
    max_workers: int = 32,
    cleanup: bool = True,
) -> dict:
    """Batch-execute the FEFF binary over generated input directories.

    Runs FEFF in every ``feff.inp`` subdirectory of ``directory`` with bounded
    concurrency, producing a ``chi.dat`` in each. Requires a FEFF9 executable.
    Afterwards FEFF scratch files are deleted, keeping only feff.inp,
    feff.out and chi.dat.

    Args:
        directory: FEFF output directory from ``mlmd_feff_input``.
        feff_bin: Path to the FEFF executable.
        max_workers: Maximum concurrent FEFF jobs.
        cleanup: Delete FEFF scratch files after the run (default True).
    """
    result = run_feff_batch(
        directory, feff_bin=feff_bin, max_workers=max_workers, cleanup=cleanup
    )
    result["status"] = "success" if result["n_failed"] == 0 else "partial"
    return result


@mcp.tool()
def mlmd_cleanup(
    root: str,
    dry_run: bool = False,
    remove_empty_dirs: bool = False,
) -> dict:
    """Delete FEFF scratch files from exafs_* directories.

    Recursively finds ``exafs_*`` directories under ``root`` and removes every
    file except feff.inp, feff.out, chi.dat and neighborhoods_*.xyz.
    ``mlmd_run_feff`` already does this by default; use this tool for older
    runs or runs made with the csh script.

    Args:
        root: Directory to search (may itself be an exafs_* directory).
        dry_run: Only report what would be deleted.
        remove_empty_dirs: Also remove directories left empty.
    """
    result = cleanup_exafs(root, dry_run=dry_run, remove_empty_dirs=remove_empty_dirs)
    result["status"] = "success" if result["exafs_dirs"] else "no_data"
    return result


@mcp.tool()
def mlmd_average_chi(
    directory: str,
    savefile: str,
    exp_file: Optional[str] = None,
    exp_col2_is_k2chi: bool = False,
    kmin: float = 2.0,
    kmax: float = 12.0,
) -> dict:
    """Average all chi.dat files into a converged chi(k) with a sampling band.

    Writes ``<savefile>-chi_avg.dat`` (columns k, chi_avg, chi_std, chi_sem).
    The std/sem columns are the MD sampling dispersion, not a force-field error
    bar. If an experimental spectrum is given, the E0 shift is also fitted
    (see ``mlmd_fit_e0``) into ``<savefile>_E0_fit/``.

    Args:
        directory: FEFF output directory containing chi.dat subdirectories.
        savefile: Base path for the averaged chi file.
        exp_file: Optional experimental chi(k) file (.dat or .csv).
        exp_col2_is_k2chi: Experimental column 2 is already k^2*chi.
        kmin: E0 fit kmin in A^-1.
        kmax: E0 fit kmax in A^-1.
    """
    result = average_chi(directory, savefile)
    # numpy arrays are not JSON-serializable; return only the scalar summary.
    for key in ("k", "chi_avg", "chi_std", "chi_sem"):
        result.pop(key, None)
    result["status"] = "success" if result["n_samples"] > 0 else "no_data"
    if exp_file and result["n_samples"] > 0:
        result["e0_fit"] = fit_e0(
            sim_file=result["output_file"],
            exp_file=exp_file,
            outdir=f"{savefile}_E0_fit",
            kmin=kmin,
            kmax=kmax,
            exp_col2_is_k2chi=exp_col2_is_k2chi,
        )
    return result


@mcp.tool()
def mlmd_fit_e0(
    chi_file: str,
    exp_file: str,
    outdir: Optional[str] = None,
    exp_col2_is_k2chi: bool = False,
    kmin: float = 2.0,
    kmax: float = 12.0,
    e0_min: float = -10.0,
    e0_max: float = 10.0,
) -> dict:
    """Fit the E0 shift aligning simulated chi(k) with an experimental spectrum.

    Shifts the simulated k grid (k'^2 = k^2 + E0/3.81, Artemis/IFEFFIT sign
    convention) and minimizes the mean
    squared k^2*chi deviation from experiment over [kmin, kmax]. Writes the
    shifted spectrum, a comparison table, a summary and calc_vs_exp.png.

    Args:
        chi_file: Averaged *-chi_avg.dat file from ``mlmd_average_chi``.
        exp_file: Experimental chi(k) file (.dat whitespace or .csv comma).
        outdir: Output directory (default ``<chi_file stem>_E0_fit``).
        exp_col2_is_k2chi: Experimental column 2 is already k^2*chi.
        kmin: Fit kmin in A^-1.
        kmax: Fit kmax in A^-1.
        e0_min: E0 search minimum in eV.
        e0_max: E0 search maximum in eV.
    """
    result = fit_e0(
        sim_file=chi_file,
        exp_file=exp_file,
        outdir=outdir or f"{Path(chi_file).with_suffix('')}_E0_fit",
        kmin=kmin,
        kmax=kmax,
        exp_col2_is_k2chi=exp_col2_is_k2chi,
        e0_min=e0_min,
        e0_max=e0_max,
    )
    result["status"] = "success"
    return result


@mcp.tool()
def mlmd_lcf(
    exp_file: str,
    standards: list[str],
    outdir: str = "lcf_fit",
    metric: str = "redchi",
    max_components: int = 3,
    kmin: Optional[float] = None,
    kmax: Optional[float] = None,
    k_weight: int = 2,
    e0_min: float = -20.0,
    e0_max: float = 20.0,
    e0_sign: float = -1.0,
    scale_for_shift: bool = True,
) -> dict:
    """Linear combination fit of several simulated chi(k) spectra to experiment.

    First fits an individual delta E0 for every simulated standard, then fits
    non-negative weights summing to 1 for every combination of up to
    ``max_components`` standards, ranking combinations by ``metric``. Writes
    lcf_results.csv, delta_e0.csv, best_lcf_fit.dat and best_lcf_fit.png.

    Args:
        exp_file: Experimental chi(k) file (.dat or .csv).
        standards: Simulated chi(k) files or glob patterns (e.g. "*-chi_avg.dat").
        outdir: Output directory.
        metric: redchi, chi2, rmsd, or rfactor.
        max_components: Maximum standards combined in one fit.
        kmin: Fit kmin in A^-1 (full experimental range if omitted).
        kmax: Fit kmax in A^-1 (full experimental range if omitted).
        k_weight: k-weight applied during E0 search and LCF.
        e0_min: Delta E0 search minimum in eV.
        e0_max: Delta E0 search maximum in eV.
        e0_sign: E0 sign convention; -1 (default) is Artemis/IFEFFIT, same as
            ``mlmd_fit_e0``. +1 flips the sign of reported delta E0.
        scale_for_shift: Optimize a temporary amplitude during each E0 search.
    """
    result = run_lcf(
        exp_file=exp_file,
        standards=standards,
        outdir=outdir,
        metric=metric,
        max_components=max_components,
        kmin=kmin,
        kmax=kmax,
        k_weight=k_weight,
        e0_min=e0_min,
        e0_max=e0_max,
        e0_sign=e0_sign,
        scale_for_shift=scale_for_shift,
    )
    result["status"] = "success"
    return result


@mcp.tool()
def mlmd_plot(
    chi_file: str,
    savefile: str,
    k_weight: int = 2,
    band: str = "sem",
    n_samples: Optional[int] = None,
) -> dict:
    """Plot k-weighted chi(k) with the MD sampling band shaded around the mean.

    Args:
        chi_file: Averaged *-chi_avg.dat file from ``mlmd_average_chi``.
        savefile: Base path for the output PNG.
        k_weight: k-weighting exponent in k^n*chi(k).
        band: Band type: "sem", "std", or "none".
        n_samples: Frame count, annotated on the plot when given.
    """
    result = plot_chi(chi_file, savefile, k_weight=k_weight, band=band, n_samples=n_samples)
    result["status"] = "success"
    return result


@mcp.tool()
def mlmd_convergence(
    directory: str,
    savefile: str = "exafs_convergence",
    step: int = 10,
    k_weight: int = 2,
    kmin: float = 2.0,
    kmax: float = 11.0,
) -> dict:
    """Plot k-space and R-space convergence versus number of averaged snapshots.

    Writes a two-panel PNG: k^n*chi(k) (left) and |chi(R)| via a
    Hanning-windowed FFT (right), each drawn for increasing snapshot counts.

    Args:
        directory: FEFF output directory.
        savefile: Base path for the output PNG.
        step: Snapshot count increment between curves.
        k_weight: k-weighting exponent.
        kmin: FT window kmin in A^-1.
        kmax: FT window kmax in A^-1.
    """
    result = plot_convergence(
        directory, savefile, step=step, k_weight=k_weight, kmin=kmin, kmax=kmax
    )
    result["status"] = "success"
    return result


def main() -> None:
    """Entry point: run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()

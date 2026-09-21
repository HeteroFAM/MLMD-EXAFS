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
from .feff import generate_feff_inputs_from_trajectory
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
) -> dict:
    """Batch-execute the FEFF binary over generated input directories.

    Runs FEFF in every ``feff.inp`` subdirectory of ``directory`` with bounded
    concurrency, producing a ``chi.dat`` in each. Requires a FEFF9 executable.

    Args:
        directory: FEFF output directory from ``mlmd_feff_input``.
        feff_bin: Path to the FEFF executable.
        max_workers: Maximum concurrent FEFF jobs.
    """
    result = run_feff_batch(directory, feff_bin=feff_bin, max_workers=max_workers)
    result["status"] = "success" if result["n_failed"] == 0 else "partial"
    return result


@mcp.tool()
def mlmd_average_chi(directory: str, savefile: str) -> dict:
    """Average all chi.dat files into a converged chi(k) with a sampling band.

    Writes ``<savefile>-chi_avg.dat`` (columns k, chi_avg, chi_std, chi_sem).
    The std/sem columns are the MD sampling dispersion, not a force-field error
    bar.

    Args:
        directory: FEFF output directory containing chi.dat subdirectories.
        savefile: Base path for the averaged chi file.
    """
    result = average_chi(directory, savefile)
    # numpy arrays are not JSON-serializable; return only the scalar summary.
    for key in ("k", "chi_avg", "chi_std", "chi_sem"):
        result.pop(key, None)
    result["status"] = "success" if result["n_samples"] > 0 else "no_data"
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

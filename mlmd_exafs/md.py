"""Cell relaxation and NVT molecular dynamics with an MLIP force engine.

The MD stage produces the thermally sampled trajectory whose snapshots feed
FEFF. Uses ASE's Nose-Hoover chain thermostat for correct canonical (NVT)
sampling. The relaxation stage optimizes both atomic positions and the cell
with a Frechet cell filter so the equilibrium lattice falls out of the run.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
from ase import units
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.io import Trajectory, read, write
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import FIRE


# ---------------------------------------------------------------------------
# Cell relaxation
# ---------------------------------------------------------------------------


def relax(
    input_structure: str,
    output_structure: str,
    calculator,
    fmax: float = 0.02,
    steps: int = 10000,
    keep_symmetry: bool = True,
    target_pressure_GPa: float = 0.0,
    max_volume_change: float = 0.25,
) -> dict:
    """Relax atomic positions and the cell with the given MLIP calculator.

    Parameters
    ----------
    input_structure : str
        Path to an ASE-readable structure (CIF, extxyz, POSCAR, ...).
    output_structure : str
        Where to write the relaxed structure.
    calculator : ase Calculator
        The MLIP force engine (see :mod:`mlmd_exafs.calculators`).
    fmax : float
        Force convergence threshold in eV/A.
    steps : int
        Maximum optimizer steps.
    keep_symmetry : bool
        Constrain the relaxation to preserve the starting crystal symmetry
        (ASE ``FixSymmetry``).
    target_pressure_GPa : float
        Target external hydrostatic pressure in GPa.
    max_volume_change : float
        Relative volume change above which a warning is issued (0.25 = 25%).

    Returns
    -------
    dict
        Convergence flag, step count, final energy, cell parameters, relative
        volume change, and a ``volume_warning`` flag.
    """
    atoms = read(input_structure)
    atoms.calc = calculator

    # Fail early if the model doesn't provide stress
    try:
        s = atoms.get_stress()
    except Exception as e:
        raise RuntimeError(
            "Calculator cannot provide stress; cell relaxation is not possible."
        ) from e
    if not np.all(np.isfinite(s)):
        raise RuntimeError("Non-finite stress from calculator.")

    v0 = atoms.get_volume()

    if keep_symmetry:
        atoms.set_constraint(FixSymmetry(atoms))

    cell0 = atoms.cell.cellpar()
    ecf = FrechetCellFilter(atoms, scalar_pressure=target_pressure_GPa * units.GPa)
    opt = FIRE(ecf)
    converged = opt.run(fmax=fmax, steps=steps)
    cell1 = atoms.cell.cellpar()

    # Drop the constraint so the output (and MD downstream) is unconstrained.
    atoms.set_constraint()

    out_path = Path(output_structure)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write(str(out_path), atoms)

    dv = atoms.get_volume() / v0 - 1
    volume_warning = abs(dv) > max_volume_change
    if volume_warning:
        warnings.warn(
            f"Volume changed by {dv:+.1%}; check the starting structure and "
            "whether the model is valid for it.",
            stacklevel=2,
        )

    return {
        "converged": bool(converged),
        "steps": int(opt.nsteps),
        "final_energy_eV": float(atoms.get_potential_energy()),
        "initial_cellpar": [float(x) for x in cell0],
        "final_cellpar": [float(x) for x in cell1],
        "volume_change": float(dv),
        "volume_warning": bool(volume_warning),
        "output": str(out_path),
    }


# ---------------------------------------------------------------------------
# Molecular dynamics
# ---------------------------------------------------------------------------


def md_engine(
    atoms,
    temperature: float = 300.0,
    step_size: float = 10.0,
    n_steps: int = 11000,
    traj_file: str = "md.traj",
    logfile: str = "-",
) -> None:
    """Run NVT MD (Nose-Hoover chain) on ``atoms`` with an attached calculator.

    ``step_size`` is given in atomic units of time and converted to fs
    internally (1 a.u. ~= 0.02419 fs). A calculator must already be attached to
    ``atoms``.
    """
    step_size_fs = step_size * 0.02419

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, force_temp=True)

    dyn = NoseHooverChainNVT(
        atoms,
        step_size_fs * units.fs,
        temperature_K=temperature,
        tdamp=100 * step_size_fs * units.fs,
        trajectory=traj_file,
        logfile=logfile,
    )
    dyn.run(n_steps)


def run_md(
    input_structure: str,
    calculator,
    save_directory: str,
    temperature: float = 300.0,
    step_size: float = 10.0,
    n_steps: int = 11000,
) -> dict:
    """Run an NVT MD trajectory and keep it as an ASE ``.traj`` file.

    Output layout::

        <save_directory>/<structure_stem>/<structure_stem>.traj   # trajectory
        <save_directory>/<structure_stem>/<structure_stem>.log    # MD log

    The trajectory is written to ``<structure_stem>.partial.traj`` while MD
    runs and renamed on completion, so an interrupted run is never mistaken
    for a finished one. Skips the run (and reports it) if the final trajectory
    already exists, so the stage is safe to re-invoke.

    Returns
    -------
    dict
        Trajectory path, frame count, trajectory length in ps, and run time.
    """
    name = Path(input_structure).stem
    save_dir = Path(save_directory, name)
    save_dir.mkdir(parents=True, exist_ok=True)

    traj_path = save_dir / f"{name}.traj"
    partial_path = save_dir / f"{name}.partial.traj"
    logfile = save_dir / f"{name}.log"

    if traj_path.exists():
        print(f"Skipping {name}: trajectory {traj_path} already exists.")
        with Trajectory(str(traj_path)) as traj:
            n_frames = len(traj)
        return {"trajectory": str(traj_path), "n_frames": n_frames, "skipped": True}

    partial_path.unlink(missing_ok=True)
    logfile.unlink(missing_ok=True)

    atoms = read(input_structure, index=0)
    atoms.calc = calculator

    start = time.time()
    md_engine(
        atoms,
        temperature=temperature,
        step_size=step_size,
        n_steps=n_steps,
        traj_file=str(partial_path),
        logfile=str(logfile),
    )
    elapsed = time.time() - start

    partial_path.replace(traj_path)
    with Trajectory(str(traj_path)) as traj:
        n_frames = len(traj)

    # a.u. step -> ps: step_size * 0.02419 fs/step / 1000 fs/ps
    ps_per_frame = step_size * 0.02419 / 1000
    traj_len_ps = n_frames * ps_per_frame

    print(
        f"MD done for {name}: {elapsed / 60:.2f} min, "
        f"{n_frames} frames, {traj_len_ps:.2f} ps."
    )
    return {
        "trajectory": str(traj_path),
        "n_frames": n_frames,
        "trajectory_length_ps": traj_len_ps,
        "elapsed_s": round(elapsed, 2),
        "skipped": False,
    }

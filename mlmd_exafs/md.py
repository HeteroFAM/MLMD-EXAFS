"""Cell relaxation and NVT molecular dynamics with an MLIP force engine.

The MD stage produces the thermally sampled trajectory whose snapshots feed
FEFF. Uses ASE's Nose-Hoover chain thermostat for correct canonical (NVT)
sampling. The relaxation stage optimizes both atomic positions and the cell
with a Frechet cell filter so the equilibrium lattice falls out of the run.
"""

from __future__ import annotations

import time
from pathlib import Path

from ase import units
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
    fmax: float = 0.05,
    steps: int = 10000,
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

    Returns
    -------
    dict
        Convergence flag, step count, final energy, and cell parameters.
    """
    atoms = read(input_structure)
    atoms.calc = calculator

    cell0 = atoms.cell.cellpar()
    ecf = FrechetCellFilter(atoms)
    opt = FIRE(ecf)
    converged = opt.run(fmax=fmax, steps=steps)
    cell1 = atoms.cell.cellpar()

    out_path = Path(output_structure)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write(str(out_path), atoms)

    return {
        "converged": bool(converged),
        "steps": int(opt.nsteps),
        "final_energy_eV": float(atoms.get_potential_energy()),
        "initial_cellpar": [float(x) for x in cell0],
        "final_cellpar": [float(x) for x in cell1],
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
    """Run an NVT MD trajectory and write it as an extxyz file.

    Output layout::

        <save_directory>/<structure_stem>/<structure_stem>.xyz   # trajectory
        <save_directory>/<structure_stem>/<structure_stem>.log    # MD log

    Skips the run (and reports it) if the extxyz trajectory already exists, so
    the stage is safe to re-invoke.

    Returns
    -------
    dict
        Trajectory path, frame count, trajectory length in ps, and run time.
    """
    name = Path(input_structure).stem
    save_dir = Path(save_directory, name)
    save_dir.mkdir(parents=True, exist_ok=True)

    traj_path = save_dir / f"{name}.traj"
    xyz_path = save_dir / f"{name}.xyz"
    logfile = save_dir / f"{name}.log"

    if xyz_path.exists():
        print(f"Skipping {name}: trajectory {xyz_path} already exists.")
        traj = read(str(xyz_path), ":")
        return {"trajectory": str(xyz_path), "n_frames": len(traj), "skipped": True}

    traj_path.unlink(missing_ok=True)
    logfile.unlink(missing_ok=True)

    atoms = read(input_structure, index=0)
    atoms.calc = calculator

    start = time.time()
    md_engine(
        atoms,
        temperature=temperature,
        step_size=step_size,
        n_steps=n_steps,
        traj_file=str(traj_path),
        logfile=str(logfile),
    )
    elapsed = time.time() - start

    traj = Trajectory(str(traj_path))
    write(str(xyz_path), traj, format="extxyz")
    n_frames = len(traj)
    traj.close()
    traj_path.unlink(missing_ok=True)

    # a.u. step -> ps: step_size * 0.02419 fs/step / 1000 fs/ps
    ps_per_frame = step_size * 0.02419 / 1000
    traj_len_ps = n_frames * ps_per_frame

    print(
        f"MD done for {name}: {elapsed / 60:.2f} min, "
        f"{n_frames} frames, {traj_len_ps:.2f} ps."
    )
    return {
        "trajectory": str(xyz_path),
        "n_frames": n_frames,
        "trajectory_length_ps": traj_len_ps,
        "elapsed_s": round(elapsed, 2),
        "skipped": False,
    }

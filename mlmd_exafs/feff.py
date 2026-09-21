"""FEFF input generation from crystal structures and MD trajectories.

Provides ``carve_out`` (extract a local atomic environment around an absorbing
atom) and ``generate_feff_inputs_from_trajectory`` (batch-write ``feff.inp``
files for sampled trajectory frames). FEFF card parameters (HOLE/edge, SCF,
RMAX, CONTROL, CORRECTIONS, S0^2) are fully configurable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase import build
from ase.data import chemical_symbols
from ase.geometry import wrap_positions
from ase.io import read, write


# ---------------------------------------------------------------------------
# Local environment extraction
# ---------------------------------------------------------------------------


def _check_distance(atoms, rmax: float) -> bool:
    positions = atoms.get_positions()
    min_distance = np.abs(
        np.concatenate((positions.min(axis=0), positions.max(axis=0)))
    ).min()
    return min_distance > rmax


def _supercell_repeats(atoms, rmax: float) -> list[int]:
    cell_lengths = np.linalg.norm(atoms.cell.array, axis=1)
    repeats = []
    for length in cell_lengths:
        repeat = max(3, int(np.ceil((2 * rmax) / length)))
        if repeat % 2 == 0:
            repeat += 1
        repeats.append(repeat)
    return repeats


def carve_out(atoms, target_atom: int, rmax: float = 8.5) -> tuple[str, str, Any]:
    """Carve a local atomic environment around an absorber for FEFF input.

    Builds a supercell, centers the target atom at the origin, and extracts all
    neighbors within ``rmax`` (hydrogen atoms are excluded — FEFF ignores them
    as scatterers). Returns the POTENTIALS and ATOMS block strings ready for
    ``feff.inp`` assembly plus the carved ASE ``Atoms`` cluster.

    Parameters
    ----------
    atoms : ase.Atoms
        Input structure (unit cell or a trajectory snapshot).
    target_atom : int
        Index of the absorbing atom.
    rmax : float
        Cluster radius in Angstroms (default 8.5). Should exceed the FEFF RMAX
        card by ~2.5 A so no scattering paths are truncated.

    Returns
    -------
    tuple[str, str, ase.Atoms]
        ``(ipots_string, atoms_string, cluster_atoms)``.
    """
    if target_atom < 0 or target_atom >= len(atoms):
        raise ValueError(
            f"target_atom index {target_atom} out of range for structure "
            f"with {len(atoms)} atoms."
        )

    supercell = build.make_supercell(
        atoms, P=np.diag(_supercell_repeats(atoms, rmax)).astype("int")
    )

    supercell.translate(-1 * supercell.get_positions()[target_atom])
    supercell.set_positions(
        wrap_positions(supercell.get_positions(), cell=supercell.cell, center=(0, 0, 0))
    )

    if not _check_distance(supercell, rmax):
        raise ValueError(
            f"Expanded supercell not large enough for rmax={rmax} A. "
            "Try a larger input cell or reduce rmax."
        )

    tags = np.ones(len(supercell))
    tags[target_atom] = 0
    tags[np.where(supercell.get_atomic_numbers() == 1)[0]] = 0
    tags[
        np.where(
            supercell.get_distances(target_atom, range(len(supercell)), mic=True) > rmax
        )[0]
    ] = 0
    supercell.set_tags(tags.astype(int))

    atoms_to_keep = np.where(supercell.get_tags() == 1)[0]

    elems = supercell[atoms_to_keep].get_atomic_numbers()
    central_tag = "" if atoms[target_atom].number not in elems else "0"
    ipot_map = {}
    ipots_string = (
        f"{0: >7}{atoms[target_atom].number: >7}"
        f"{chemical_symbols[atoms[target_atom].number]: >7}{central_tag}\n"
    )
    for i, elem in enumerate(sorted(set(elems)), start=1):
        ipots_string += f"{i: >7}{elem: >7}{chemical_symbols[elem]: >7}\n"
        ipot_map[elem] = i

    if not np.isclose(supercell.get_positions()[target_atom].sum(), 0.0, atol=0.0001):
        raise ValueError("Target atom not at (0,0,0) after centering.")

    atoms_string = "     0.000000    0.000000    0.000000    0    0.000000\n"
    for neighbor in atoms_to_keep:
        x = f"{supercell[neighbor].position[0]:.6f}"
        y = f"{supercell[neighbor].position[1]:.6f}"
        z = f"{supercell[neighbor].position[2]:.6f}"
        ipot = ipot_map[supercell[neighbor].number]
        nn = f"{supercell.get_distance(target_atom, neighbor, mic=True):.6f}"
        atoms_string += f"   {x: >10}   {y: >9}   {z: >9}    {ipot: >1}   {nn: >3}\n"

    cluster = supercell[np.append(atoms_to_keep, target_atom)]
    return ipots_string, atoms_string, cluster


# ---------------------------------------------------------------------------
# feff.inp assembly
# ---------------------------------------------------------------------------


def _build_feff_inp(
    run_name: str,
    frame: int,
    target_atom: int,
    output_dir: Path,
    hole: int,
    s02: float,
    control: str,
    print_flags: str,
    rmax: float,
    scf: str,
    corrections: str | None,
    ipots_string: str,
    atoms_string: str,
) -> str:
    """Assemble a complete feff.inp from card parameters and carved blocks."""
    lines = [
        f"* label:{output_dir}/feff_run{frame:0>6}_{target_atom}.inp:label",
        f"* data_dir:{output_dir}:data_dir",
        f"* frame: {frame} :frame  aindex: {target_atom} :aindex",
        f"TITLE {run_name} frame {frame}",
        "",
        f"HOLE {hole} {s02:.6f}",
        f"CONTROL {control}",
        f"PRINT   {print_flags}",
        "",
        f"RMAX {rmax:<8.4f}",
        f"SCF {scf}",
    ]
    if corrections is not None:
        lines.append(f"CORRECTIONS {corrections}")
    lines += [
        "",
        "POTENTIALS",
        "*  IPOT     Z     tag",
        ipots_string,
        "ATOMS",
        "*      X           Y           Z      IPOT    NN-DIST",
        atoms_string,
        "END",
    ]
    return "\n".join(lines) + "\n"


def generate_feff_inputs_from_trajectory(
    trajectory_path: str,
    target_atom: int,
    hole: int,
    rmax: float,
    scf: str = "6.0 0 30 0.2 1",
    s02: float = 1.0,
    control: str = "1 1 1 1 1 1",
    print_flags: str = "0 0 0 0 0 0",
    corrections: str | None = None,
    step_size: int = 250,
    sampling_start: int = 0,
) -> dict[str, Any]:
    """Write batch FEFF inputs from an MD trajectory.

    Samples frames at a fixed interval, carves the local environment around the
    absorber in each, and writes a ``feff.inp`` per sampled frame into its own
    subdirectory.

    Parameters
    ----------
    trajectory_path : str
        ASE-readable trajectory (extxyz, traj, ...).
    target_atom : int
        Index of the absorbing atom (consistent across all frames).
    hole : int
        HOLE card index: 1=K, 2=L1, 3=L2, 4=L3.
    rmax : float
        FEFF RMAX path cutoff in Angstroms. The carve radius is rmax + 2.5 A.
    scf : str
        SCF card parameters, e.g. ``"6.0 0 30 0.2 1"``.
    s02 : float
        S0^2 amplitude reduction factor.
    control, print_flags : str
        CONTROL / PRINT card values.
    corrections : str or None
        CORRECTIONS card (``"vrcorr vicorr"``), or None to omit.
    step_size : int
        Sample every N-th frame.
    sampling_start : int
        First frame index to sample (use to skip equilibration).

    Returns
    -------
    dict
        ``output_dir``, ``n_inputs``, ``frames``.
    """
    traj_path = Path(trajectory_path)
    trajectory = read(str(traj_path), ":")
    run_name = traj_path.stem
    traj_dir = traj_path.parent

    absorber_symbol = trajectory[0][target_atom].symbol
    vrcorr_label = corrections.split()[0] if corrections else "0.0"

    output_dir = traj_dir / (
        f"exafs_{absorber_symbol}"
        f"_hole{hole}"
        f"_de_{vrcorr_label}"
        f"_s02_{s02:.1f}"
        f"_rc_{rmax:.1f}"
    )
    output_dir.mkdir(exist_ok=True)

    n_frames = len(trajectory)
    frames = list(range(sampling_start, n_frames, step_size))
    cluster_rmax = rmax + 2.5

    carved_regions = []
    for frame in frames:
        struct = trajectory[frame]
        ipots_string, atoms_string, region = carve_out(
            struct, target_atom, rmax=cluster_rmax
        )
        region.pbc = False
        region.cell = None
        carved_regions.append(region)

        subdir = output_dir / f"{frame:0>6}_{target_atom}"
        subdir.mkdir(exist_ok=True)

        inp_content = _build_feff_inp(
            run_name=run_name,
            frame=frame,
            target_atom=target_atom,
            output_dir=output_dir,
            hole=hole,
            s02=s02,
            control=control,
            print_flags=print_flags,
            rmax=rmax,
            scf=scf,
            corrections=corrections,
            ipots_string=ipots_string,
            atoms_string=atoms_string,
        )
        (subdir / "feff.inp").write_text(inp_content)

    neighborhoods_path = output_dir / f"neighborhoods_{target_atom}.xyz"
    write(str(neighborhoods_path), carved_regions, format="extxyz")

    return {
        "output_dir": str(output_dir),
        "n_inputs": len(frames),
        "frames": frames,
    }

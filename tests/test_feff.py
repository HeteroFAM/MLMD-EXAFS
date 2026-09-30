import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from mlmd_exafs.feff import carve_out, generate_feff_inputs_from_trajectory


def _parse_atoms_block(atoms_string):
    return np.array([[float(x) for x in line.split()] for line in atoms_string.strip().splitlines()])


def test_carve_out_rejects_bad_index(cu_atoms):
    with pytest.raises(ValueError, match="out of range"):
        carve_out(cu_atoms, 99, rmax=5.0)


def test_carve_out_cluster_geometry(cu_atoms):
    rmax = 6.0
    ipots, atoms_string, cluster = carve_out(cu_atoms, 0, rmax=rmax)

    rows = _parse_atoms_block(atoms_string)
    # absorber first, at origin, ipot 0
    assert np.allclose(rows[0, :3], 0.0) and rows[0, 3] == 0
    dists = np.linalg.norm(rows[1:, :3], axis=1)
    assert np.all(dists <= rmax + 1e-6)
    assert np.allclose(dists, rows[1:, 4], atol=1e-5)
    # fcc Cu: 12 nearest neighbors at a/sqrt(2)
    assert np.sum(np.isclose(dists, 3.61 / np.sqrt(2), atol=1e-3)) == 12
    assert len(cluster) == len(rows)

    # absorber also appears as a scatterer, so its ipot 0 line is tagged "0"
    lines = ipots.strip().splitlines()
    assert lines[0].split() == ["0", "29", "Cu0"]
    assert lines[1].split() == ["1", "29", "Cu"]


@pytest.mark.xfail(
    raises=ValueError,
    strict=True,
    reason=(
        "Known bug: _supercell_repeats picks n = ceil(2*rmax/L), which only "
        "guarantees n*L/2 >= rmax. After the absorber is centered, the "
        "farthest atom plane is ~n*L/2 minus one atom spacing away, so "
        "_check_distance rejects the supercell (e.g. cubic Cu, a=3.61 A, at "
        "the default carve radius 6.0 + 2.5 = 8.5 A)."
    ),
)
@pytest.mark.parametrize("rmax", [5.0, 8.5])
def test_carve_out_supercell_large_enough(cu_atoms, rmax):
    carve_out(cu_atoms, 0, rmax=rmax)


def test_carve_out_excludes_hydrogen():
    atoms = Atoms(
        "ZnOH",
        positions=[[0, 0, 0], [2.0, 0, 0], [0, 1.0, 0]],
        cell=[6, 6, 6],
        pbc=True,
    )
    ipots, atoms_string, cluster = carve_out(atoms, 0, rmax=4.0)
    assert "H" not in cluster.get_chemical_symbols()
    assert " H" not in ipots
    # Zn absorber is not a scatterer element here (only its periodic images
    # are, at 6 A > rmax), so no "0" tag on the absorber ipot
    assert ipots.splitlines()[0].split() == ["0", "30", "Zn"]


def test_generate_feff_inputs(short_traj):
    result = generate_feff_inputs_from_trajectory(
        str(short_traj), target_atom=0, hole=1, rmax=4.0, step_size=2, sampling_start=1
    )
    out = short_traj.parent / "exafs_Cu_hole1_de_0.0_s02_1.0_rc_4.0"
    assert result["output_dir"] == str(out)
    assert result["frames"] == [1, 3]
    assert result["n_inputs"] == 2

    inp = (out / "000001_0" / "feff.inp").read_text()
    assert "HOLE 1 1.000000" in inp
    assert "RMAX 4.0000" in inp
    assert "SCF 6.0 0 30 0.2 1" in inp
    assert "CORRECTIONS" not in inp
    assert "TITLE cu_md frame 1" in inp
    assert inp.rstrip().endswith("END")
    assert (out / "000003_0" / "feff.inp").is_file()

    neighborhoods = read(str(out / "neighborhoods_0.xyz"), ":")
    assert len(neighborhoods) == 2


def test_generate_feff_inputs_with_corrections(short_traj):
    result = generate_feff_inputs_from_trajectory(
        str(short_traj), target_atom=0, hole=4, rmax=4.0, s02=0.9,
        corrections="2.5 0.0", step_size=10,
    )
    assert result["output_dir"].endswith("exafs_Cu_hole4_de_2.5_s02_0.9_rc_4.0")
    inp = (short_traj.parent / result["output_dir"] / "000000_0" / "feff.inp").read_text()
    assert "CORRECTIONS 2.5 0.0" in inp
    assert "HOLE 4 0.900000" in inp

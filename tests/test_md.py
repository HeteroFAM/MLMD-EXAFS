import pytest
import math

from ase.calculators.emt import EMT
from ase.io import read

from mlmd_exafs.md import relax, run_md


def test_relax_with_emt(cu_cif, tmp_path):
    out = tmp_path / "out" / "relaxed.xyz"
    result = relax(str(cu_cif), str(out), EMT(), fmax=0.05, steps=200)

    assert out.is_file()
    assert set(result) == {
        "converged", "steps", "final_energy_eV", "initial_cellpar",
        "final_cellpar", "output", "volume_change", "volume_warning",
    }
    assert result["converged"] is True
    assert math.isfinite(result["final_energy_eV"])
    assert len(result["final_cellpar"]) == 6
    assert len(read(str(out))) == 4


def test_relax_volume_warning_and_options(cu_cif, tmp_path):
    out = tmp_path / "relaxed.xyz"
    with pytest.warns(UserWarning, match="Volume changed"):
        result = relax(
            str(cu_cif), str(out), EMT(), fmax=0.05, steps=200,
            keep_symmetry=False, target_pressure_GPa=0.0, max_volume_change=0.0,
        )
    assert result["volume_warning"] is True
    assert not read(str(out)).constraints


def test_relax_keep_symmetry_no_output_constraint(cu_cif, tmp_path):
    out = tmp_path / "relaxed.xyz"
    result = relax(str(cu_cif), str(out), EMT(), fmax=0.05, steps=200)
    assert result["volume_warning"] is False
    assert not read(str(out)).constraints


def test_run_md_writes_traj_and_skips_rerun(cu_cif, tmp_path):
    md_dir = tmp_path / "md_out"
    result = run_md(str(cu_cif), EMT(), str(md_dir), temperature=300, n_steps=5)

    traj = md_dir / "cu" / "cu.traj"
    assert result["trajectory"] == str(traj)
    assert traj.is_file()
    assert not (md_dir / "cu" / "cu.xyz").exists()
    assert not (md_dir / "cu" / "cu.partial.traj").exists()
    assert result["n_frames"] == len(read(str(traj), ":")) >= 5

    again = run_md(str(cu_cif), EMT(), str(md_dir), n_steps=5)
    assert again["skipped"] is True
    assert again["n_frames"] == result["n_frames"]

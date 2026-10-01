import json

import numpy as np
import pytest
from ase.calculators.emt import EMT

from mlmd_exafs import cli

SUBCOMMANDS = [
    "relax", "md", "feff-input", "run-feff", "cleanup",
    "average", "fit-e0", "lcf", "plot", "convergence",
]


def _output(capsys):
    """Parse the JSON result the CLI prints after any progress output."""
    lines = capsys.readouterr().out.splitlines()
    start = len(lines) - 1 - lines[::-1].index("{")
    return json.loads("\n".join(lines[start:]))


def test_parser_has_all_subcommands():
    parser = cli.build_parser()
    sub = next(a for a in parser._actions if a.dest == "command")
    assert sorted(sub.choices) == sorted(SUBCOMMANDS)


@pytest.mark.parametrize("cmd", SUBCOMMANDS)
def test_help_exits_zero(cmd, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([cmd, "--help"])
    assert exc.value.code == 0
    assert "usage: mlmd-exafs" in capsys.readouterr().out


def test_unknown_backend_rejected(cu_cif):
    with pytest.raises(SystemExit):
        cli.main(["relax", "-i", str(cu_cif), "-o", "x.xyz", "--backend", "nope"])


def test_model_and_checkpoint_together_error(cu_cif, tmp_path):
    ckpt = tmp_path / "m.model"
    ckpt.write_text("x")
    with pytest.raises(ValueError, match="either model or checkpoint"):
        cli.main([
            "relax", "-i", str(cu_cif), "-o", str(tmp_path / "r.xyz"),
            "--backend", "mace", "--model", "small", "--checkpoint", str(ckpt),
        ])


@pytest.fixture
def emt_backend(monkeypatch):
    calls = []

    def fake_build(backend, **kwargs):
        calls.append((backend, kwargs))
        return EMT()

    monkeypatch.setattr("mlmd_exafs.calculators.build_calculator", fake_build)
    return calls


def test_relax_and_md(cu_cif, tmp_path, emt_backend, capsys):
    out = tmp_path / "relaxed.xyz"
    cli.main([
        "relax", "-i", str(cu_cif), "-o", str(out), "--backend", "uma",
        "--head", "omol", "--steps", "50",
    ])
    assert _output(capsys)["output"] == str(out)
    assert emt_backend[0] == ("uma", {
        "device": "cpu", "model": None, "head": "omol", "modal": "mpa", "checkpoint": None,
    })

    cli.main([
        "md", "-i", str(out), "-d", str(tmp_path / "md"), "--backend", "orb",
        "--n-steps", "3",
    ])
    assert _output(capsys)["trajectory"] == str(tmp_path / "md" / "relaxed" / "relaxed.traj")


def test_feff_input(short_traj, capsys):
    cli.main([
        "feff-input", "-f", str(short_traj), "-i", "0", "--rmax", "4.0",
        "--step-size", "2", "--corrections", "1.0 0.0",
    ])
    result = _output(capsys)
    assert result["frames"] == [0, 2, 4]
    assert result["output_dir"].endswith("exafs_Cu_hole1_de_1.0_s02_1.0_rc_4.0")


def test_run_feff(exafs_dir, fake_feff_bin, capsys):
    cli.main(["run-feff", "-d", str(exafs_dir), "--feff-bin", str(fake_feff_bin), "--no-cleanup"])
    result = _output(capsys)
    assert result["n_ok"] == 4 and result["cleanup"] is None


def test_cleanup_dry_run(exafs_dir, capsys):
    cli.main(["cleanup", str(exafs_dir.parent), "--dry-run"])
    result = _output(capsys)
    assert result["deleted_files"] == 8
    assert (exafs_dir / "000000_0" / "pot.bin").exists()


def test_average_plot_convergence(exafs_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cli.main(["average", "-d", str(exafs_dir), "--savefile", "run"])
    result = _output(capsys)
    assert result["n_samples"] == 4
    assert "e0_fit" not in result and "chi_avg" not in result

    cli.main(["plot", "--chi-file", "run-chi_avg.dat", "--savefile", "k2", "--band", "std"])
    cli.main(["convergence", "-d", str(exafs_dir), "--savefile", "conv", "--step", "2"])
    assert (tmp_path / "k2.png").is_file()
    assert (tmp_path / "conv.png").is_file()


def test_average_with_e0_fit(exafs_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    k = np.linspace(2.0, 12.0, 200)
    exp = tmp_path / "exp.csv"
    exp.write_text("k,chi\n" + "\n".join(f"{a},{np.sin(5 * a) / a}" for a in k))
    cli.main([
        "average", "-d", str(exafs_dir), "--savefile", "run", "--exp-file", str(exp),
        "--e0-grid-n", "41",
    ])
    result = _output(capsys)
    assert result["e0_fit"]["outdir"] == str(tmp_path / "run_E0_fit")
    assert -10.0 <= result["e0_fit"]["best_E0"] <= 10.0


def test_lcf(standards, tmp_path, capsys):
    k = np.linspace(3.0, 13.0, 200)
    data = np.loadtxt(standards[0])
    exp = tmp_path / "exp.dat"
    np.savetxt(exp, np.column_stack([k, np.interp(k, data[:, 0], data[:, 1])]))
    cli.main([
        "lcf", "--exp-file", str(exp), "--standards", str(tmp_path / "sims" / "*.dat"),
        "-o", str(tmp_path / "lcf"), "--max-components", "1", "--metric", "rmse",
    ])
    result = _output(capsys)
    assert result["best"]["standards"] == ["a-chi_avg.dat"]
    assert result["best"]["optimized_metric"] == "rmsd"

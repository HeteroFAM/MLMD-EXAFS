"""Shared fixtures for the MLMD-EXAFS test suite.

The suite runs offline: the ASE EMT calculator stands in for the MLIP
backends and a small Python script stands in for the FEFF9 binary.
"""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.io import write

from mlmd_exafs.analysis import CHI_HEADER

K_GRID = np.round(np.arange(0.0, 20.0 + 1e-9, 0.05), 4)


def synthetic_chi(k, r=2.5, amp=1.0, sigma2=0.005):
    """Single-shell EXAFS-like damped sine."""
    k = np.asarray(k, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = amp * np.exp(-2 * sigma2 * k**2) * np.sin(2 * k * r) / np.where(k > 0, k, 1)
    return chi


def write_feff_chi(path: Path, k, chi) -> Path:
    """Write a FEFF-style chi.dat (header lines, CHI_HEADER, data rows)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [f" {ki:10.4f} {ci:12.6e} {abs(ci):12.6e} {0.0:12.6e}" for ki, ci in zip(k, chi)]
    path.write_text(
        "# Synthetic FEFF chi.dat\n# ----\n" + CHI_HEADER + "\n".join(rows)
    )
    return path


def write_two_column(path: Path, k, y, delimiter=" ") -> Path:
    path = Path(path)
    lines = ["# k chi"] + [f"{ki:.6f}{delimiter}{yi:.10e}" for ki, yi in zip(k, y)]
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def cu_atoms():
    return bulk("Cu", "fcc", a=3.61, cubic=True)


@pytest.fixture
def cu_cif(tmp_path, cu_atoms):
    path = tmp_path / "cu.cif"
    write(str(path), cu_atoms)
    return path


@pytest.fixture
def short_traj(tmp_path, cu_atoms):
    """extxyz trajectory of 5 rattled Cu frames at tmp/traj/cu_md.xyz."""
    frames = []
    for seed in range(5):
        a = cu_atoms.copy()
        a.rattle(stdev=0.02, seed=seed)
        frames.append(a)
    path = tmp_path / "traj" / "cu_md.xyz"
    path.parent.mkdir()
    write(str(path), frames, format="extxyz")
    return path


@pytest.fixture
def chi_file_factory():
    return write_feff_chi


@pytest.fixture
def exafs_dir(tmp_path):
    """exafs_* dir with 4 snapshot subdirs holding chi.dat plus FEFF scratch."""
    root = tmp_path / "exafs_Cu_hole1_de_0.0_s02_1.0_rc_6.0"
    for i, amp in enumerate([0.8, 0.9, 1.1, 1.2]):
        sub = root / f"{i * 250:0>6}_0"
        write_feff_chi(sub / "chi.dat", K_GRID, synthetic_chi(K_GRID, amp=amp))
        (sub / "feff.inp").write_text("TITLE test\nEND\n")
        (sub / "feff.out").write_text("ok\n")
        (sub / "pot.bin").write_bytes(b"\x00\x01")
        (sub / "log1.dat").write_text("scratch\n")
    (root / "neighborhoods_0.xyz").write_text("1\n\nCu 0 0 0\n")
    return root


def _make_script(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.fixture
def fake_feff_bin(tmp_path):
    """Executable stand-in for feff.x: writes chi.dat and a scratch file in CWD."""
    body = f"""
import numpy as np
k = np.round(np.arange(0.0, 20.0 + 1e-9, 0.05), 4)
chi = np.exp(-0.01 * k**2) * np.sin(5.0 * k) / np.where(k > 0, k, 1)
rows = [f" {{a:10.4f}} {{b:12.6e}} {{abs(b):12.6e}} {{0.0:12.6e}}" for a, b in zip(k, chi)]
with open("chi.dat", "w") as f:
    f.write("# fake feff\\n" + {CHI_HEADER!r} + "\\n".join(rows))
with open("pot.bin", "wb") as f:
    f.write(b"scratch")
print("fake feff done")
"""
    (tmp_path / "bin").mkdir()
    return _make_script(tmp_path / "bin" / "feff.x", body)


@pytest.fixture
def failing_feff_bin(tmp_path):
    (tmp_path / "badbin").mkdir()
    return _make_script(tmp_path / "badbin" / "feff.x", "import sys\nsys.exit(1)\n")


@pytest.fixture
def standards(tmp_path):
    """Two averaged-chi standards (4-column chi_avg.dat format) on K_GRID."""
    sims = tmp_path / "sims"
    sims.mkdir()
    paths = []
    for name, r in (("a", 2.2), ("b", 2.9)):
        chi = synthetic_chi(K_GRID, r=r)
        data = np.column_stack([K_GRID, chi, np.zeros_like(chi), np.zeros_like(chi)])
        path = sims / f"{name}-chi_avg.dat"
        np.savetxt(path, data, header="k chi_avg chi_std chi_sem")
        paths.append(path)
    return paths

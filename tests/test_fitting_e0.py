import os

import numpy as np
import pytest

from mlmd_exafs.fitting_E0 import (
    fit_e0,
    golden_section_minimize,
    grid_then_refine_minimize,
    read_experimental,
    read_two_column_numeric,
)

from conftest import K_GRID, synthetic_chi, write_two_column

K2EV = 3.81


def test_read_two_column_numeric_skips_junk_and_sorts(tmp_path):
    path = tmp_path / "d.dat"
    path.write_text("# header\nk chi\n\n2.0 20\n1.0 10\nbad line here\n3.0\nnan 5\n")
    k, y = read_two_column_numeric(path)
    assert k.tolist() == [1.0, 2.0]
    assert y.tolist() == [10.0, 20.0]


def test_read_two_column_numeric_empty(tmp_path):
    path = tmp_path / "d.dat"
    path.write_text("# nothing\n")
    with pytest.raises(RuntimeError, match="No valid numeric data"):
        read_two_column_numeric(path)


def test_read_experimental_csv(tmp_path):
    path = tmp_path / "e.csv"
    path.write_text("k,chi\n1.0, 0.5\n2.0 ,0.25\n")
    k, y = read_experimental(str(path))
    assert k.tolist() == [1.0, 2.0]
    assert y.tolist() == [0.5, 0.25]


def test_read_experimental_rejects_extension(tmp_path):
    path = tmp_path / "e.txt"
    path.write_text("1 2\n")
    with pytest.raises(ValueError, match="Unsupported experimental file type"):
        read_experimental(str(path))


def test_golden_section_minimize():
    x, fx = golden_section_minimize(lambda x: (x - 1.3) ** 2 + 2, -5, 5)
    assert x == pytest.approx(1.3, abs=1e-5)
    assert fx == pytest.approx(2.0)


def test_grid_then_refine_minimize():
    # minima of cos(x) + 0.01 x^2 solve sin(x) = 0.02 x, x ~ +-3.0800
    x, _ = grid_then_refine_minimize(lambda x: np.cos(x) + 0.01 * x**2, -6, 6, ngrid=121)
    assert abs(x) == pytest.approx(3.0800, abs=1e-3)


def test_grid_then_refine_all_nonfinite():
    with pytest.raises(RuntimeError, match="non-finite"):
        grid_then_refine_minimize(lambda x: np.inf, 0, 1, ngrid=5)


def _shifted_exp(tmp_path, e0_true, ext=".dat", delimiter=" "):
    """Experiment where chi_exp(k') = chi_sim(k), k'^2 = k^2 + E0/k2ev."""
    k_exp = np.linspace(1.0, 14.0, 400)
    k_sim_equiv = np.sqrt(k_exp**2 - e0_true / K2EV)
    chi_exp = synthetic_chi(k_sim_equiv)
    return write_two_column(tmp_path / f"exp{ext}", k_exp, chi_exp, delimiter)


def _sim_file(tmp_path):
    chi = synthetic_chi(K_GRID)
    data = np.column_stack([K_GRID[1:], chi[1:], chi[1:] * 0, chi[1:] * 0])
    path = tmp_path / "sim-chi_avg.dat"
    np.savetxt(path, data, header="k chi_avg chi_std chi_sem")
    return path


@pytest.mark.parametrize("e0_true,ext,delim", [(3.0, ".dat", " "), (-2.5, ".csv", ",")])
def test_fit_e0_recovers_shift(tmp_path, e0_true, ext, delim):
    exp = _shifted_exp(tmp_path, e0_true, ext, delim)
    result = fit_e0(
        str(_sim_file(tmp_path)), str(exp), outdir=str(tmp_path / "fit"),
        kmin=3.0, kmax=12.0, e0_grid_n=201,
    )
    assert result["best_E0"] == pytest.approx(e0_true, abs=0.05)
    # residual is only linear-interpolation error on the 0.05 A^-1 sim grid
    assert result["min_rsmd"] < 1e-3
    for key in ("shifted_sim_file", "comparison_file", "summary_file", "plot_file"):
        assert os.path.isfile(result[key])


def test_fit_e0_validates_ranges(tmp_path):
    sim, exp = _sim_file(tmp_path), _shifted_exp(tmp_path, 0.0)
    with pytest.raises(ValueError, match="kmin"):
        fit_e0(str(sim), str(exp), outdir=str(tmp_path / "o"), kmin=10, kmax=5)
    with pytest.raises(ValueError, match="e0-min"):
        fit_e0(str(sim), str(exp), outdir=str(tmp_path / "o"), e0_min=5, e0_max=-5)
    with pytest.raises(FileNotFoundError):
        fit_e0(str(tmp_path / "none.dat"), str(exp), outdir=str(tmp_path / "o"))

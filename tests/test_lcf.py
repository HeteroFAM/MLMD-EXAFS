import numpy as np
import pytest

from mlmd_exafs.lcf import (
    apply_k_weight,
    calculate_metrics,
    canonical_metric,
    resolve_standards,
    restrict_k_range,
    run_lcf,
    shifted_standard_on_exp_grid,
)

from conftest import K_GRID, synthetic_chi, write_two_column


def test_canonical_metric():
    assert canonical_metric(" RMSE ") == "rmsd"
    assert canonical_metric("r-factor") == "rfactor"
    assert canonical_metric("chi2red") == "redchi"
    with pytest.raises(ValueError, match="Unknown metric"):
        canonical_metric("mae")


def test_resolve_standards(standards, tmp_path):
    pattern = str(tmp_path / "sims" / "*-chi_avg.dat")
    assert resolve_standards(pattern) == [str(p) for p in standards]
    # explicit list plus a duplicating glob: order kept, duplicates dropped
    assert resolve_standards([str(standards[1]), pattern]) == [str(standards[1]), str(standards[0])]
    with pytest.raises(FileNotFoundError):
        resolve_standards([str(tmp_path / "missing.dat")])


def test_restrict_and_weight():
    k = np.array([1.0, 2.0, 3.0, 4.0])
    kk, yy = restrict_k_range(k, k * 10, kmin=2.0, kmax=3.0)
    assert kk.tolist() == [2.0, 3.0] and yy.tolist() == [20.0, 30.0]
    assert apply_k_weight(k, np.ones(4), 0).tolist() == [1, 1, 1, 1]
    assert apply_k_weight(k, np.ones(4), 2).tolist() == [1, 4, 9, 16]


def test_calculate_metrics():
    y = np.array([1.0, -1.0, 2.0, np.nan])
    perfect = calculate_metrics(y, y)
    assert perfect["n_points"] == 3
    assert perfect["chi2"] == perfect["rfactor"] == perfect["rmsd"] == 0.0

    m = calculate_metrics(np.array([1.0, 1.0]), np.array([0.0, 0.0]), n_params=1)
    assert m["chi2"] == 2.0 and m["redchi"] == 2.0 and m["rmsd"] == 1.0 and m["rfactor"] == 1.0

    assert calculate_metrics([np.nan], [1.0])["chi2"] == np.inf


def test_shifted_standard_zero_shift_is_identity():
    chi = synthetic_chi(K_GRID)
    k_exp = np.linspace(2, 12, 50)
    out = shifted_standard_on_exp_grid(k_exp, K_GRID, chi, 0.0)
    assert np.allclose(out, np.interp(k_exp, K_GRID, chi))
    # shifts pushing the query past the grid are NaN
    assert np.isnan(shifted_standard_on_exp_grid(np.array([19.99]), K_GRID, chi, -50.0)[0])


def _mix_exp(tmp_path, standards, weights, delta_e0=(1.5, -1.0)):
    k_exp = np.linspace(3.0, 13.0, 300)
    chi = np.zeros_like(k_exp)
    for path, w, de0 in zip(standards, weights, delta_e0):
        data = np.loadtxt(path)
        chi += w * shifted_standard_on_exp_grid(k_exp, data[:, 0], data[:, 1], de0)
    return write_two_column(tmp_path / "exp.dat", k_exp, chi)


def test_run_lcf_recovers_mixture(tmp_path, standards):
    exp = _mix_exp(tmp_path, standards, (0.6, 0.4))
    result = run_lcf(
        str(exp), [str(p) for p in standards], outdir=str(tmp_path / "lcf"),
        max_components=2, verbose=False,
    )
    best = result["best"]
    assert best["standards"] == ["a-chi_avg.dat", "b-chi_avg.dat"]
    assert best["weights"] == pytest.approx([0.6, 0.4], abs=0.02)
    assert best["delta_e0_eV"] == pytest.approx([1.5, -1.0], abs=0.2)
    assert result["n_fits"] == 3
    for f in ("lcf_results.csv", "delta_e0.csv", "best_lcf_fit.dat", "best_lcf_fit.png"):
        assert (tmp_path / "lcf" / f).is_file()


def test_run_lcf_validates_args(tmp_path, standards):
    exp = _mix_exp(tmp_path, standards, (1.0, 0.0))
    with pytest.raises(ValueError, match="max_components"):
        run_lcf(str(exp), [str(standards[0])], outdir=str(tmp_path / "o"), max_components=0)
    with pytest.raises(ValueError, match="e0_min"):
        run_lcf(str(exp), [str(standards[0])], outdir=str(tmp_path / "o"), e0_min=1, e0_max=0)
    with pytest.raises(RuntimeError, match="Too few"):
        run_lcf(str(exp), [str(standards[0])], outdir=str(tmp_path / "o"), kmin=50)

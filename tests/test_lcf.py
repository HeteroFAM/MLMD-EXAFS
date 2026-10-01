import itertools

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
    for f in ("lcf_results.csv", "delta_e0.csv", "best_lcf_fit.dat", "best_lcf_fit.png",
              "best_lcf_fit_R.png", "fit.log"):
        assert (tmp_path / "lcf" / f).is_file()
    log = (tmp_path / "lcf" / "fit.log").read_text()
    assert "Ran 3 LCF fits." in log and '"n_fits": 3' in log


def test_run_lcf_validates_args(tmp_path, standards):
    exp = _mix_exp(tmp_path, standards, (1.0, 0.0))
    with pytest.raises(ValueError, match="max_components"):
        run_lcf(str(exp), [str(standards[0])], outdir=str(tmp_path / "o"), max_components=0)
    with pytest.raises(ValueError, match="e0_min"):
        run_lcf(str(exp), [str(standards[0])], outdir=str(tmp_path / "o"), e0_min=1, e0_max=0)
    with pytest.raises(RuntimeError, match="Too few"):
        run_lcf(str(exp), [str(standards[0])], outdir=str(tmp_path / "o"), kmin=50)


# ---------------------------------------------------------------------------
# Joint delta E0 + weight refinement
# ---------------------------------------------------------------------------


def _legacy_fit_lcf_combo(k_exp, chi_exp, standards, combo, delta_e0_by_name,
                          metric="redchi", e0_sign=-1.0, kweight=2, min_valid_frac=0.95):
    """Verbatim copy of fit_lcf_combo before e0_mode was added (reference)."""
    from scipy.optimize import minimize

    from mlmd_exafs.lcf import _shifted_matrix, metric_score

    metric = canonical_metric(metric)
    n = len(combo)
    y_all = apply_k_weight(k_exp, chi_exp, kweight)
    X_all = _shifted_matrix(k_exp, standards, combo, delta_e0_by_name, e0_sign, kweight)
    valid = np.isfinite(y_all) & np.all(np.isfinite(X_all), axis=1)
    min_valid = max(5, int(np.ceil(min_valid_frac * k_exp.size)))
    if valid.sum() < min_valid:
        return None
    y = y_all[valid]
    X = X_all[valid, :]
    n_params = n + max(n - 1, 0)
    if n == 1:
        return {"weights": np.array([1.0]),
                "metrics": calculate_metrics(y, X[:, 0], n_params=n_params)}
    x0 = np.full(n, 1.0 / n)
    result = minimize(
        lambda w: metric_score(y, X @ w, metric, n_params=n_params),
        x0, method="SLSQP", bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    weights = np.clip(result.x, 0.0, 1.0)
    weights = weights / np.sum(weights) if np.sum(weights) > 0 else x0
    return {"weights": weights, "metrics": calculate_metrics(y, X @ weights, n_params=n_params)}


def _damped(k, freq, sigma2, phase):
    return np.exp(-2 * sigma2 * k**2) * np.sin(freq * k + phase) / np.where(k > 0, k, 1)


@pytest.fixture
def mixture():
    """Library {A, B, C} plus exp = 0.3*A(+3 eV) + 0.7*B(-2 eV) on k = 3..13."""
    library = {
        "A": (K_GRID, _damped(K_GRID, 4.4, 0.004, 0.3)),
        "B": (K_GRID, _damped(K_GRID, 5.8, 0.006, 1.1)),
        "C": (K_GRID, _damped(K_GRID, 7.1, 0.008, -0.4)),
    }
    k_exp = np.linspace(3.0, 13.0, 301)

    def make(e0_sign):
        chi = (0.3 * shifted_standard_on_exp_grid(k_exp, *library["A"], 3.0, e0_sign)
               + 0.7 * shifted_standard_on_exp_grid(k_exp, *library["B"], -2.0, e0_sign))
        return k_exp, chi

    return library, make


def _step1(library, k_exp, chi_exp, e0_sign=-1.0, metric="redchi"):
    from mlmd_exafs.lcf import determine_delta_e0_for_standard

    return {n: determine_delta_e0_for_standard(k_exp, chi_exp, *library[n], metric=metric,
                                               e0_sign=e0_sign)[0] for n in library}


def test_joint_recovers_synthetic_mixture(mixture):
    from mlmd_exafs.lcf import fit_lcf_combo

    library, make = mixture
    k_exp, chi_exp = make(-1.0)
    de0 = _step1(library, k_exp, chi_exp)
    fit = fit_lcf_combo(k_exp, chi_exp, library, ("A", "B"), de0, e0_mode="joint")
    assert fit["weights"] == pytest.approx([0.3, 0.7], abs=1e-3)
    assert fit["delta_e0"] == pytest.approx([3.0, -2.0], abs=0.1)
    assert fit["n_params"] == 3
    assert fit["mask"] == "bounds"
    assert fit["metrics"]["chi2"] < 1e-8
    assert np.all(np.isfinite(fit["weight_err"])) and np.all(np.isfinite(fit["delta_e0_err"]))


def test_fixed_mode_reproduces_legacy(mixture):
    from mlmd_exafs.lcf import fit_lcf_combo

    library, make = mixture
    k_exp, chi_exp = make(-1.0)
    de0 = _step1(library, k_exp, chi_exp)
    for n in (1, 2, 3):
        for combo in itertools.combinations(library, n):
            new = fit_lcf_combo(k_exp, chi_exp, library, combo, de0, e0_mode="fixed")
            old = _legacy_fit_lcf_combo(k_exp, chi_exp, library, combo, de0)
            np.testing.assert_array_equal(new["weights"], old["weights"])
            np.testing.assert_array_equal(new["delta_e0"], [de0[c] for c in combo])
            # chi2/rmsd/rfactor identical; redchi uses the corrected count (n - 1)
            for m in ("chi2", "rmsd", "rfactor", "n_points"):
                assert new["metrics"][m] == old["metrics"][m]
            n_pts = new["metrics"]["n_points"]
            assert new["n_params"] == n - 1
            assert new["metrics"]["redchi"] == pytest.approx(
                old["metrics"]["chi2"] / (n_pts - (n - 1)), rel=1e-12)
    # default of fit_lcf_combo stays "fixed"
    default = fit_lcf_combo(k_exp, chi_exp, library, ("A", "B"), de0)
    assert default["e0_mode"] == "fixed"


@pytest.mark.parametrize("metric", ["redchi", "chi2", "rmsd", "rfactor"])
def test_joint_never_worse_than_fixed(mixture, metric):
    from mlmd_exafs.lcf import fit_lcf_combo

    library, make = mixture
    k_exp, chi_exp = make(-1.0)
    de0 = _step1(library, k_exp, chi_exp, metric=metric)
    for n in (1, 2, 3):
        for combo in itertools.combinations(library, n):
            fixed = fit_lcf_combo(k_exp, chi_exp, library, combo, de0, metric=metric,
                                  e0_mode="fixed")
            joint = fit_lcf_combo(k_exp, chi_exp, library, combo, de0, metric=metric,
                                  e0_mode="joint")
            # same k points here, so the misfit itself is comparable
            assert joint["metrics"]["n_points"] == fixed["metrics"]["n_points"]
            assert joint["metrics"]["chi2"] <= fixed["metrics"]["chi2"] * (1 + 1e-12)


def test_weights_nonnegative_and_normalized(mixture):
    from mlmd_exafs.lcf import fit_lcf_combo

    library, make = mixture
    k_exp, chi_exp = make(-1.0)
    de0 = _step1(library, k_exp, chi_exp)
    for mode in ("joint", "joint_shared", "fixed"):
        for n in (1, 2, 3):
            for combo in itertools.combinations(library, n):
                fit = fit_lcf_combo(k_exp, chi_exp, library, combo, de0, e0_mode=mode)
                assert np.all(fit["weights"] >= 0)
                assert np.sum(fit["weights"]) == pytest.approx(1.0, abs=1e-12)
                if n == 1:
                    assert fit["weights"].tolist() == [1.0]
                    assert np.isnan(fit["weight_err"][0])
                assert np.all(fit["delta_e0"] >= -20.0) and np.all(fit["delta_e0"] <= 20.0)


@pytest.mark.parametrize("mode", ["joint", "joint_shared", "fixed"])
def test_e0_sign_flips_delta_e0_only(mixture, mode):
    from mlmd_exafs.lcf import fit_lcf_combo

    library, make = mixture
    fits = {}
    for sign in (-1.0, 1.0):
        # the "experiment" is the same spectrum; only the convention differs.
        # Noise keeps the minimum away from chi2 = 0, where metrics are
        # optimizer round-off only.
        k_exp, chi_exp = make(-1.0)
        chi_exp = chi_exp + 0.01 * np.random.default_rng(0).standard_normal(k_exp.size)
        de0 = _step1(library, k_exp, chi_exp, e0_sign=sign)
        fits[sign] = fit_lcf_combo(k_exp, chi_exp, library, ("A", "B"), de0,
                                   e0_sign=sign, e0_mode=mode)
    a, b = fits[-1.0], fits[1.0]
    np.testing.assert_allclose(b["delta_e0"], -a["delta_e0"], atol=1e-5)
    np.testing.assert_allclose(b["delta_e0_initial"], -a["delta_e0_initial"], atol=1e-6)
    np.testing.assert_allclose(b["weights"], a["weights"], atol=1e-6)
    for m in ("chi2", "redchi", "rmsd", "rfactor"):
        assert b["metrics"][m] == pytest.approx(a["metrics"][m], rel=1e-6, abs=1e-14)


def test_joint_shared_uses_one_delta_e0(mixture):
    from mlmd_exafs.lcf import fit_lcf_combo

    library, make = mixture
    k_exp, chi_exp = make(-1.0)
    de0 = _step1(library, k_exp, chi_exp)
    for combo in (("A", "B"), ("A", "B", "C")):
        fit = fit_lcf_combo(k_exp, chi_exp, library, combo, de0, e0_mode="joint_shared")
        assert np.all(fit["delta_e0"] == fit["delta_e0"][0])
        assert fit["n_params"] == len(combo)


def test_joint_falls_back_to_initial_mask():
    from mlmd_exafs.lcf import fit_lcf_combo

    # standard ends at k = 13.2: a -20 eV shift (k_query^2 = k^2 + 5.2) leaves
    # its range at the top of the fit window, so the bounds mask is too small
    k_std = np.round(np.arange(0.0, 13.2 + 1e-9, 0.05), 4)
    library = {"A": (k_std, _damped(k_std, 4.4, 0.004, 0.3))}
    k_exp = np.linspace(3.0, 13.0, 201)
    chi_exp = shifted_standard_on_exp_grid(k_exp, *library["A"], 1.0)
    fit = fit_lcf_combo(k_exp, chi_exp, library, ("A",), {"A": 0.0}, e0_mode="joint",
                        min_valid_frac=1.0)
    assert fit["mask"] == "initial"
    assert fit["metrics"]["n_points"] == k_exp.size
    assert fit["delta_e0"][0] == pytest.approx(1.0, abs=0.05)


def test_run_lcf_e0_modes_and_outputs(tmp_path, standards):
    exp = _mix_exp(tmp_path, standards, (0.6, 0.4))
    stds = [str(p) for p in standards]
    res = {}
    for mode in ("joint", "joint_shared", "fixed"):
        res[mode] = run_lcf(str(exp), stds, outdir=str(tmp_path / mode), max_components=2,
                            e0_mode=mode, verbose=False)
        best = res[mode]["best"]
        assert res[mode]["e0_mode"] == best["e0_mode"] == mode
        assert len(best["delta_e0_initial_eV"]) == len(best["standards"])
        header = (tmp_path / mode / "best_lcf_fit.dat").read_text().splitlines()[:9]
        assert any(line == f"# e0_mode: {mode}" for line in header)
        assert any(line.startswith("# delta_e0_initial_eV: ") for line in header)
    joint = res["joint"]["best"]
    assert joint["weights"] == pytest.approx([0.6, 0.4], abs=1e-3)
    assert joint["delta_e0_eV"] == pytest.approx([1.5, -1.0], abs=0.1)
    assert joint["n_params"] == 3
    with pytest.raises(ValueError, match="e0_mode"):
        run_lcf(str(exp), stds, outdir=str(tmp_path / "x"), e0_mode="free", verbose=False)


def test_best_fit_curve_uses_refined_delta_e0(tmp_path, standards):
    exp = _mix_exp(tmp_path, standards, (0.6, 0.4))
    res = run_lcf(str(exp), [str(p) for p in standards], outdir=str(tmp_path / "o"),
                  max_components=2, e0_mode="joint", verbose=False)
    best = res["best"]
    assert best["delta_e0_eV"] != best["delta_e0_initial_eV"]
    data = np.loadtxt(tmp_path / "o" / "best_lcf_fit.dat")
    k, fitted = data[:, 0], data[:, 2]
    expected = sum(
        w * shifted_standard_on_exp_grid(k, *np.loadtxt(p)[:, :2].T, e)
        for p, w, e in zip(standards, best["weights"], best["delta_e0_eV"])
    )
    np.testing.assert_allclose(fitted, expected, atol=1e-9)

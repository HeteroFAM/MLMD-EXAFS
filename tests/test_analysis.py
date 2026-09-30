import numpy as np
import pytest

from mlmd_exafs.analysis import (
    _extract_chi,
    average_chi,
    hanning_window,
    plot_chi,
    plot_convergence,
    xftf,
)

from conftest import K_GRID, synthetic_chi


def test_extract_chi_prepends_k0(tmp_path, chi_file_factory):
    k = K_GRID[1:]  # FEFF grids start at 0.05
    path = chi_file_factory(tmp_path / "chi.dat", k, synthetic_chi(k))
    k_out, chi_out = _extract_chi(path)
    assert k_out[0] == 0.0 and np.isnan(chi_out[0])
    assert len(k_out) == len(K_GRID)
    assert np.allclose(chi_out[1:], synthetic_chi(k), rtol=1e-5, atol=1e-12)


def test_extract_chi_without_header(tmp_path):
    path = tmp_path / "chi.dat"
    path.write_text("0.0 1.0\n0.05 2.0\n")
    assert _extract_chi(path) is None


def test_average_chi_statistics(exafs_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (exafs_dir / "999999_0").mkdir()  # no chi.dat
    bad = exafs_dir / "888888_0"
    bad.mkdir()
    (bad / "chi.dat").write_text("no header here\n")

    result = average_chi(str(exafs_dir), "avg")

    assert result["n_samples"] == 4
    assert result["skipped"] == 2
    assert result["output_file"] == "avg-chi_avg.dat"

    stack = np.array([synthetic_chi(K_GRID, amp=a) for a in (0.8, 0.9, 1.1, 1.2)])
    mask = K_GRID > 0
    assert np.allclose(result["chi_avg"][mask], stack.mean(0)[mask], rtol=1e-5, atol=1e-10)
    assert np.allclose(result["chi_std"][mask], stack.std(0)[mask], rtol=1e-4, atol=1e-10)
    assert np.allclose(result["chi_sem"], result["chi_std"] / 2, equal_nan=True)

    data = np.loadtxt(tmp_path / "avg-chi_avg.dat")
    assert data.shape == (len(K_GRID), 4)


def test_average_chi_empty_dir(tmp_path):
    (tmp_path / "exafs_empty").mkdir()
    result = average_chi(str(tmp_path / "exafs_empty"), str(tmp_path / "x"))
    assert result["n_samples"] == 0
    assert result["output_file"] == ""


def test_hanning_window():
    k = np.array([0.0, 1.5, 2.0, 2.5, 5.0, 10.5, 11.0, 11.5, 15.0])
    w = hanning_window(k, kmin=2.0, kmax=11.0, dk_win=1.0)
    assert w[0] == 0.0 and w[-1] == 0.0
    assert w[1] == 0.0 and w[7] == 0.0
    assert w[2] == pytest.approx(0.5) and w[6] == pytest.approx(0.5)
    assert w[3] == 1.0 and w[4] == 1.0 and w[5] == 1.0


def test_xftf_peak_position():
    r0 = 2.5
    chi = np.sin(2 * K_GRID * r0) / np.where(K_GRID > 0, K_GRID**2, 1)
    r, mag = xftf(K_GRID, chi, kmin=2.0, kmax=14.0, kweight=2, nfft=2048)
    assert len(r) == len(mag) == 1024
    assert r[np.argmax(mag)] == pytest.approx(r0, abs=0.05)


def test_xftf_ignores_nan():
    chi = synthetic_chi(K_GRID)
    chi[0] = np.nan
    _, mag = xftf(K_GRID, chi)
    assert np.all(np.isfinite(mag))


def _avg_file(tmp_path):
    chi = synthetic_chi(K_GRID)
    data = np.column_stack([K_GRID, chi, np.abs(chi) * 0.1, np.abs(chi) * 0.05])
    path = tmp_path / "s-chi_avg.dat"
    np.savetxt(path, data, header="k chi_avg chi_std chi_sem")
    return path


@pytest.mark.parametrize("band", ["sem", "std", "none"])
def test_plot_chi_writes_png(tmp_path, band):
    result = plot_chi(str(_avg_file(tmp_path)), str(tmp_path / "plot"), band=band, n_samples=4)
    assert result == {"output_file": str(tmp_path / "plot.png"), "k_weight": 2, "band": band}
    assert (tmp_path / "plot.png").stat().st_size > 0


def test_plot_chi_bad_band(tmp_path):
    with pytest.raises(ValueError, match="band"):
        plot_chi(str(_avg_file(tmp_path)), str(tmp_path / "plot"), band="bogus")


def test_plot_chi_needs_two_columns(tmp_path):
    path = tmp_path / "one.dat"
    np.savetxt(path, np.column_stack([K_GRID]))
    with pytest.raises(ValueError, match="2 columns"):
        plot_chi(str(path), str(tmp_path / "plot"))


def test_plot_convergence(exafs_dir, tmp_path):
    result = plot_convergence(str(exafs_dir), str(tmp_path / "conv"), step=2)
    assert result["output_file"] == str(tmp_path / "conv.png")
    assert result["n_samples"] == 4
    assert (tmp_path / "conv.png").is_file()

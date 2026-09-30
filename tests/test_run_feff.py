import pytest

from mlmd_exafs.run_feff import run_feff_batch


def _job_dirs(root, n=3):
    root.mkdir()
    for i in range(n):
        d = root / f"{i:0>6}_0"
        d.mkdir()
        (d / "feff.inp").write_text("END\n")
    (root / "not_a_job").mkdir()
    return root


def test_missing_binary_raises(tmp_path):
    root = _job_dirs(tmp_path / "exafs_x")
    with pytest.raises(FileNotFoundError, match="FEFF binary not found"):
        run_feff_batch(str(root), feff_bin=str(tmp_path / "nope"))


def test_no_jobs_raises(tmp_path, fake_feff_bin):
    (tmp_path / "exafs_empty").mkdir()
    with pytest.raises(ValueError, match="No feff.inp"):
        run_feff_batch(str(tmp_path / "exafs_empty"), feff_bin=str(fake_feff_bin))


def test_batch_runs_and_cleans_up(tmp_path, fake_feff_bin):
    root = _job_dirs(tmp_path / "exafs_Cu")
    result = run_feff_batch(str(root), feff_bin=str(fake_feff_bin), max_workers=2)

    assert result["n_jobs"] == 3
    assert result["n_ok"] == 3
    assert result["failed"] == []
    assert result["cleanup"]["deleted_files"] == 3
    for d in sorted(root.glob("0*_0")):
        assert sorted(p.name for p in d.iterdir()) == ["chi.dat", "feff.inp", "feff.out"]
        assert "fake feff done" in (d / "feff.out").read_text()


def test_batch_no_cleanup_keeps_scratch(tmp_path, fake_feff_bin):
    root = _job_dirs(tmp_path / "exafs_Cu", n=1)
    result = run_feff_batch(str(root), feff_bin=str(fake_feff_bin), cleanup=False)
    assert result["cleanup"] is None
    assert (root / "000000_0" / "pot.bin").is_file()


def test_failed_jobs_reported(tmp_path, failing_feff_bin):
    root = _job_dirs(tmp_path / "exafs_Cu", n=2)
    result = run_feff_batch(str(root), feff_bin=str(failing_feff_bin), cleanup=False)
    assert result["n_ok"] == 0
    assert result["n_failed"] == 2
    assert sorted(result["failed"]) == [str(root / "000000_0"), str(root / "000001_0")]

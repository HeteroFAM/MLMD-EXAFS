import pytest

from mlmd_exafs.cleanup_exafs import DEFAULT_KEEP, cleanup_exafs, find_exafs_roots


def _remaining(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def test_default_keep():
    assert DEFAULT_KEEP == {"feff.inp", "feff.out", "chi.dat", "neighborhoods_*.xyz"}


def test_cleanup_keeps_only_default_files(exafs_dir):
    result = cleanup_exafs(str(exafs_dir))
    assert result["exafs_dirs"] == [str(exafs_dir.resolve())]
    assert result["deleted_files"] == 8  # pot.bin + log1.dat in 4 subdirs
    assert result["deleted_dirs"] == 0
    remaining = _remaining(exafs_dir)
    assert "neighborhoods_0.xyz" in remaining
    assert not any(r.endswith(("pot.bin", "log1.dat")) for r in remaining)
    assert sum(r.endswith("chi.dat") for r in remaining) == 4


def test_dry_run_deletes_nothing(exafs_dir):
    before = _remaining(exafs_dir)
    result = cleanup_exafs(str(exafs_dir), dry_run=True)
    assert result["dry_run"] is True
    assert result["deleted_files"] == 8
    assert _remaining(exafs_dir) == before


def test_custom_keep_and_remove_empty_dirs(tmp_path):
    root = tmp_path / "exafs_a"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "junk.bin").write_text("x")
    (root / "keep.me").write_text("x")
    result = cleanup_exafs(str(tmp_path), keep=["*.me"], remove_empty_dirs=True)
    assert result["deleted_files"] == 1
    assert result["deleted_dirs"] == 1
    assert not (root / "sub").exists()
    assert (root / "keep.me").is_file()


def test_only_exafs_dirs_touched_and_nested_not_double_counted(tmp_path):
    other = tmp_path / "md_out"
    other.mkdir()
    (other / "pot.bin").write_text("x")
    nested = tmp_path / "exafs_top" / "exafs_inner"
    nested.mkdir(parents=True)
    (nested / "pot.bin").write_text("x")

    assert find_exafs_roots(tmp_path) == [tmp_path / "exafs_top"]
    result = cleanup_exafs(str(tmp_path))
    assert result["deleted_files"] == 1
    assert (other / "pot.bin").is_file()


def test_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        cleanup_exafs(str(tmp_path / "missing"))

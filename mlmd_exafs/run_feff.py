"""Batch FEFF execution over generated input directories.

Runs the FEFF binary in every subdirectory of a FEFF output directory that
contains a ``feff.inp``, with bounded parallelism. Each job runs inside its own
directory (FEFF writes its outputs to the working directory) and its stdout is
captured to ``feff.out``. Afterwards the FEFF scratch files are removed with
:func:`mlmd_exafs.cleanup_exafs.cleanup_exafs`, keeping only ``feff.inp``,
``feff.out``, ``chi.dat`` (and the ``neighborhoods_*.xyz`` inspection file).
"""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .cleanup_exafs import cleanup_exafs

DEFAULT_FEFF_BIN = "/share/feff/feff90_binaries/feff.x"


def _run_one(job_dir: Path, feff_bin: str) -> tuple[Path, int]:
    with open(job_dir / "feff.out", "w") as out:
        proc = subprocess.run(
            [feff_bin, "feff.inp"],
            cwd=str(job_dir),
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    return job_dir, proc.returncode


def run_feff_batch(
    directory: str,
    feff_bin: str = DEFAULT_FEFF_BIN,
    max_workers: int = 32,
    cleanup: bool = True,
) -> dict[str, Any]:
    """Run FEFF in every ``feff.inp`` subdirectory of ``directory``.

    Parameters
    ----------
    directory : str
        FEFF output directory produced by
        :func:`mlmd_exafs.feff.generate_feff_inputs_from_trajectory`.
    feff_bin : str
        Path to the FEFF executable.
    max_workers : int
        Maximum number of concurrent FEFF jobs.
    cleanup : bool
        Remove FEFF scratch files afterwards, keeping only the files needed
        downstream (see :mod:`mlmd_exafs.cleanup_exafs`). Only affects
        ``exafs_*`` directories at or below ``directory``.

    Returns
    -------
    dict
        ``n_jobs``, ``n_ok``, ``n_failed``, ``failed`` (list of dirs), and
        ``cleanup`` (summary dict, or None when skipped).
    """
    feff_dir = Path(directory)
    if not Path(feff_bin).is_file():
        raise FileNotFoundError(
            f"FEFF binary not found at {feff_bin!r}. Pass --feff-bin with the "
            "correct path."
        )

    jobs = [
        d for d in sorted(feff_dir.iterdir()) if d.is_dir() and (d / "feff.inp").is_file()
    ]
    if not jobs:
        raise ValueError(f"No feff.inp subdirectories found under {directory}.")

    ok, failed = 0, []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_one, d, feff_bin): d for d in jobs}
        for fut in as_completed(futures):
            job_dir, rc = fut.result()
            if rc == 0:
                ok += 1
            else:
                failed.append(str(job_dir))
            print(f"[{ok + len(failed)}/{len(jobs)}] {job_dir.name} rc={rc}")

    cleanup_result = None
    if cleanup:
        cleanup_result = cleanup_exafs(str(feff_dir))
        print(
            f"Cleanup: deleted {cleanup_result['deleted_files']} FEFF scratch "
            f"files in {len(cleanup_result['exafs_dirs'])} exafs_* directories"
        )

    return {
        "n_jobs": len(jobs),
        "n_ok": ok,
        "n_failed": len(failed),
        "failed": failed,
        "cleanup": cleanup_result,
    }

#!/usr/bin/env python3
"""Clean EXAFS output directories by keeping only selected files.

This script recursively searches for directories whose names match
"exafs_*" starting from a root directory. Inside each matched directory tree,
it removes all files except a configurable keep-list. Keep-list entries are
basenames or shell-style patterns (e.g. ``neighborhoods_*.xyz``).

Used by :func:`mlmd_exafs.run_feff.run_feff_batch` after FEFF finishes, and
available standalone as ``mlmd-exafs cleanup`` or ``python -m
mlmd_exafs.cleanup_exafs``.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
from pathlib import Path
from typing import Any, Iterable


DEFAULT_KEEP = {"feff.inp", "feff.out", "chi.dat", "neighborhoods_*.xyz"}


def _is_kept(name: str, keep_names: set[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in keep_names)


def find_exafs_roots(search_root: Path) -> list[Path]:
    """Return top-level exafs_* directories found under search_root."""
    exafs_roots: list[Path] = []

    for dirpath, dirnames, _ in os.walk(search_root, topdown=True):
        current = Path(dirpath)
        if fnmatch.fnmatch(current.name, "exafs_*"):
            exafs_roots.append(current)
            # No need to discover nested exafs_*; this root will be handled recursively.
            dirnames[:] = []

    return exafs_roots


def cleanup_exafs_root(
    exafs_root: Path,
    keep_names: set[str],
    dry_run: bool,
    remove_empty_dirs: bool,
) -> tuple[int, int]:
    """Clean one exafs tree and return (deleted_files, deleted_dirs)."""
    deleted_files = 0
    deleted_dirs = 0

    for dirpath, dirnames, filenames in os.walk(exafs_root, topdown=False):
        current = Path(dirpath)

        for name in filenames:
            file_path = current / name
            if _is_kept(name, keep_names):
                continue

            if dry_run:
                print(f"[DRY-RUN] delete file: {file_path}")
            else:
                try:
                    file_path.unlink()
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    print(f"[WARN] could not delete file {file_path}: {exc}")
                    continue
            deleted_files += 1

        if remove_empty_dirs:
            for dirname in dirnames:
                child_dir = current / dirname
                if not child_dir.exists():
                    continue
                try:
                    is_empty = not any(child_dir.iterdir())
                except OSError:
                    continue
                if not is_empty:
                    continue

                if dry_run:
                    print(f"[DRY-RUN] delete empty dir: {child_dir}")
                else:
                    try:
                        child_dir.rmdir()
                    except OSError as exc:
                        print(f"[WARN] could not delete dir {child_dir}: {exc}")
                        continue
                deleted_dirs += 1

    return deleted_files, deleted_dirs


def cleanup_exafs(
    root: str,
    keep: Iterable[str] | None = None,
    dry_run: bool = False,
    remove_empty_dirs: bool = False,
) -> dict[str, Any]:
    """Clean every exafs_* tree under ``root`` (``root`` itself may be one).

    Parameters
    ----------
    root : str
        Directory to search from. Passing a FEFF output directory produced by
        :func:`mlmd_exafs.feff.generate_feff_inputs_from_trajectory` (named
        ``exafs_*``) cleans only that directory.
    keep : iterable of str, optional
        Basenames or shell-style patterns to keep (default :data:`DEFAULT_KEEP`).
    dry_run : bool
        Report what would be deleted without removing anything.
    remove_empty_dirs : bool
        Also remove directories left empty after file cleanup.

    Returns
    -------
    dict
        ``exafs_dirs``, ``deleted_files``, ``deleted_dirs``, ``dry_run``.
    """
    search_root = Path(root).resolve()
    if not search_root.exists():
        raise FileNotFoundError(f"Search root does not exist: {search_root}")

    keep_names = set(keep) if keep is not None else set(DEFAULT_KEEP)
    exafs_roots = find_exafs_roots(search_root)

    total_deleted_files = 0
    total_deleted_dirs = 0
    for exafs_root in exafs_roots:
        deleted_files, deleted_dirs = cleanup_exafs_root(
            exafs_root=exafs_root,
            keep_names=keep_names,
            dry_run=dry_run,
            remove_empty_dirs=remove_empty_dirs,
        )
        total_deleted_files += deleted_files
        total_deleted_dirs += deleted_dirs

    return {
        "exafs_dirs": [str(d) for d in exafs_roots],
        "deleted_files": total_deleted_files,
        "deleted_dirs": total_deleted_dirs,
        "dry_run": dry_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively find exafs_* directories and delete all files except "
            "the keep-list (default: feff.inp, feff.out, chi.dat, "
            "neighborhoods_*.xyz)."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Directory to search from (default: current directory).",
    )
    parser.add_argument(
        "--keep",
        nargs="+",
        default=sorted(DEFAULT_KEEP),
        help="Basenames or shell-style patterns to keep inside exafs_* trees.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without removing files.",
    )
    parser.add_argument(
        "--remove-empty-dirs",
        action="store_true",
        help="Also remove empty directories left behind after file cleanup.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Search root does not exist: {root}")

    result = cleanup_exafs(
        str(root),
        keep=args.keep,
        dry_run=args.dry_run,
        remove_empty_dirs=args.remove_empty_dirs,
    )

    if not result["exafs_dirs"]:
        print(f"No exafs_* directories found under: {root}")
        return

    print(f"Found {len(result['exafs_dirs'])} exafs_* directories under: {root}")

    mode = "Would delete" if args.dry_run else "Deleted"
    print(f"{mode} {result['deleted_files']} files.")
    if args.remove_empty_dirs:
        print(f"{mode} {result['deleted_dirs']} empty directories.")

if __name__ == "__main__":
    main()
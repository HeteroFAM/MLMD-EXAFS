"""Human-readable summaries of ``lcf_results.csv`` files.

``run_lcf`` calls both writers on its own output directory. They also accept
several LCF output directories (e.g. one per MD model) to compare runs side by
side::

    python -m mlmd_exafs.lcf_summary lcf_model_a lcf_model_b -o summaries/

* :func:`write_top_fits` - best (rank-1) fit of each run, one column per run.
* :func:`write_top_candidates` - the top ``n`` ranked fits of each run, one row
  per candidate and one column per standard.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Iterable

FIT_ROWS = ("rfactor", "redchi", "deltaE0", "n_components")


def standard_label(filename: str) -> str:
    """Standard name without directory, extension and a trailing ``-chi_avg``."""
    return re.sub(r"(-chi_avg)?\.(dat|csv)$", "", Path(filename).name)


def _num(text) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def _read_rows(outdir: Path) -> list[dict]:
    with open(Path(outdir) / "lcf_results.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return sorted(rows, key=lambda r: int(r["rank"]))


def _components(row: dict) -> list[dict]:
    """Fitted components of one result row: label, weight, weight error, deltaE0, error."""
    comps, i = [], 1
    while f"std_{i}" in row:
        if row[f"std_{i}"]:
            comps.append({
                "label": standard_label(row[f"std_{i}"]),
                "weight": _num(row[f"weight_{i}"]),
                "weight_err": _num(row.get(f"weight_err_{i}")),
                "e0": _num(row.get(f"delta_e0_eV_{i}")),
                "e0_err": _num(row.get(f"delta_e0_err_eV_{i}")),
            })
        i += 1
    return comps


def _pm(val: float, err: float, fmt: str = ".2f") -> str:
    if math.isnan(val):
        return ""
    return f"{val:{fmt}}" if math.isnan(err) else f"{val:{fmt}} ± {err:{fmt}}"


def _e0_text(comps: list[dict], fmt: str = ".2f") -> str:
    """One shifted E0 (with error) if shared by all components, else all values."""
    values = [c["e0"] for c in comps if not math.isnan(c["e0"])]
    if not values:
        return ""
    if max(values) - min(values) < 1e-6:
        return _pm(comps[0]["e0"], comps[0]["e0_err"], fmt)
    return " | ".join(format(v, fmt) for v in values)


def _label(outdir: Path) -> str:
    return Path(outdir).resolve().name


def _available(outdir: Path) -> list[str]:
    path = Path(outdir) / "delta_e0.csv"
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return [standard_label(r["standard"]) for r in csv.DictReader(f)]


def write_top_fits(outdirs: Iterable[str | Path], out: str | Path) -> str:
    """Write the rank-1 fit of each LCF output directory, one column per directory.

    Rows: rfactor, redchi, deltaE0, n_components, then one row per standard
    (``weight ± error``). A standard in the run's library but not in its best
    fit shows ``none``; one not in the library at all shows ``n/a``.
    """
    cols, avail, fitted = {}, {}, {}
    for d in outdirs:
        best = _read_rows(Path(d))[0]
        comps = _components(best)
        name = _label(d)
        cols[name] = {
            "rfactor": f"{_num(best['rfactor']):.3f}",
            "redchi": f"{_num(best['redchi']):.4f}",
            "deltaE0": _e0_text(comps),
            "n_components": str(int(best["n_components"])),
        }
        for c in comps:
            cols[name][c["label"]] = _pm(c["weight"], c["weight_err"])
        avail[name] = set(_available(d)) | {c["label"] for c in comps}
        fitted[name] = {c["label"] for c in comps}

    standards = sorted(set().union(*avail.values())) if avail else []
    rows = [*FIT_ROWS, *standards]
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", *cols])
        for r in rows:
            writer.writerow([r, *(
                cols[m].get(r, "none" if r in avail[m] else "n/a") for m in cols)])
    return str(out)


def write_top_candidates(outdirs: Iterable[str | Path], out: str | Path, n: int = 10) -> str:
    """Write the top ``n`` ranked fits of each LCF output directory, one row per fit.

    Columns: model (directory name), rank, n_components, rfactor, redchi,
    deltaE0, then one column per standard (``weight ± error``; blank when the
    standard is not part of that fit).
    """
    rows, standards = [], []
    for d in outdirs:
        for r in _read_rows(Path(d))[:n]:
            comps = _components(r)
            row = {
                "model": _label(d),
                "rank": r["rank"],
                "n_components": r["n_components"],
                "rfactor": f"{_num(r['rfactor']):.4g}",
                "redchi": f"{_num(r['redchi']):.4g}",
                "deltaE0": _e0_text(comps, ".4g"),
            }
            for c in comps:
                if c["label"] not in standards:
                    standards.append(c["label"])
                row[c["label"]] = _pm(c["weight"], c["weight_err"], ".4g")
            rows.append(row)

    header = ["model", "rank", "n_components", "rfactor", "redchi", "deltaE0", *standards]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return str(out)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Summarize one or more LCF output directories (each holding lcf_results.csv).")
    p.add_argument("outdirs", nargs="+", help="LCF output directories, one column/model each.")
    p.add_argument("-o", "--out-dir", default=".", help="Where to write the two CSV files.")
    p.add_argument("-n", "--n-top", type=int, default=10, help="Candidates per directory.")
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"wrote {write_top_fits(args.outdirs, out / 'top_fits.csv')}")
    print(f"wrote {write_top_candidates(args.outdirs, out / f'top_{args.n_top}_candidates.csv', args.n_top)}")


if __name__ == "__main__":
    main()

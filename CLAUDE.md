# CLAUDE.md

MLMD-EXAFS computes theoretical EXAFS χ(k) spectra. It runs MLIP molecular dynamics on a structure, sends the sampled MD snapshots through FEFF9, and averages the χ(k) results. It is pure Python (ASE, numpy, scipy, matplotlib).

## Pipeline (one stage per CLI subcommand)

relax, then md, then feff-input, then run-feff (cleans up automatically), then average (optional E0 fit), then plot and convergence. lcf is optional: it fits several averaged χ(k) spectra to an experimental spectrum.

## Layout

| File | Role |
|------|------|
| `mlmd_exafs/calculators.py` | `build_calculator(backend, device, model, head, modal, checkpoint)`, `BACKENDS` tuple. Backend imports are lazy (chgnet, mace, uma, orb, sevennet). |
| `mlmd_exafs/md.py` | `relax`, `md_engine`, `run_md`. `run_md` writes `<save_dir>/<stem>/<stem>.traj` (via `<stem>.partial.traj`, renamed on completion) and skips the run if that file already exists. |
| `mlmd_exafs/feff.py` | `carve_out`, `generate_feff_inputs_from_trajectory`. Writes `exafs_<El>_hole<h>_de_<vrcorr>_s02_<x.x>_rc_<rmax>/<frame:06>_<atom>/feff.inp`. The carve radius is rmax + 2.5 Å. |
| `mlmd_exafs/run_feff.py` | `run_feff_batch`, `DEFAULT_FEFF_BIN=/share/feff/feff90_binaries/feff.x` |
| `mlmd_exafs/cleanup_exafs.py` | `cleanup_exafs`. Keeps feff.inp, feff.out, and chi.dat. |
| `mlmd_exafs/analysis.py` | `average_chi` (writes `<savefile>-chi_avg.dat`), `xftf`, `plot_chi`, `plot_convergence` |
| `mlmd_exafs/fitting_E0.py` | `fit_e0`. Reads experimental files in `.dat` or `.csv` format. Also has a legacy argparse `main`. |
| `mlmd_exafs/lcf.py` | `run_lcf` |
| `mlmd_exafs/cli.py` | `mlmd-exafs` entry point. Uses `_cmd_*` handlers, `build_parser`, and the shared `_add_backend_args` / `_add_e0_fit_args` helpers. |
| `mlmd_exafs/mcp_server.py` | `mlmd-exafs-mcp` entry point. Exposes 10 `mlmd_*` tools and supports both the mcp 1.x FastMCP and 2.x MCPServer APIs. |
| `tool/mlmd_exafs_tool.py` | SciLink plug-in: wrapper functions, a `tool_schemas` list (JSON schemas), and a `create_tool_functions(data_path, output_dir)` factory |
| `tests/` | pytest suite with one file per module, plus `test_cli.py` and `test_interfaces.py`. Shared fixtures are in `conftest.py`. |
| `scripts/run_feff.csh` | csh batch alternative to run-feff for clusters |
| `README.md`, `TUTORIAL.md`, `docs/*.md`, `tool/README.md` | User docs. `docs/` covers EXAFS background, FEFF parameter advice, and FEFF9 setup. |

## Key rule: three interfaces over one package

The CLI, the MCP server, and the SciLink tool are thin wrappers around the same package functions. Any change to a stage's parameters or behavior must be made in all of these places:
1. the core function in `mlmd_exafs/*.py`
2. `cli.py` (argparse args and the `_cmd_*` handler)
3. `mcp_server.py` (tool signature and docstring)
4. `tool/mlmd_exafs_tool.py` (wrapper function and the matching `tool_schemas` entry)
5. docs: `README.md`, `TUTORIAL.md`, and `tool/README.md` where relevant

`tests/test_interfaces.py` fails when the stage names or parameters drift apart across the three layers. The output-location parameters differ on purpose; those differences are listed in `OUTPUT_PARAM_DIFFS`.

## Conventions

- Public functions return JSON-serializable dicts, and the tool wrappers add `"status"`. They use numpy-style docstrings and `from __future__ import annotations`.
- MLIP backends have conflicting dependencies, so they are installed as extras (`pip install -e ".[mace]"`, etc.). Usually only one backend is installed at a time, so don't assume the others can be imported.
- FEFF9 is an external binary and may not be present. UMA models are gated on Hugging Face.
- Workflow outputs are gitignored (`md_out/`, `exafs_*/`, `*-chi_avg.dat`, `*_E0_fit/`, `*.png`, `*.traj`, `*.log`).

## Checking changes

Install the test dependencies with `pip install -e ".[test]"`, then run `pytest`. The suite takes about 10 s and runs in CI (`.github/workflows/tests.yml`, Python 3.10 to 3.12).

The tests run offline, with no MLIP backend and no FEFF9:
- The ASE `EMT` calculator stands in for the MLIPs. `test_cli.py` monkeypatches `mlmd_exafs.calculators.build_calculator`.
- A fake `feff.x` Python script writes a synthetic `chi.dat` (the `fake_feff_bin` fixture).
- Synthetic χ(k) comes from `synthetic_chi`, and the FEFF-format writer is `write_feff_chi`. Both are in `tests/conftest.py`.
- Tests write only inside `tmp_path`. `average_chi` writes relative to the CWD, so its tests use `monkeypatch.chdir`.
- The MCP import test skips when `mcp` is not installed.

A full real run still needs an installed MLIP backend and FEFF9.

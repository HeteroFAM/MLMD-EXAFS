# MLMD-EXAFS plug-in custom tool

`mlmd_exafs_tool.py` wraps the full MLMD-EXAFS pipeline as a **plug-in custom
tool**: a set of callables a host chat orchestrator can invoke during a
session, one tool per pipeline stage. The LLM picks them up when the
conversation calls for computing an EXAFS spectrum from a structure.

The heavy lifting stays in the installed `mlmd_exafs` package; this file is
only the thin wrapper that adapts each stage to the custom-tool contract and
returns JSON-serializable status dicts.

## The contract

A custom-tool file provides exactly two objects:

- **`tool_schemas`** — a list of JSON function schemas (what the LLM sees).
- **`create_tool_functions(data_path, output_dir)`** — a factory returning
  `{tool_name: callable}`.

The factory's first parameter is named `data_path`, so the host passes the
session's **active structure file path** (a string). `output_dir` is where the
host wants artifacts; the host auto-injects its session results directory.

## Installation

```bash
cd MLMD-EXAFS
pip install -e .              # installs the mlmd_exafs package
pip install -e ".[chgnet]"    # + at least one MLIP backend
```

FEFF9 is a separate binary (not a Python package) — install it independently
and pass its path to the `mlmd_run_feff` tool via `feff_bin`.

## Registering the tool

The registration mechanism depends on your host, but the shape is the same:
hand the host the path to `mlmd_exafs_tool.py`. Two common patterns:

**CLI-style flag**

```bash
<host> --tools MLMD-EXAFS/tool/mlmd_exafs_tool.py
```

**Programmatic**

```python
from mlmd_exafs_tool import tool_schemas, create_tool_functions

orchestrator.register_tools(tool_schemas, create_tool_functions)
# then drive the session, e.g. orchestrator.chat("...")
```

If your host follows the "first factory parameter decides the payload"
convention, the name `data_path` means it passes the active file as a path
string (not a loaded array).

## The tools

| Tool | Stage | Key arguments |
|------|-------|---------------|
| `mlmd_relax` | Cell + position relaxation | `backend`, `fmax` |
| `mlmd_md` | NVT molecular dynamics | `backend`, `temperature`, `n_steps`, `step_size` |
| `mlmd_feff_input` | Carve snapshots → `feff.inp` | `trajectory_path`, `target_atom`, `hole`, `rmax` |
| `mlmd_run_feff` | Batch FEFF execution | `directory`, `feff_bin`, `max_workers` |
| `mlmd_average_chi` | Average χ(k) | `directory`, `savefile` |
| `mlmd_plot` | k-weighted χ(k) + band | `chi_file`, `k_weight`, `band` |
| `mlmd_convergence` | k-/R-space convergence panels | `directory`, `step` |

Each tool returns a status dict with a `status` field and, where useful, a
`next_step` hint naming the following tool and the path to feed it — so the LLM
can chain the pipeline without the user threading paths between calls.

## Example session flow

```
> relax this structure with chgnet, run 300 K MD, then compute the Zn K-edge
> EXAFS for atom 0
```

The orchestrator will, in order:

1. `mlmd_relax(backend="chgnet")` → writes `relaxed.xyz`.
2. `mlmd_md(backend="chgnet", structure_path="…/relaxed.xyz", temperature=300)`
   → writes the trajectory.
3. `mlmd_feff_input(trajectory_path="…", target_atom=0, hole=1)` → FEFF inputs.
4. `mlmd_run_feff(directory="…")` → runs FEFF (needs the binary).
5. `mlmd_average_chi(directory="…")` → averaged χ(k).
6. `mlmd_plot(chi_file="…")` and `mlmd_convergence(directory="…")` → figures.

## Notes

- The MLIP backends have conflicting dependencies; use a separate environment
  per backend. Each backend is imported lazily, so only the one you select
  needs to be installed.
- The shaded band in the plots is the MD **sampling** dispersion (SD or SEM),
  not a systematic force-field error bar.
- For direct CLI use of the same pipeline (no host orchestrator), see the
  repository's top-level `README.md` and `TUTORIAL.md` — the `mlmd-exafs`
  command exposes the identical stages.
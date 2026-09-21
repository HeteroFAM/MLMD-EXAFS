# MLMD-EXAFS

Compute theoretical EXAFS χ(k) spectra from a crystal structure by sampling
thermal and structural disorder with machine-learning interatomic potential
(MLIP) molecular dynamics (MD) and computing scattering paths with FEFF9.

The pipeline:

```
cell relaxation → NVT molecular dynamics → FEFF input generation
   → batch FEFF → configurational averaging of χ(k) → k-/R-space plotting
```

Each MD snapshot is turned into a FEFF calculation; averaging the per-snapshot
χ(k) curves reproduces the thermal/static disorder that broadens and damps an
experimental EXAFS signal (the effect described by Debye–Waller factors in
path-based fitting).

## Supported MLIP backends

The MD force engine can be any of these universal potentials. Install at least
one; each is imported lazily so only the chosen backend must be present. Dependencies
for many of the MLIPs conflict, so separate environments are recommended.

| Backend | Best for |
|---------|----------|
| **CHGNet** | Magnetic systems (Fe, Co, Ni, Mn, Cr); predicts magnetic moments |
| **MACE** | High-accuracy forces, materials and organic systems |
| **UMA** (FAIRChem) | State-of-the-art accuracy; catalysis, MOFs, diverse chemistry |
| **ORB** | Fast inference, large/high-throughput systems |
| **SevenNet** | General materials, scalable |

**NOTE:** UMA models are gated. Access can be requested at
[https://huggingface.co/facebook/UMA](https://huggingface.co/facebook/UMA).

## Installation

```bash
cd MLMD-EXAFS
pip install -e .

# Then install the MLIP backend(s) you need, e.g.:
pip install -e ".[chgnet]"     # or .[mace] / .[uma] / .[orb] / .[sevennet]
```

**NOTE:** FEFF9 is a separate binary, not a Python package. Install it independently
from [https://feff.phys.washington.edu/feffproject-feff.html](https://feff.phys.washington.edu/feffproject-feff.html)
and point `mlmd-exafs run-feff --feff-bin` at the executable (default path:
`/share/feff/feff90_binaries/feff.x`).

## Quick start

```bash
# 1. Relax the cell
mlmd-exafs relax -i structure.cif -o relaxed.xyz --backend chgnet

# 2. Run NVT MD (writes md_out/relaxed/relaxed.xyz)
mlmd-exafs md -i relaxed.xyz -d md_out --backend chgnet --temperature 300

# 3. Generate FEFF inputs (Zn K-edge here; --target-atom is the absorber index)
mlmd-exafs feff-input -f md_out/relaxed/relaxed.xyz --target-atom 0 \
    --hole 1 --rmax 6.0

# 4. Run FEFF over all generated inputs
mlmd-exafs run-feff -d md_out/relaxed/exafs_Zn_hole1_de_0.0_s02_1.0_rc_6.0

# 5. Average χ(k) and plot
mlmd-exafs average -d md_out/relaxed/exafs_Zn_hole1_de_0.0_s02_1.0_rc_6.0 \
    --savefile exafs
mlmd-exafs plot --chi-file exafs-chi_avg.dat --savefile exafs_k2 --k-weight 2
mlmd-exafs convergence -d md_out/relaxed/exafs_Zn_hole1_de_0.0_s02_1.0_rc_6.0
```

See **[TUTORIAL.md](TUTORIAL.md)** for a full walkthrough with parameter
guidance for each stage.

## Using with AI agents & orchestrators

The same pipeline is exposed three ways so it works across different agent
frameworks. All three call the identical `mlmd_exafs` package functions — pick
the one that matches your orchestrator.

### 1. Command line (universal)

Any agent that can run shell commands — **Codex**, **Claude Code**, **Cline**,
**Cursor**, plain scripts — drives the workflow through the `mlmd-exafs` CLI.
No integration needed; point the agent at the commands in
[Quick start](#quick-start) above and [TUTORIAL.md](TUTORIAL.md). This is the
lowest-friction path and always works.

### 2. MCP server (Claude Code, Cline, Cursor, …)

MCP-capable clients can call each stage as a native tool via the bundled
Model Context Protocol server. Install the extra and register the command:

```bash
pip install -e ".[mcp]"    # installs the mcp SDK (works with mcp 1.x and 2.x)
```

```jsonc
// client MCP config (e.g. Claude Code / Cline / Cursor)
{
  "mcpServers": {
    "mlmd-exafs": { "command": "mlmd-exafs-mcp" }
  }
}
```

The server (`mlmd_exafs/mcp_server.py`, entry point `mlmd-exafs-mcp`) exposes
seven tools — `mlmd_relax`, `mlmd_md`, `mlmd_feff_input`, `mlmd_run_feff`,
`mlmd_average_chi`, `mlmd_plot`, `mlmd_convergence` — matching the CLI stages.

### 3. SciLink plug-in custom tool

For SciLink, register the plug-in in `tool/mlmd_exafs_tool.py`, which
follows SciLink's `tool_schemas` + `create_tool_functions` contract:

```bash
scilink simulate --tools MLMD-EXAFS/tool/mlmd_exafs_tool.py
```

It surfaces the same seven stages as SciLink custom tools, wired to the
session's active structure file and results directory. See
[tool/README.md](tool/README.md) for details.

> This SciLink plug-in format is specific to SciLink — other orchestrators do
> not read `tool_schemas` files. Use the CLI or MCP server for them.

## Package layout

| Module | Contents |
|--------|----------|
| `mlmd_exafs/calculators.py` | MLIP calculator factory (`build_calculator`) |
| `mlmd_exafs/md.py` | Cell relaxation (`relax`) and NVT MD (`run_md`, `md_engine`) |
| `mlmd_exafs/feff.py` | `carve_out`, `generate_feff_inputs_from_trajectory` |
| `mlmd_exafs/run_feff.py` | Batch FEFF execution (`run_feff_batch`) |
| `mlmd_exafs/analysis.py` | `average_chi`, `xftf`, `plot_chi`, `plot_convergence` |
| `mlmd_exafs/cli.py` | `mlmd-exafs` command-line entry point |
| `mlmd_exafs/mcp_server.py` | MCP server (`mlmd-exafs-mcp`) for MCP-capable agents |
| `tool/mlmd_exafs_tool.py` | SciLink plug-in custom tool |
| `scripts/run_feff.csh` | csh alternative to `run-feff` for cluster batch use |

Everything is also importable as a library — see the tutorial's "Python API"
section.

## A note on the sampling band

The shaded band in the plots is the snapshot-to-snapshot statistical
dispersion of the MD ensemble (standard deviation or standard error of the
mean). It shows how well converged the average is and how much the geometry
fluctuates thermally. It is not a systematic force-field error bar.

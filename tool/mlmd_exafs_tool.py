"""MLMD-EXAFS plug-in custom tool.

Exposes the MLIP molecular-dynamics + FEFF EXAFS workflow as a set of custom
tools that a host chat orchestrator can call during a session. Each pipeline
stage is one tool; together they cover:

    relax -> md -> feff-input -> run-feff (+ cleanup) -> average (+ E0 fit)
    -> plot / convergence, and LCF of several simulated spectra to experiment

Contract (matches the host's custom-tool convention):

- ``tool_schemas`` — the JSON schemas the LLM sees.
- ``create_tool_functions(data_path, output_dir)`` — factory returning the
  bound callables. ``data_path`` is the session's active structure file (a
  path string, because the parameter is named ``data_path``); ``output_dir``
  is where the host wants artifacts written.

The heavy lifting lives in the installed ``mlmd_exafs`` package; this file is
only the wrapper that adapts those functions to the tool contract, returning
JSON-serializable status dicts. Install the package first:

    pip install -e .            # from the MLMD-EXAFS repo root
    pip install -e ".[chgnet]"  # plus at least one MLIP backend

FEFF9 is a separate binary; point ``run_feff`` at it with ``feff_bin``.
"""

from __future__ import annotations

from pathlib import Path

from mlmd_exafs.analysis import average_chi, plot_chi, plot_convergence
from mlmd_exafs.calculators import BACKENDS, build_calculator
from mlmd_exafs.cleanup_exafs import cleanup_exafs
from mlmd_exafs.feff import generate_feff_inputs_from_trajectory
from mlmd_exafs.fitting_E0 import fit_e0
from mlmd_exafs.lcf import run_lcf
from mlmd_exafs.md import relax, run_md
from mlmd_exafs.run_feff import DEFAULT_FEFF_BIN, run_feff_batch


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _resolve(path: str | None, fallback: str | None) -> str:
    """Use an explicit path if given, else the session's active file."""
    chosen = path or fallback
    if not chosen:
        raise ValueError(
            "No structure/trajectory path provided and no active data file is "
            "set. Pass the path explicitly."
        )
    return chosen


def mlmd_relax(
    data_path: str,
    output_dir: str,
    backend: str,
    structure_path: str | None = None,
    device: str = "cpu",
    model: str | None = None,
    head: str = "omat",
    modal: str = "mpa",
    checkpoint: str | None = None,
    fmax: float = 0.05,
) -> dict:
    """Relax cell + positions with an MLIP; write relaxed.xyz to output_dir."""
    src = _resolve(structure_path, data_path)
    out = str(Path(output_dir) / "relaxed.xyz")
    calc = build_calculator(
        backend, device=device, model=model, head=head, modal=modal,
        checkpoint=checkpoint,
    )
    result = relax(src, out, calc, fmax=fmax)
    result["status"] = "success"
    result["next_step"] = (
        f"Run mlmd_md with structure_path='{out}' to sample a trajectory."
    )
    return result


def mlmd_md(
    data_path: str,
    output_dir: str,
    backend: str,
    structure_path: str | None = None,
    device: str = "cpu",
    model: str | None = None,
    head: str = "omat",
    modal: str = "mpa",
    checkpoint: str | None = None,
    temperature: float = 300.0,
    step_size: float = 10.0,
    n_steps: int = 11000,
) -> dict:
    """Run NVT molecular dynamics; write an ASE .traj trajectory under output_dir."""
    src = _resolve(structure_path, data_path)
    calc = build_calculator(
        backend, device=device, model=model, head=head, modal=modal,
        checkpoint=checkpoint,
    )
    result = run_md(
        src,
        calc,
        output_dir,
        temperature=temperature,
        step_size=step_size,
        n_steps=n_steps,
    )
    result["status"] = "success"
    result["next_step"] = (
        f"Run mlmd_feff_input with trajectory_path='{result['trajectory']}' and "
        "the absorbing-atom index."
    )
    return result


def mlmd_feff_input(
    trajectory_path: str,
    target_atom: int,
    hole: int = 1,
    rmax: float = 6.0,
    scf: str = "6.0 0 30 0.2 1",
    s02: float = 1.0,
    control: str = "1 1 1 1 1 1",
    corrections: str | None = None,
    step_size: int = 250,
    sampling_start: int = 0,
    cluster_buffer: float = 2.5,
) -> dict:
    """Carve snapshots and write feff.inp files for sampled trajectory frames."""
    result = generate_feff_inputs_from_trajectory(
        trajectory_path=trajectory_path,
        target_atom=target_atom,
        hole=hole,
        rmax=rmax,
        scf=scf,
        s02=s02,
        control=control,
        corrections=corrections,
        step_size=step_size,
        sampling_start=sampling_start,
        cluster_buffer=cluster_buffer,
    )
    result["status"] = "success"
    result["next_step"] = (
        f"Run mlmd_run_feff with directory='{result['output_dir']}'."
    )
    return result


def mlmd_run_feff(
    directory: str,
    feff_bin: str = DEFAULT_FEFF_BIN,
    max_workers: int = 32,
    cleanup: bool = True,
) -> dict:
    """Batch-execute FEFF in every feff.inp subdirectory, then clean scratch files."""
    result = run_feff_batch(
        directory, feff_bin=feff_bin, max_workers=max_workers, cleanup=cleanup
    )
    result["status"] = "success" if result["n_failed"] == 0 else "partial"
    result["next_step"] = (
        f"Run mlmd_average_chi with directory='{directory}' (pass exp_file if an "
        "experimental spectrum is available to fit E0)."
    )
    return result


def mlmd_cleanup(
    root: str,
    dry_run: bool = False,
    remove_empty_dirs: bool = False,
) -> dict:
    """Delete FEFF scratch files from exafs_* directories under ``root``."""
    result = cleanup_exafs(root, dry_run=dry_run, remove_empty_dirs=remove_empty_dirs)
    result["status"] = "success" if result["exafs_dirs"] else "no_data"
    return result


def mlmd_average_chi(
    directory: str,
    output_dir: str,
    savefile: str = "exafs",
    exp_file: str | None = None,
    exp_col2_is_k2chi: bool = False,
    kmin: float = 2.0,
    kmax: float = 12.0,
) -> dict:
    """Average all chi.dat files; fit E0 against experiment when exp_file given."""
    base = str(Path(output_dir) / savefile)
    result = average_chi(directory, base)
    # numpy arrays are not JSON-serializable; drop them from the returned dict
    for key in ("k", "chi_avg", "chi_std", "chi_sem"):
        result.pop(key, None)
    result["status"] = "success" if result["n_samples"] > 0 else "no_data"
    if exp_file and result["n_samples"] > 0:
        result["e0_fit"] = fit_e0(
            sim_file=result["output_file"],
            exp_file=exp_file,
            outdir=f"{base}_E0_fit",
            kmin=kmin,
            kmax=kmax,
            exp_col2_is_k2chi=exp_col2_is_k2chi,
        )
    result["next_step"] = (
        f"Run mlmd_plot with chi_file='{result['output_file']}'."
    )
    return result


def mlmd_fit_e0(
    chi_file: str,
    exp_file: str,
    output_dir: str,
    savefile: str = "exafs_E0_fit",
    exp_col2_is_k2chi: bool = False,
    kmin: float = 2.0,
    kmax: float = 12.0,
    e0_min: float = -10.0,
    e0_max: float = 10.0,
) -> dict:
    """Fit the E0 shift of averaged simulated chi(k) against experiment."""
    result = fit_e0(
        sim_file=chi_file,
        exp_file=exp_file,
        outdir=str(Path(output_dir) / savefile),
        kmin=kmin,
        kmax=kmax,
        exp_col2_is_k2chi=exp_col2_is_k2chi,
        e0_min=e0_min,
        e0_max=e0_max,
    )
    result["status"] = "success"
    return result


def mlmd_lcf(
    exp_file: str,
    standards: list[str],
    output_dir: str,
    savefile: str = "lcf_fit",
    metric: str = "redchi",
    max_components: int = 3,
    kmin: float | None = None,
    kmax: float | None = None,
    k_weight: int = 2,
    e0_min: float = -20.0,
    e0_max: float = 20.0,
    e0_sign: float = -1.0,
    scale_for_shift: bool = True,
    e0_mode: str = "joint",
    n_starts: int = 5,
    e0_start_step: float = 2.5,
) -> dict:
    """Linear combination fit of several simulated chi(k) spectra to experiment."""
    result = run_lcf(
        exp_file=exp_file,
        standards=standards,
        outdir=str(Path(output_dir) / savefile),
        metric=metric,
        max_components=max_components,
        kmin=kmin,
        kmax=kmax,
        k_weight=k_weight,
        e0_min=e0_min,
        e0_max=e0_max,
        e0_sign=e0_sign,
        scale_for_shift=scale_for_shift,
        e0_mode=e0_mode,
        n_starts=n_starts,
        e0_start_step=e0_start_step,
    )
    result["status"] = "success"
    return result


def mlmd_plot(
    chi_file: str,
    output_dir: str,
    savefile: str = "exafs_k2",
    k_weight: int = 2,
    band: str = "sem",
    n_samples: int | None = None,
) -> dict:
    """Plot k-weighted chi(k) with the MD sampling band shaded around the mean."""
    base = str(Path(output_dir) / savefile)
    result = plot_chi(
        chi_file, base, k_weight=k_weight, band=band, n_samples=n_samples
    )
    result["status"] = "success"
    return result


def mlmd_convergence(
    directory: str,
    output_dir: str,
    savefile: str = "exafs_convergence",
    step: int = 10,
    k_weight: int = 2,
    kmin: float = 2.0,
    kmax: float = 11.0,
) -> dict:
    """Plot k-space and R-space convergence versus number of averaged snapshots."""
    base = str(Path(output_dir) / savefile)
    result = plot_convergence(
        directory, base, step=step, k_weight=k_weight, kmin=kmin, kmax=kmax
    )
    result["status"] = "success"
    return result


# ---------------------------------------------------------------------------
# Tool schemas — what the LLM sees
# ---------------------------------------------------------------------------

_BACKEND_ENUM = list(BACKENDS)

tool_schemas = [
    {
        "type": "function",
        "function": {
            "name": "mlmd_relax",
            "description": (
                "Relax a crystal structure's atomic positions and cell with an "
                "MLIP (Frechet cell filter + FIRE). Required first step before "
                "molecular dynamics. Writes relaxed.xyz. Uses the session's "
                "active structure file unless structure_path is given."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "backend": {"type": "string", "enum": _BACKEND_ENUM},
                    "structure_path": {
                        "type": "string",
                        "description": "Structure file; defaults to the active data file.",
                    },
                    "device": {"type": "string", "enum": ["cpu", "cuda"]},
                    "model": {"type": "string", "description": "Pretrained model name (backend default if omitted)."},
                    "head": {"type": "string", "enum": ["oc20", "omat", "omol", "odac", "omc"]},
                    "modal": {"type": "string", "enum": ["mpa", "omat24"]},
                    "checkpoint": {
                        "type": "string",
                        "description": "Path to a local (e.g. fine-tuned) model checkpoint; replaces model.",
                    },
                    "fmax": {"type": "number", "description": "Force threshold eV/A (default 0.05)."},
                },
                "required": ["backend"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mlmd_md",
            "description": (
                "Run NVT molecular dynamics (Nose-Hoover chain) with an MLIP to "
                "generate a thermally sampled trajectory for EXAFS. Writes an "
                "ASE .traj trajectory. Point structure_path at the relaxed "
                "structure from mlmd_relax."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "backend": {"type": "string", "enum": _BACKEND_ENUM},
                    "structure_path": {
                        "type": "string",
                        "description": "Relaxed structure file; defaults to the active data file.",
                    },
                    "device": {"type": "string", "enum": ["cpu", "cuda"]},
                    "model": {"type": "string", "description": "Pretrained model name."},
                    "head": {"type": "string", "enum": ["oc20", "omat", "omol", "odac", "omc"]},
                    "modal": {"type": "string", "enum": ["mpa", "omat24"]},
                    "checkpoint": {
                        "type": "string",
                        "description": "Path to a local (e.g. fine-tuned) model checkpoint; replaces model.",
                    },
                    "temperature": {"type": "number", "description": "Temperature in K (default 300)."},
                    "step_size": {"type": "number", "description": "MD time step in a.u. (default 10)."},
                    "n_steps": {"type": "integer", "description": "Number of MD steps (default 11000)."},
                },
                "required": ["backend"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mlmd_feff_input",
            "description": (
                "Generate FEFF input files from an MD trajectory: carve the local "
                "environment around the absorbing atom in sampled frames and "
                "write a feff.inp per frame. Returns the output directory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "trajectory_path": {"type": "string", "description": "MD trajectory (traj/extxyz)."},
                    "target_atom": {"type": "integer", "description": "Index of the absorbing atom."},
                    "hole": {"type": "integer", "description": "HOLE: 1=K, 2=L1, 3=L2, 4=L3 (default 1)."},
                    "rmax": {"type": "number", "description": "FEFF RMAX cutoff in A (default 6.0)."},
                    "scf": {"type": "string", "description": 'SCF card (default "6.0 0 30 0.2 1").'},
                    "s02": {"type": "number", "description": "S0^2 amplitude factor (default 1.0)."},
                    "control": {"type": "string", "description": 'CONTROL card (default "1 1 1 1 1 1").'},
                    "corrections": {"type": "string", "description": 'CORRECTIONS "vrcorr vicorr", or omit.'},
                    "step_size": {"type": "integer", "description": "Sample every N-th frame (default 250)."},
                    "sampling_start": {"type": "integer", "description": "First frame to sample (default 0)."},
                    "cluster_buffer": {"type": "number", "description": "Carve radius beyond rmax in A (default 2.5). rmax + cluster_buffer must be >= the SCF radius."},
                },
                "required": ["trajectory_path", "target_atom"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mlmd_run_feff",
            "description": (
                "Batch-execute the FEFF binary in every feff.inp subdirectory of "
                "the given directory (bounded concurrency). Produces a chi.dat in "
                "each. Requires a FEFF9 executable."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "FEFF output dir from mlmd_feff_input."},
                    "feff_bin": {"type": "string", "description": f"Path to FEFF executable (default {DEFAULT_FEFF_BIN})."},
                    "max_workers": {"type": "integer", "description": "Max concurrent FEFF jobs (default 32)."},
                    "cleanup": {"type": "boolean", "description": "Delete FEFF scratch files afterwards, keeping feff.inp/feff.out/chi.dat (default true)."},
                },
                "required": ["directory"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mlmd_cleanup",
            "description": (
                "Delete FEFF scratch files from every exafs_* directory under "
                "root, keeping only feff.inp, feff.out, chi.dat and "
                "neighborhoods_*.xyz. mlmd_run_feff already does this by "
                "default; use for older runs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "root": {"type": "string", "description": "Directory to search (may itself be an exafs_* dir)."},
                    "dry_run": {"type": "boolean", "description": "Only report deletions (default false)."},
                    "remove_empty_dirs": {"type": "boolean", "description": "Also remove emptied directories (default false)."},
                },
                "required": ["root"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mlmd_average_chi",
            "description": (
                "Average all chi.dat files in a FEFF output directory into a "
                "converged chi(k) spectrum with per-k standard deviation and "
                "standard error of the mean (MD sampling band). Writes "
                "<savefile>-chi_avg.dat. If an experimental spectrum (exp_file) "
                "is available, also fits the E0 shift against it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "FEFF output directory."},
                    "savefile": {"type": "string", "description": "Base name for the averaged chi file (default 'exafs')."},
                    "exp_file": {"type": "string", "description": "Optional experimental chi(k) file (.dat or .csv) for E0 fitting."},
                    "exp_col2_is_k2chi": {"type": "boolean", "description": "Experimental column 2 is already k^2*chi (default false: chi)."},
                    "kmin": {"type": "number", "description": "E0 fit kmin in A^-1 (default 2.0)."},
                    "kmax": {"type": "number", "description": "E0 fit kmax in A^-1 (default 12.0)."},
                },
                "required": ["directory"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mlmd_fit_e0",
            "description": (
                "Fit the E0 energy shift that best aligns the averaged simulated "
                "chi(k) with an experimental spectrum (.dat or .csv) by "
                "minimizing the k^2*chi mean squared deviation over [kmin, kmax]. "
                "Writes the shifted spectrum, comparison table, summary and "
                "calc_vs_exp.png."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chi_file": {"type": "string", "description": "Averaged *-chi_avg.dat file."},
                    "exp_file": {"type": "string", "description": "Experimental chi(k) file (.dat or .csv)."},
                    "savefile": {"type": "string", "description": "Output subdirectory name (default 'exafs_E0_fit')."},
                    "exp_col2_is_k2chi": {"type": "boolean", "description": "Experimental column 2 is already k^2*chi (default false)."},
                    "kmin": {"type": "number", "description": "Fit kmin in A^-1 (default 2.0)."},
                    "kmax": {"type": "number", "description": "Fit kmax in A^-1 (default 12.0)."},
                    "e0_min": {"type": "number", "description": "E0 search min in eV (default -10)."},
                    "e0_max": {"type": "number", "description": "E0 search max in eV (default 10)."},
                },
                "required": ["chi_file", "exp_file"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mlmd_lcf",
            "description": (
                "Linear combination fit (LCF) of several simulated chi(k) "
                "spectra (e.g. averaged spectra of candidate structures) to an "
                "experimental spectrum (.dat or .csv). Fits an individual delta "
                "E0 per standard, then non-negative weights summing to 1 for "
                "every combination of up to max_components standards, ranked "
                "by metric. By default (e0_mode='joint') each combination's "
                "delta E0 values are refined together with its weights. Writes lcf_results.csv, delta_e0.csv, "
                "best_lcf_fit.dat, best_lcf_fit.png, best_lcf_fit_R.png and fit.log."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "exp_file": {"type": "string", "description": "Experimental chi(k) file (.dat or .csv)."},
                    "standards": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Simulated chi(k) files or glob patterns, e.g. ['sims/*-chi_avg.dat'].",
                    },
                    "savefile": {"type": "string", "description": "Output subdirectory name (default 'lcf_fit')."},
                    "metric": {"type": "string", "enum": ["redchi", "chi2", "rmsd", "rfactor"]},
                    "max_components": {"type": "integer", "description": "Max standards per combination (default 3)."},
                    "kmin": {"type": "number", "description": "Fit kmin in A^-1 (default: full range)."},
                    "kmax": {"type": "number", "description": "Fit kmax in A^-1 (default: full range)."},
                    "k_weight": {"type": "integer", "description": "k-weight exponent (default 2)."},
                    "e0_min": {"type": "number", "description": "Delta E0 lower bound in eV (default -20)."},
                    "e0_max": {"type": "number", "description": "Delta E0 upper bound in eV (default 20)."},
                    "e0_sign": {"type": "number", "enum": [1, -1], "description": "E0 sign convention (default -1 = Artemis/IFEFFIT, same as mlmd_fit_e0)."},
                    "scale_for_shift": {"type": "boolean", "description": "Optimize a temporary amplitude during each E0 search (default true)."},
                    "e0_mode": {
                        "type": "string",
                        "enum": ["joint", "joint_shared", "fixed"],
                        "description": (
                            "joint (default): one delta E0 per standard refined together with the weights; "
                            "joint_shared: one delta E0 shared by the combination; "
                            "fixed: per-standard delta E0 kept fixed (original behavior)."
                        ),
                    },
                    "n_starts": {"type": "integer", "description": "Starting points per joint fit (default 5)."},
                    "e0_start_step": {"type": "number", "description": "Delta E0 offset in eV between joint-fit starting points (default 2.5)."},
                },
                "required": ["exp_file", "standards"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mlmd_plot",
            "description": (
                "Plot k-weighted chi(k) from an averaged chi file, with the MD "
                "sampling band (SEM or SD) shaded around the mean. Writes a PNG."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chi_file": {"type": "string", "description": "Averaged *-chi_avg.dat file."},
                    "savefile": {"type": "string", "description": "Base name for the PNG (default 'exafs_k2')."},
                    "k_weight": {"type": "integer", "description": "k-weight exponent (default 2)."},
                    "band": {"type": "string", "enum": ["sem", "std", "none"]},
                    "n_samples": {"type": "integer", "description": "Frame count annotation."},
                },
                "required": ["chi_file"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mlmd_convergence",
            "description": (
                "Plot k-space (k^n*chi(k)) and R-space (|chi(R)|, Hanning-windowed "
                "FFT) convergence versus number of averaged snapshots, straight "
                "from a FEFF output directory. Writes a two-panel PNG."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "FEFF output directory."},
                    "savefile": {"type": "string", "description": "Base name for the PNG (default 'exafs_convergence')."},
                    "step": {"type": "integer", "description": "Snapshot count increment between curves (default 10)."},
                    "k_weight": {"type": "integer", "description": "k-weight exponent (default 2)."},
                    "kmin": {"type": "number", "description": "FT window kmin in A^-1 (default 2.0)."},
                    "kmax": {"type": "number", "description": "FT window kmax in A^-1 (default 11.0)."},
                },
                "required": ["directory"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Factory — returns bound callables the orchestrator invokes
# ---------------------------------------------------------------------------


def create_tool_functions(data_path: str, output_dir: str) -> dict:
    """Bind the pipeline tools to the session's active file and output dir.

    ``data_path`` is passed as a string (the parameter is named ``data_path``,
    so the host hands over the active file path rather than loading it). Stages
    that operate on intermediate artifacts (trajectory, FEFF directory, chi
    file) take those paths as explicit arguments.
    """
    return {
        "mlmd_relax": lambda backend, structure_path=None, device="cpu", model=None,
        head="omat", modal="mpa", checkpoint=None, fmax=0.05: mlmd_relax(
            data_path, output_dir, backend, structure_path=structure_path,
            device=device, model=model, head=head, modal=modal,
            checkpoint=checkpoint, fmax=fmax,
        ),
        "mlmd_md": lambda backend, structure_path=None, device="cpu", model=None,
        head="omat", modal="mpa", checkpoint=None, temperature=300.0,
        step_size=10.0, n_steps=11000: mlmd_md(
            data_path, output_dir, backend, structure_path=structure_path,
            device=device, model=model, head=head, modal=modal,
            checkpoint=checkpoint,
            temperature=temperature, step_size=step_size, n_steps=n_steps,
        ),
        "mlmd_feff_input": lambda trajectory_path, target_atom, hole=1, rmax=6.0,
        scf="6.0 0 30 0.2 1", s02=1.0, control="1 1 1 1 1 1", corrections=None,
        step_size=250, sampling_start=0, cluster_buffer=2.5: mlmd_feff_input(
            trajectory_path, target_atom, hole=hole, rmax=rmax, scf=scf, s02=s02,
            control=control, corrections=corrections, step_size=step_size,
            sampling_start=sampling_start, cluster_buffer=cluster_buffer,
        ),
        "mlmd_run_feff": lambda directory, feff_bin=DEFAULT_FEFF_BIN,
        max_workers=32, cleanup=True: mlmd_run_feff(
            directory, feff_bin=feff_bin, max_workers=max_workers, cleanup=cleanup,
        ),
        "mlmd_cleanup": lambda root, dry_run=False, remove_empty_dirs=False: mlmd_cleanup(
            root, dry_run=dry_run, remove_empty_dirs=remove_empty_dirs,
        ),
        "mlmd_average_chi": lambda directory, savefile="exafs", exp_file=None,
        exp_col2_is_k2chi=False, kmin=2.0, kmax=12.0: mlmd_average_chi(
            directory, output_dir, savefile=savefile, exp_file=exp_file,
            exp_col2_is_k2chi=exp_col2_is_k2chi, kmin=kmin, kmax=kmax,
        ),
        "mlmd_fit_e0": lambda chi_file, exp_file, savefile="exafs_E0_fit",
        exp_col2_is_k2chi=False, kmin=2.0, kmax=12.0, e0_min=-10.0,
        e0_max=10.0: mlmd_fit_e0(
            chi_file, exp_file, output_dir, savefile=savefile,
            exp_col2_is_k2chi=exp_col2_is_k2chi, kmin=kmin, kmax=kmax,
            e0_min=e0_min, e0_max=e0_max,
        ),
        "mlmd_lcf": lambda exp_file, standards, savefile="lcf_fit", metric="redchi",
        max_components=3, kmin=None, kmax=None, k_weight=2, e0_min=-20.0,
        e0_max=20.0, e0_sign=-1.0, scale_for_shift=True, e0_mode="joint", n_starts=5,
        e0_start_step=2.5: mlmd_lcf(
            exp_file, standards, output_dir, savefile=savefile, metric=metric,
            max_components=max_components, kmin=kmin, kmax=kmax, k_weight=k_weight,
            e0_min=e0_min, e0_max=e0_max, e0_sign=e0_sign,
            scale_for_shift=scale_for_shift, e0_mode=e0_mode, n_starts=n_starts,
            e0_start_step=e0_start_step,
        ),
        "mlmd_plot": lambda chi_file, savefile="exafs_k2", k_weight=2, band="sem",
        n_samples=None: mlmd_plot(
            chi_file, output_dir, savefile=savefile, k_weight=k_weight, band=band,
            n_samples=n_samples,
        ),
        "mlmd_convergence": lambda directory, savefile="exafs_convergence", step=10,
        k_weight=2, kmin=2.0, kmax=11.0: mlmd_convergence(
            directory, output_dir, savefile=savefile, step=step, k_weight=k_weight,
            kmin=kmin, kmax=kmax,
        ),
    }

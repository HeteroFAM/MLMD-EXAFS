"""MLMD-EXAFS plug-in custom tool.

Exposes the MLIP molecular-dynamics + FEFF EXAFS workflow as a set of custom
tools that a host chat orchestrator can call during a session. Each pipeline
stage is one tool; together they cover:

    relax -> md -> feff-input -> run-feff -> average -> plot / convergence

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
from mlmd_exafs.feff import generate_feff_inputs_from_trajectory
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
    fmax: float = 0.05,
) -> dict:
    """Relax cell + positions with an MLIP; write relaxed.xyz to output_dir."""
    src = _resolve(structure_path, data_path)
    out = str(Path(output_dir) / "relaxed.xyz")
    calc = build_calculator(backend, device=device, model=model, head=head, modal=modal)
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
    temperature: float = 300.0,
    step_size: float = 10.0,
    n_steps: int = 11000,
) -> dict:
    """Run NVT molecular dynamics; write an extxyz trajectory under output_dir."""
    src = _resolve(structure_path, data_path)
    calc = build_calculator(backend, device=device, model=model, head=head, modal=modal)
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
) -> dict:
    """Batch-execute FEFF in every feff.inp subdirectory of ``directory``."""
    result = run_feff_batch(directory, feff_bin=feff_bin, max_workers=max_workers)
    result["status"] = "success" if result["n_failed"] == 0 else "partial"
    result["next_step"] = f"Run mlmd_average_chi with directory='{directory}'."
    return result


def mlmd_average_chi(directory: str, output_dir: str, savefile: str = "exafs") -> dict:
    """Average all chi.dat files into a converged chi(k) with a sampling band."""
    base = str(Path(output_dir) / savefile)
    result = average_chi(directory, base)
    # numpy arrays are not JSON-serializable; drop them from the returned dict
    for key in ("k", "chi_avg", "chi_std", "chi_sem"):
        result.pop(key, None)
    result["status"] = "success" if result["n_samples"] > 0 else "no_data"
    result["next_step"] = (
        f"Run mlmd_plot with chi_file='{result['output_file']}'."
    )
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
                    "model": {"type": "string", "description": "Model name (backend default if omitted)."},
                    "head": {"type": "string", "enum": ["oc20", "omat", "omol", "odac", "omc"]},
                    "modal": {"type": "string", "enum": ["mpa", "omat24"]},
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
                "extxyz trajectory. Point structure_path at the relaxed "
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
                    "model": {"type": "string"},
                    "head": {"type": "string", "enum": ["oc20", "omat", "omol", "odac", "omc"]},
                    "modal": {"type": "string", "enum": ["mpa", "omat24"]},
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
                    "trajectory_path": {"type": "string", "description": "MD trajectory (extxyz/traj)."},
                    "target_atom": {"type": "integer", "description": "Index of the absorbing atom."},
                    "hole": {"type": "integer", "description": "HOLE: 1=K, 2=L1, 3=L2, 4=L3 (default 1)."},
                    "rmax": {"type": "number", "description": "FEFF RMAX cutoff in A (default 6.0)."},
                    "scf": {"type": "string", "description": 'SCF card (default "6.0 0 30 0.2 1").'},
                    "s02": {"type": "number", "description": "S0^2 amplitude factor (default 1.0)."},
                    "control": {"type": "string", "description": 'CONTROL card (default "1 1 1 1 1 1").'},
                    "corrections": {"type": "string", "description": 'CORRECTIONS "vrcorr vicorr", or omit.'},
                    "step_size": {"type": "integer", "description": "Sample every N-th frame (default 250)."},
                    "sampling_start": {"type": "integer", "description": "First frame to sample (default 0)."},
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
                },
                "required": ["directory"],
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
                "<savefile>-chi_avg.dat."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "FEFF output directory."},
                    "savefile": {"type": "string", "description": "Base name for the averaged chi file (default 'exafs')."},
                },
                "required": ["directory"],
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
        head="omat", modal="mpa", fmax=0.05: mlmd_relax(
            data_path, output_dir, backend, structure_path=structure_path,
            device=device, model=model, head=head, modal=modal, fmax=fmax,
        ),
        "mlmd_md": lambda backend, structure_path=None, device="cpu", model=None,
        head="omat", modal="mpa", temperature=300.0, step_size=10.0,
        n_steps=11000: mlmd_md(
            data_path, output_dir, backend, structure_path=structure_path,
            device=device, model=model, head=head, modal=modal,
            temperature=temperature, step_size=step_size, n_steps=n_steps,
        ),
        "mlmd_feff_input": lambda trajectory_path, target_atom, hole=1, rmax=6.0,
        scf="6.0 0 30 0.2 1", s02=1.0, control="1 1 1 1 1 1", corrections=None,
        step_size=250, sampling_start=0: mlmd_feff_input(
            trajectory_path, target_atom, hole=hole, rmax=rmax, scf=scf, s02=s02,
            control=control, corrections=corrections, step_size=step_size,
            sampling_start=sampling_start,
        ),
        "mlmd_run_feff": lambda directory, feff_bin=DEFAULT_FEFF_BIN,
        max_workers=32: mlmd_run_feff(directory, feff_bin=feff_bin, max_workers=max_workers),
        "mlmd_average_chi": lambda directory, savefile="exafs": mlmd_average_chi(
            directory, output_dir, savefile=savefile
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

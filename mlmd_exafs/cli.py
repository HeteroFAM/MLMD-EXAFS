"""Command-line interface for the MLMD-EXAFS workflow.

Subcommands, in pipeline order::

    mlmd-exafs relax       # MLIP cell + position relaxation
    mlmd-exafs md          # NVT molecular dynamics -> trajectory
    mlmd-exafs feff-input  # carve snapshots -> feff.inp files
    mlmd-exafs run-feff    # batch FEFF execution (+ scratch-file cleanup)
    mlmd-exafs cleanup     # remove FEFF scratch files from exafs_* dirs
    mlmd-exafs average     # average chi.dat -> chi_avg.dat (+ E0 fit if --exp-file)
    mlmd-exafs fit-e0      # fit E0 shift of chi_avg.dat against experiment
    mlmd-exafs lcf         # linear combination fit of several spectra to experiment
    mlmd-exafs plot        # k-weighted chi(k) with sampling band
    mlmd-exafs convergence # k- and R-space convergence panels

Run ``mlmd-exafs <subcommand> -h`` for per-stage options.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .calculators import BACKENDS


def _add_backend_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--backend",
        required=True,
        choices=BACKENDS,
        help="MLIP force-engine backend.",
    )
    p.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"], help="Compute device."
    )
    p.add_argument(
        "--model", default=None, help="Pretrained model name (backend default if omitted)."
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a local (e.g. fine-tuned) model checkpoint; replaces --model.",
    )
    p.add_argument(
        "--head",
        default="omat",
        choices=["oc20", "omat", "omol", "odac", "omc"],
        help="UMA task head.",
    )
    p.add_argument(
        "--modal",
        default="mpa",
        choices=["mpa", "omat24"],
        help="ORB / SevenNet dataset modality.",
    )


def _add_e0_fit_args(p: argparse.ArgumentParser) -> None:
    from .fitting_E0 import (
        DEFAULT_KMAX,
        DEFAULT_KMIN,
        E0_GRID_N_DEFAULT,
        E0_MAX_DEFAULT,
        E0_MIN_DEFAULT,
        K2EV_DEFAULT,
    )

    g = p.add_argument_group("E0 fitting")
    g.add_argument(
        "--fit-kmin", type=float, default=DEFAULT_KMIN, help="E0 fit kmin (A^-1)."
    )
    g.add_argument(
        "--fit-kmax", type=float, default=DEFAULT_KMAX, help="E0 fit kmax (A^-1)."
    )
    g.add_argument(
        "--exp-col2-is-k2chi",
        action="store_true",
        help="Experimental column 2 is already k^2*chi (default: chi).",
    )
    g.add_argument("--e0-min", type=float, default=E0_MIN_DEFAULT, help="E0 search min (eV).")
    g.add_argument("--e0-max", type=float, default=E0_MAX_DEFAULT, help="E0 search max (eV).")
    g.add_argument("--e0-grid-n", type=int, default=E0_GRID_N_DEFAULT, help="E0 grid points.")
    g.add_argument("--k2ev", type=float, default=K2EV_DEFAULT, help="hbar^2/2m_e (eV A^2).")


def _run_e0_fit(args, sim_file: str, outdir: str) -> dict:
    from .fitting_E0 import fit_e0

    return fit_e0(
        sim_file=sim_file,
        exp_file=args.exp_file,
        outdir=outdir,
        kmin=args.fit_kmin,
        kmax=args.fit_kmax,
        exp_col2_is_k2chi=args.exp_col2_is_k2chi,
        e0_min=args.e0_min,
        e0_max=args.e0_max,
        e0_grid_n=args.e0_grid_n,
        k2ev=args.k2ev,
    )


def _make_calculator(args):
    from .calculators import build_calculator

    return build_calculator(
        args.backend,
        device=args.device,
        model=args.model,
        head=args.head,
        modal=args.modal,
        checkpoint=args.checkpoint,
    )


def _cmd_relax(args):
    from .md import relax

    calc = _make_calculator(args)
    result = relax(args.input, args.output, calc, fmax=args.fmax, steps=args.steps)
    print(json.dumps(result, indent=2))


def _cmd_md(args):
    from .md import run_md

    calc = _make_calculator(args)
    result = run_md(
        args.input,
        calc,
        args.directory,
        temperature=args.temperature,
        step_size=args.step_size,
        n_steps=args.n_steps,
    )
    print(json.dumps(result, indent=2))


def _cmd_feff_input(args):
    from .feff import generate_feff_inputs_from_trajectory

    result = generate_feff_inputs_from_trajectory(
        trajectory_path=args.trajectory,
        target_atom=args.target_atom,
        hole=args.hole,
        rmax=args.rmax,
        scf=args.scf,
        s02=args.s02,
        control=args.control,
        corrections=args.corrections,
        step_size=args.step_size,
        sampling_start=args.sampling_start,
    )
    print(json.dumps(result, indent=2))


def _cmd_run_feff(args):
    from .run_feff import run_feff_batch

    result = run_feff_batch(
        args.directory,
        feff_bin=args.feff_bin,
        max_workers=args.max_workers,
        cleanup=not args.no_cleanup,
    )
    print(json.dumps(result, indent=2))


def _cmd_cleanup(args):
    from .cleanup_exafs import cleanup_exafs

    result = cleanup_exafs(
        args.root,
        keep=args.keep,
        dry_run=args.dry_run,
        remove_empty_dirs=args.remove_empty_dirs,
    )
    print(json.dumps(result, indent=2))


def _cmd_average(args):
    from .analysis import average_chi

    result = average_chi(args.directory, args.savefile)
    result.pop("k", None)
    for key in ("chi_avg", "chi_std", "chi_sem"):
        result.pop(key, None)
    if args.exp_file and result["n_samples"] > 0:
        outdir = args.e0_outdir or f"{args.savefile}_E0_fit"
        result["e0_fit"] = _run_e0_fit(args, result["output_file"], outdir)
    print(json.dumps(result, indent=2))


def _cmd_fit_e0(args):
    outdir = args.outdir or f"{os.path.splitext(args.chi_file)[0]}_E0_fit"
    result = _run_e0_fit(args, args.chi_file, outdir)
    print(json.dumps(result, indent=2))


def _cmd_lcf(args):
    from .lcf import run_lcf

    result = run_lcf(
        exp_file=args.exp_file,
        standards=args.standards,
        outdir=args.outdir,
        metric=args.metric,
        max_components=args.max_components,
        kmin=args.kmin,
        kmax=args.kmax,
        k_weight=args.k_weight,
        e0_min=args.e0_min,
        e0_max=args.e0_max,
        e0_sign=args.e0_sign,
        scale_for_shift=not args.no_scale_for_shift,
        min_valid_frac=args.min_valid_frac,
        n_top=args.n_top,
    )
    print(json.dumps(result, indent=2))


def _cmd_plot(args):
    from .analysis import plot_chi

    plot_chi(
        args.chi_file,
        args.savefile,
        k_weight=args.k_weight,
        band=args.band,
        n_samples=args.n_samples,
    )


def _cmd_convergence(args):
    from .analysis import plot_convergence

    plot_convergence(
        args.directory,
        args.savefile,
        step=args.step,
        k_weight=args.k_weight,
        kmin=args.kmin,
        kmax=args.kmax,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlmd-exafs",
        description="MLIP molecular dynamics + FEFF EXAFS simulation workflow.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # relax
    p = sub.add_parser("relax", help="MLIP cell + position relaxation.")
    p.add_argument("-i", "--input", required=True, help="Input structure file.")
    p.add_argument("-o", "--output", required=True, help="Output relaxed structure.")
    p.add_argument("--fmax", type=float, default=0.05, help="Force threshold (eV/A).")
    p.add_argument("--steps", type=int, default=10000, help="Max optimizer steps.")
    _add_backend_args(p)
    p.set_defaults(func=_cmd_relax)

    # md
    p = sub.add_parser("md", help="NVT molecular dynamics.")
    p.add_argument("-i", "--input", required=True, help="(Relaxed) structure file.")
    p.add_argument("-d", "--directory", default="./md_out", help="Output directory.")
    p.add_argument("--temperature", type=float, default=300.0, help="Temperature (K).")
    p.add_argument("--step-size", type=float, default=10.0, help="Time step (a.u.).")
    p.add_argument("--n-steps", type=int, default=11000, help="Number of MD steps.")
    _add_backend_args(p)
    p.set_defaults(func=_cmd_md)

    # feff-input
    p = sub.add_parser("feff-input", help="Generate FEFF inputs from a trajectory.")
    p.add_argument("-f", "--trajectory", required=True, help="MD trajectory (xyz/traj).")
    p.add_argument("-i", "--target-atom", type=int, required=True, help="Absorber index.")
    p.add_argument("--hole", type=int, default=1, help="HOLE: 1=K, 2=L1, 3=L2, 4=L3.")
    p.add_argument("--rmax", type=float, default=6.0, help="FEFF RMAX cutoff (A).")
    p.add_argument("--scf", default="6.0 0 30 0.2 1", help="SCF card parameters.")
    p.add_argument("--s02", type=float, default=1.0, help="S0^2 amplitude factor.")
    p.add_argument("--control", default="1 1 1 1 1 1", help="CONTROL card.")
    p.add_argument(
        "--corrections",
        default=None,
        help='CORRECTIONS card "vrcorr vicorr" (omit to leave out).',
    )
    p.add_argument("--step-size", type=int, default=250, help="Sample every N-th frame.")
    p.add_argument("--sampling-start", type=int, default=0, help="First frame to sample.")
    p.set_defaults(func=_cmd_feff_input)

    # run-feff
    p = sub.add_parser("run-feff", help="Batch-execute FEFF over input directories.")
    p.add_argument("-d", "--directory", required=True, help="FEFF output directory.")
    p.add_argument(
        "--feff-bin",
        default="/share/feff/feff90_binaries/feff.x",
        help="Path to the FEFF executable.",
    )
    p.add_argument("--max-workers", type=int, default=32, help="Concurrent FEFF jobs.")
    p.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep all FEFF scratch files (default: keep only feff.inp/feff.out/chi.dat).",
    )
    p.set_defaults(func=_cmd_run_feff)

    # cleanup
    from .cleanup_exafs import DEFAULT_KEEP

    p = sub.add_parser("cleanup", help="Remove FEFF scratch files from exafs_* dirs.")
    p.add_argument(
        "root", nargs="?", default=".", help="Directory to search for exafs_* dirs."
    )
    p.add_argument(
        "--keep",
        nargs="+",
        default=sorted(DEFAULT_KEEP),
        help="Basenames or shell-style patterns to keep.",
    )
    p.add_argument("--dry-run", action="store_true", help="Only report deletions.")
    p.add_argument(
        "--remove-empty-dirs", action="store_true", help="Also remove emptied dirs."
    )
    p.set_defaults(func=_cmd_cleanup)

    # average
    p = sub.add_parser("average", help="Average chi.dat files.")
    p.add_argument("-d", "--directory", required=True, help="FEFF output directory.")
    p.add_argument("--savefile", required=True, help="Base path for averaged chi file.")
    p.add_argument(
        "--exp-file",
        default=None,
        help="Experimental chi(k) (.dat or .csv); if given, fit the E0 shift.",
    )
    p.add_argument(
        "--e0-outdir",
        default=None,
        help="E0 fit output dir (default: <savefile>_E0_fit).",
    )
    _add_e0_fit_args(p)
    p.set_defaults(func=_cmd_average)

    # fit-e0
    p = sub.add_parser("fit-e0", help="Fit E0 shift of averaged chi(k) to experiment.")
    p.add_argument("--chi-file", required=True, help="Averaged *-chi_avg.dat file.")
    p.add_argument(
        "--exp-file", required=True, help="Experimental chi(k) file (.dat or .csv)."
    )
    p.add_argument(
        "-o", "--outdir", default=None, help="Output dir (default: <chi-file>_E0_fit)."
    )
    _add_e0_fit_args(p)
    p.set_defaults(func=_cmd_fit_e0)

    # lcf
    from .lcf import METRIC_ALIASES

    p = sub.add_parser(
        "lcf", help="Linear combination fit of simulated spectra to experiment."
    )
    p.add_argument(
        "--exp-file", required=True, help="Experimental chi(k) file (.dat or .csv)."
    )
    p.add_argument(
        "--standards",
        nargs="+",
        required=True,
        help='Simulated chi(k) files or glob patterns, e.g. "sims/*-chi_avg.dat".',
    )
    p.add_argument("-o", "--outdir", default="lcf_fit", help="Output directory.")
    p.add_argument(
        "--metric",
        default="redchi",
        choices=sorted(METRIC_ALIASES),
        help="Optimization/ranking metric.",
    )
    p.add_argument(
        "--max-components", type=int, default=3, help="Max standards per combination."
    )
    p.add_argument("--kmin", type=float, default=None, help="Fit kmin (A^-1).")
    p.add_argument("--kmax", type=float, default=None, help="Fit kmax (A^-1).")
    p.add_argument("--k-weight", type=int, default=2, help="k-weight exponent.")
    p.add_argument("--e0-min", type=float, default=-20.0, help="Delta E0 search min (eV).")
    p.add_argument("--e0-max", type=float, default=20.0, help="Delta E0 search max (eV).")
    p.add_argument(
        "--e0-sign",
        type=float,
        default=-1.0,
        choices=[1.0, -1.0],
        help=(
            "E0 sign convention: k_query^2 = k^2 + sign*0.2625*dE0. "
            "-1 (default) = Artemis/IFEFFIT, same as fit-e0."
        ),
    )
    p.add_argument(
        "--no-scale-for-shift",
        action="store_true",
        help="Do not optimize a temporary amplitude during each E0 search.",
    )
    p.add_argument(
        "--min-valid-frac",
        type=float,
        default=0.95,
        help="Min fraction of experimental points a shifted standard must cover.",
    )
    p.add_argument("--n-top", type=int, default=10, help="Top fits in JSON output.")
    p.set_defaults(func=_cmd_lcf)

    # plot
    p = sub.add_parser("plot", help="Plot k-weighted chi(k) with sampling band.")
    p.add_argument("--chi-file", required=True, help="Averaged *-chi_avg.dat file.")
    p.add_argument("--savefile", required=True, help="Base path for output PNG.")
    p.add_argument("--k-weight", type=int, default=2, help="k-weight exponent.")
    p.add_argument("--band", default="sem", choices=["sem", "std", "none"], help="Band type.")
    p.add_argument("--n-samples", type=int, default=None, help="Frame count annotation.")
    p.set_defaults(func=_cmd_plot)

    # convergence
    p = sub.add_parser("convergence", help="k- and R-space convergence panels.")
    p.add_argument("-d", "--directory", required=True, help="FEFF output directory.")
    p.add_argument("--savefile", default="exafs_convergence", help="Base path for PNG.")
    p.add_argument("--step", type=int, default=10, help="Snapshot count increment.")
    p.add_argument("--k-weight", type=int, default=2, help="k-weight exponent.")
    p.add_argument("--kmin", type=float, default=2.0, help="FT window kmin (A^-1).")
    p.add_argument("--kmax", type=float, default=11.0, help="FT window kmax (A^-1).")
    p.set_defaults(func=_cmd_convergence)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Command-line interface for the MLMD-EXAFS workflow.

Subcommands, in pipeline order::

    mlmd-exafs relax       # MLIP cell + position relaxation
    mlmd-exafs md          # NVT molecular dynamics -> trajectory
    mlmd-exafs feff-input  # carve snapshots -> feff.inp files
    mlmd-exafs run-feff    # batch FEFF execution
    mlmd-exafs average     # average chi.dat -> chi_avg.dat
    mlmd-exafs plot        # k-weighted chi(k) with sampling band
    mlmd-exafs convergence # k- and R-space convergence panels

Run ``mlmd-exafs <subcommand> -h`` for per-stage options.
"""

from __future__ import annotations

import argparse
import json
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
    p.add_argument("--model", default=None, help="Model name/checkpoint (backend default if omitted).")
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


def _make_calculator(args):
    from .calculators import build_calculator

    return build_calculator(
        args.backend,
        device=args.device,
        model=args.model,
        head=args.head,
        modal=args.modal,
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
        args.directory, feff_bin=args.feff_bin, max_workers=args.max_workers
    )
    print(json.dumps(result, indent=2))


def _cmd_average(args):
    from .analysis import average_chi

    result = average_chi(args.directory, args.savefile)
    result.pop("k", None)
    for key in ("chi_avg", "chi_std", "chi_sem"):
        result.pop(key, None)
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
    p.set_defaults(func=_cmd_run_feff)

    # average
    p = sub.add_parser("average", help="Average chi.dat files.")
    p.add_argument("-d", "--directory", required=True, help="FEFF output directory.")
    p.add_argument("--savefile", required=True, help="Base path for averaged chi file.")
    p.set_defaults(func=_cmd_average)

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

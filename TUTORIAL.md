# MLMD-EXAFS Tutorial

This tutorial walks through computing an EXAFS χ(k) spectrum end to end, from a
crystal structure to k- and R-space plots. It uses Zn K-edge EXAFS of a
Zn-substituted hematite as the running example, but the steps are identical for
any absorber/material.

Every stage is a subcommand of `mlmd-exafs`. Run `mlmd-exafs <stage> -h` for
the complete option list. The same operations are available as Python functions
(see [Python API](#python-api) at the end).

---

## Prerequisites

- One MLIP backend installed (`pip install -e ".[chgnet]"`, etc.).
- A FEFF9 executable available on the machine.
- A structure file readable by ASE (`.cif`, `.extxyz`, `POSCAR`, ...).

Pick the backend that fits your chemistry:

| System | Recommended backend |
|--------|--------------------|
| Magnetic (Fe, Co, Ni, Mn, Cr) | `chgnet` |
| High-accuracy general materials | `mace` or `uma` |
| Large cells / fast screening | `orb` |

---

## Step 1 — Cell relaxation

Relax atomic positions and the cell before MD, so the dynamics start from
the potential's equilibrium geometry.

```bash
mlmd-exafs relax \
    -i zn_hematite.cif \
    -o relaxed.xyz \
    --backend chgnet \
    --device cpu \
    --fmax 0.02
```

- `--fmax` — force convergence threshold in eV/Å (default 0.02; loosen to 0.05
  for a faster, rougher relaxation).
- `--no-symmetry` — do not constrain the relaxation to the starting crystal
  symmetry (symmetry is kept by default).
- `--pressure <GPa>` — target external pressure (default 0).
- `--max-volume-change <frac>` — warn if the cell volume changes by more than
  this fraction (default 0.25); check the starting structure if it fires.
  The result JSON reports `volume_change` and `volume_warning`.
- `--device cuda` — use the GPU if available.
- `--checkpoint <path>` — load a local (e.g. fine-tuned) model file instead of
  the backend's pretrained model. Use the same `--checkpoint` in Step 2. See
  the README's "Fine-tuned models" section for the file format each backend
  expects.

The command prints a JSON summary (convergence flag, step count, final energy,
initial/final cell parameters). Check that `"converged": true`.

---

## Step 2 — NVT molecular dynamics

Propagate the relaxed structure under a canonical (NVT) ensemble with a
Nose–Hoover chain thermostat. The trajectory snapshots supply the thermal
disorder for EXAFS averaging.

```bash
mlmd-exafs md \
    -i relaxed.xyz \
    -d md_out \
    --backend chgnet \
    --temperature 300 \
    --step-size 10 \
    --n-steps 11000
```

| Option | Default | Notes |
|--------|---------|-------|
| `--temperature` | 300 K | Match the experimental measurement temperature |
| `--step-size` | 10 a.u. (≈ 0.24 fs) | Conservative; lower it if MD becomes unstable |
| `--n-steps` | 11000 (≈ 2.6 ps) | Increase for slow dynamics or better convergence |

Output is written to `md_out/relaxed/relaxed.traj` (ASE trajectory) plus a
`.log`. If that trajectory already exists the stage is skipped, so re-running is
safe.

**Sampling guidance.** Every trajectory frame is a candidate FEFF snapshot.
Typically 20–40 snapshots suffice for first-shell EXAFS; extended-range
analysis may need 50+. The number of snapshots is controlled in the next step by
`--step-size` (frame stride) and `--sampling-start`.

---

## Step 3 — Generate FEFF inputs

Carve the local environment around the absorbing atom in each sampled snapshot
and write a `feff.inp` for it.

```bash
mlmd-exafs feff-input \
    -f md_out/relaxed/relaxed.traj \
    --target-atom 0 \
    --hole 1 \
    --rmax 6.0 \
    --scf "6.0 0 30 0.2 1" \
    --step-size 250 \
    --sampling-start 1000
```

- `--target-atom` — index of the absorbing atom in the trajectory frames
  (0-based). This must be the same atom you compare to experiment.
- `--step-size` — sample every N-th frame. `--sampling-start` skips the
  equilibration portion of the trajectory.

### Choosing the edge (`--hole`)

| Element type | `--hole` | Edge |
|-------------|----------|------|
| 3d metals (Ti–Zn) | 1 | K |
| 4d metals (Zr–Cd) | 4 | L3 |
| 5d metals (Hf–Hg) | 4 | L3 |
| Light elements (C–Si) | 1 | K |

### Other FEFF cards

| Option | Default | Guidance |
|--------|---------|----------|
| `--rmax` | 6.0 Å | 4.5 Å first shell only; 6.0 Å for 2–4 shells; 8.0 Å for extended multiple scattering. The carve radius is `rmax + --cluster-buffer` (default 2.5 Å). |
| `--scf` | `6.0 0 30 0.2 1` | Standard metals/oxides. Use `5.5 0 30 0.05 10` for f-electron systems. |
| `--s02` | 1.0 | S0² amplitude reduction; refine against experiment (0.8–1.0). |
| `--corrections` | omitted | Fermi-level shift `"vrcorr vicorr"`, e.g. `"3.0 0.0"`. Omit for a first pass, fit later. |
| `--control` | `1 1 1 1 1 1` | Full calculation. `1 1 0 1 1 1` skips the XANES path module (EXAFS only). |
| `--cluster-buffer` | 2.5 Å | Carve radius beyond `--rmax`. Lower it (0 allowed) if periodic images of the absorber enter the cluster in a small cell. `rmax + buffer` must be ≥ the SCF radius. |

Inputs land in a directory named after the parameters, e.g.
`md_out/relaxed/exafs_Zn_hole1_de_0.0_s02_1.0_rc_6.0/`, with one
`000250_0/feff.inp` subdirectory per sampled frame plus a
`neighborhoods_0.xyz` file for inspection.

---

## Step 4 — Run FEFF

Execute FEFF in every generated input directory.

```bash
mlmd-exafs run-feff \
    -d md_out/relaxed/exafs_Zn_hole1_de_0.0_s02_1.0_rc_6.0 \
    --feff-bin /share/feff/feff90_binaries/feff.x \
    --max-workers 32
```

Each subdirectory gets a `chi.dat` (the EXAFS signal) and a `feff.out` (log).
`--max-workers` bounds the number of concurrent FEFF jobs.

### Scratch-file cleanup

FEFF leaves many large intermediate files (`phase.bin`, `feff*.dat`, `pot.bin`,
...) in every snapshot directory. Once all jobs finish, `run-feff` deletes them,
keeping only `feff.inp`, `feff.out`, `chi.dat` and the `neighborhoods_*.xyz`
inspection file. Pass `--no-cleanup` to keep everything.

To clean existing runs (or runs made with the csh script) directly:

```bash
mlmd-exafs cleanup md_out --dry-run            # preview
mlmd-exafs cleanup md_out                      # every exafs_* dir under md_out
mlmd-exafs cleanup md_out --keep feff.inp chi.dat "*.xyz" --remove-empty-dirs
```

### csh alternative (cluster batch)

If you prefer the classic csh batch loop, use `scripts/run_feff.csh`:

```csh
set DIRS=`find md_out/relaxed/exafs_Zn_hole1_de_0.0_s02_1.0_rc_6.0/* -type d`
source scripts/run_feff.csh
```

Edit `FEFF_BIN` and `max_num_processes` at the top of the script as needed.
It runs `mlmd-exafs cleanup` on the parent `exafs_*` directory when done;
`set CLEANUP=0` before sourcing to skip that.

---

## Step 5 — Average χ(k)

Average all `chi.dat` files into a converged spectrum with a per-k sampling
band.

```bash
mlmd-exafs average \
    -d md_out/relaxed/exafs_Zn_hole1_de_0.0_s02_1.0_rc_6.0 \
    --savefile exafs
```

Writes `exafs-chi_avg.dat` with columns `k  chi_avg  chi_std  chi_sem`:

- `chi_std` — snapshot-to-snapshot standard deviation.
- `chi_sem` — standard error of the mean (`chi_std / sqrt(N)`).

Both quantify MD sampling spread, not a force-field error bar.

### E0 shift against experiment (optional)

If you have an experimental spectrum, pass it with `--exp-file` and the E0
shift is fitted right after averaging:

```bash
mlmd-exafs average \
    -d md_out/relaxed/exafs_Zn_hole1_de_0.0_s02_1.0_rc_6.0 \
    --savefile exafs \
    --exp-file exp_k.csv \
    --fit-kmin 2.0 --fit-kmax 12.0
```

The same fit is available on its own for an existing averaged file:

```bash
mlmd-exafs fit-e0 --chi-file exafs-chi_avg.dat --exp-file exp_k.dat
```

The simulated k grid is shifted, k'² = k² + E0/3.81 (Artemis/IFEFFIT sign
convention: positive E0 moves the theory edge up in energy), and E0 is chosen to
minimize the mean squared deviation of k²χ(k) from experiment over
[`--fit-kmin`, `--fit-kmax`] (grid search over [`--e0-min`, `--e0-max`] =
[-10, 10] eV, then golden-section refinement).

- **Experimental file format** — `.dat` (whitespace-delimited) or `.csv`
  (comma-delimited). The first two numeric columns are read as k (Å⁻¹) and
  χ(k); header and `#` lines are skipped. Add `--exp-col2-is-k2chi` if
  column 2 is already k²χ(k).
- **Output** — `exafs_E0_fit/` (or `--e0-outdir` / `fit-e0 -o`) containing
  `avg_chi_E0_shifted.dat` (k, χ, k²χ), `best_E0_comparison.dat`,
  `E0_fit_summary.txt` and `calc_vs_exp.png`. The JSON summary printed by the
  command includes `best_E0`.

---

## Step 6 — Plot

### k-weighted spectrum with sampling band

```bash
mlmd-exafs plot \
    --chi-file exafs-chi_avg.dat \
    --savefile exafs_k2 \
    --k-weight 2 \
    --band sem
```

- `--k-weight` — raise (2→3) to emphasize the high-k region, lower (→1) for
  low-k.
- `--band` — `sem` (uncertainty on the mean), `std` (snapshot spread), or
  `none`.

Writes `exafs_k2.png`.

### k- and R-space convergence panels

```bash
mlmd-exafs convergence \
    -d md_out/relaxed/exafs_Zn_hole1_de_0.0_s02_1.0_rc_6.0 \
    --savefile exafs_convergence \
    --step 10
```

Writes `exafs_convergence.png`: a left panel of k²χ(k) and a right panel of
|χ̃(R)| (Hanning-windowed Fourier transform, `kmin=2`, `kmax=11` Å⁻¹ by
default), each drawn for increasing snapshot counts. Convergence is reached when
the curves stop changing as more snapshots are added.

---

## Step 7 — Linear combination fitting (optional)

When several candidate structures could be present in the sample (different
adsorption sites, protonation states, substitution sites, ...), run Steps 1–5
for each candidate and fit the experimental spectrum as a weighted mix of
their averaged spectra:

```bash
mlmd-exafs lcf \
    --exp-file goethite_pt_exp_k.dat \
    --standards "sims/*-chi_avg.dat" \
    --kmin 2.5 --kmax 14 \
    --max-components 3 \
    --metric redchi \
    -o lcf_fit
```

The fit has two steps:

1. **Per-standard ΔE0.** Each simulated spectrum ("standard") is shifted in
   energy against experiment, k_query² = k_exp² + `e0_sign`·0.262468·ΔE0,
   and ΔE0 is chosen within [`--e0-min`, `--e0-max`] = [-20, 20] eV to
   minimize the metric on k²-weighted χ(k). By default a temporary amplitude
   is fitted during this search so spectral shape drives the shift
   (`--no-scale-for-shift` to disable).
2. **Combinations.** Every combination of 1 to `--max-components` standards
   is fitted. Weights are ≥ 0 and sum to 1 (a single standard has weight 1).
   `--e0-mode` controls ΔE0 in this step:
   - `joint` (default): each standard's ΔE0 is refined together with the
     weights, starting from its step-1 value.
   - `joint_shared`: one ΔE0 shared by all standards of the combination is
     refined together with the weights (Athena's "single E0 shift").
   - `fixed`: ΔE0 values stay at their step-1 values (the pre-joint behavior).

   Joint fits start from the step-1 ΔE0 values and the `fixed` weights, plus
   `--n-starts` − 1 further starts with every ΔE0 offset by ±`--e0-start-step`,
   ±2·`--e0-start-step`, ... eV, and keep the best. The combinations are then
   ranked by `--metric`.

| Option | Default | Notes |
|--------|---------|-------|
| `--standards` | required | Files and/or glob patterns (`.dat` or `.csv`; columns k, χ) |
| `--metric` | `redchi` | `redchi`, `chi2`, `rmsd`, or `rfactor`. `redchi` counts (n−1) weights plus n (`joint`), 1 (`joint_shared`) or 0 (`fixed`) ΔE0 values as parameters, penalizing larger combinations |
| `--e0-mode` | `joint` | `joint`, `joint_shared`, or `fixed` (see above) |
| `--e0-min`/`--e0-max` | −20 / 20 | ΔE0 bounds (eV) for step 1 and the joint fits |
| `--n-starts` | 5 | Starting points per joint fit |
| `--e0-start-step` | 2.5 | ΔE0 offset (eV) between starting points |
| `--kmin`/`--kmax` | full range | Experimental k range used in the fit |
| `--k-weight` | 2 | k-weight used for the ΔE0 search and LCF |
| `--e0-sign` | −1 | −1 = Artemis/IFEFFIT, same ΔE0 as `fit-e0`. +1 flips every ΔE0 sign (weights unchanged) |
| `--min-valid-frac` | 0.95 | Shifted standards must cover this fraction of experimental points |

Outputs in `lcf_fit/`:

- `lcf_results.csv` — every combination, ranked, with all metrics, the
  parameter count, weights and fitted ΔE0 values (with 1σ uncertainties, NaN
  when pinned at a bound or not fitted), and the step-1 ΔE0 values.
- `delta_e0.csv` — step-1 ΔE0 and single-standard score for each standard.
- `best_lcf_fit.dat` — k, experimental/fitted/residual χ and k²χ for the
  best combination, built with its own fitted ΔE0 values. The header records
  `e0_mode`, weights, fitted and step-1 ΔE0 values.
- `best_lcf_fit.png` — experiment vs. best LCF (k²χ).
- `best_lcf_fit_R.png` — the same comparison in R-space (|χ(R)| of the
  k²-weighted χ(k), Hanning window over the fitted k range).
- `top_fits.csv` — rank-1 fit summary (rfactor, redchi, ΔE0, weight ± error
  per standard; `none` = in the library but unused).
- `top_<n_top>_candidates.csv` — the `n_top` best fits, one row each, one column per
  standard. To compare several runs side by side, use
  `python -m mlmd_exafs.lcf_summary dir1 dir2 ... -o summaries/`.
- `fit.log` — the progress output printed to the terminal, followed by the
  returned result as JSON (written even with `verbose=False`).

The number of combinations grows quickly: 13 standards with
`--max-components 3` means 377 fits.

---

## Interpreting the result

- **k-space (k²χ(k))** — the raw oscillations. A noisy high-k region signals
  insufficient MD sampling or too short a trajectory.
- **R-space (|χ̃(R)|)** — peaks sit at coordination-shell distances, shifted
  ~0.3–0.5 Å inward from the true distances by the EXAFS phase shift. Compare
  peak positions and amplitudes to experiment.

Common issues:

| Symptom | Likely cause / fix |
|---------|--------------------|
| Noisy high-k | More MD sampling / longer trajectory |
| Missing R-space peaks | `--rmax` too small; increase it |
| Amplitude mismatch | Fit `--s02` against experiment (0.8–1.0) |
| Peak position shift | Fit E0 (`fit-e0`); SCF convergence or wrong `--corrections`; adjust vrcorr |
| Non-physical oscillations | MD `--step-size` too large; reduce it |

---

## Python API

Every stage is a plain function:

```python
from mlmd_exafs.calculators import build_calculator
from mlmd_exafs.md import relax, run_md
from mlmd_exafs.feff import generate_feff_inputs_from_trajectory
from mlmd_exafs.run_feff import run_feff_batch
from mlmd_exafs.analysis import average_chi, plot_chi, plot_convergence
from mlmd_exafs.cleanup_exafs import cleanup_exafs
from mlmd_exafs.fitting_E0 import fit_e0
from mlmd_exafs.lcf import run_lcf

calc = build_calculator("chgnet", device="cpu")

relax("zn_hematite.cif", "relaxed.xyz", calc, fmax=0.02)

md = run_md("relaxed.xyz", calc, "md_out", temperature=300, n_steps=11000)

gen = generate_feff_inputs_from_trajectory(
    md["trajectory"], target_atom=0, hole=1, rmax=6.0, step_size=250,
    sampling_start=1000,
)

# cleans FEFF scratch files afterwards (cleanup=False to keep them);
# cleanup_exafs("md_out") does the same for any exafs_* dirs under a root
run_feff_batch(gen["output_dir"], feff_bin="/share/feff/feff90_binaries/feff.x")

avg = average_chi(gen["output_dir"], savefile="exafs")
# optional, when an experimental spectrum (.dat or .csv) is available
fit = fit_e0(avg["output_file"], "exp_k.csv", outdir="exafs_E0_fit")
print(fit["best_E0"])

# optional: linear combination fit of several candidates' averaged spectra
lcf = run_lcf("exp_k.dat", "sims/*-chi_avg.dat", outdir="lcf_fit",
              kmin=2.5, kmax=14, max_components=3)
print(lcf["best"]["standards"], lcf["best"]["weights"])
plot_chi(avg["output_file"], "exafs_k2", k_weight=2, band="sem",
         n_samples=avg["n_samples"])
plot_convergence(gen["output_dir"], "exafs_convergence")
```

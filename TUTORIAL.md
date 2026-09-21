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
    --fmax 0.05
```

- `--fmax` — force convergence threshold in eV/Å (0.05 is a good default; use
  0.02 for tighter geometries).
- `--device cuda` — use the GPU if available.

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

Output is written to `md_out/relaxed/relaxed.xyz` (extxyz trajectory) plus a
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
    -f md_out/relaxed/relaxed.xyz \
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
| `--rmax` | 6.0 Å | 4.5 Å first shell only; 6.0 Å for 2–4 shells; 8.0 Å for extended multiple scattering. The carve radius is automatically `rmax + 2.5 Å`. |
| `--scf` | `6.0 0 30 0.2 1` | Standard metals/oxides. Use `5.5 0 30 0.05 10` for f-electron systems. |
| `--s02` | 1.0 | S0² amplitude reduction; refine against experiment (0.8–1.0). |
| `--corrections` | omitted | Fermi-level shift `"vrcorr vicorr"`, e.g. `"3.0 0.0"`. Omit for a first pass, fit later. |
| `--control` | `1 1 1 1 1 1` | Full calculation. `1 1 0 1 1 1` skips the XANES path module (EXAFS only). |

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

### csh alternative (cluster batch)

If you prefer the classic csh batch loop, use `scripts/run_feff.csh`:

```csh
set DIRS=`find md_out/relaxed/exafs_Zn_hole1_de_0.0_s02_1.0_rc_6.0/* -type d`
source scripts/run_feff.csh
```

Edit `FEFF_BIN` and `max_num_processes` at the top of the script as needed.

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
| Peak position shift | SCF convergence or wrong `--corrections`; adjust vrcorr |
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

calc = build_calculator("chgnet", device="cpu")

relax("zn_hematite.cif", "relaxed.xyz", calc, fmax=0.05)

md = run_md("relaxed.xyz", calc, "md_out", temperature=300, n_steps=11000)

gen = generate_feff_inputs_from_trajectory(
    md["trajectory"], target_atom=0, hole=1, rmax=6.0, step_size=250,
    sampling_start=1000,
)

run_feff_batch(gen["output_dir"], feff_bin="/share/feff/feff90_binaries/feff.x")

avg = average_chi(gen["output_dir"], savefile="exafs")
plot_chi(avg["output_file"], "exafs_k2", k_weight=2, band="sem",
         n_samples=avg["n_samples"])
plot_convergence(gen["output_dir"], "exafs_convergence")
```

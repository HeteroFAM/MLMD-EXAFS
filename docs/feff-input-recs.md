# FEFF Input Recommendations

FEFF9 computes X-ray absorption spectra (EXAFS, XANES) from atomic clusters. Each calculation requires a `feff.inp` file specifying the absorbing atom, its local environment (POTENTIALS and ATOMS lists), and computation parameters (HOLE, SCF, RMAX, CONTROL, CORRECTIONS cards).

This doc provides guidance on:
- How MLMD-EXAFS writes `feff.inp` files from MD snapshots, and the defaults it uses
- How to change each default from the CLI, Python, the MCP server, or the SciLink plug-in
- Selecting appropriate FEFF card parameters for different systems

For the physics behind these choices, see [`exafs-overview.md`](exafs-overview.md). For building FEFF and pointing the workflow at the binary, see [`feff9-setup.md`](feff9-setup.md).

## 1. How the workflow writes `feff.inp` files

`feff.inp` files are written by the `feff-input` stage:

| Interface | Call |
|---|---|
| CLI | `mlmd-exafs feff-input -f <trajectory> -i <absorber index> [options]` |
| Python | `mlmd_exafs.feff.generate_feff_inputs_from_trajectory(trajectory_path, target_atom, hole, rmax, ...)` |
| MCP server / SciLink plug-in | tool `mlmd_feff_input(trajectory_path, target_atom, ...)` |

For every sampled frame, the `carve_out` function (in `mlmd_exafs/feff.py`) handles the geometric operations: building a supercell, centering the absorber at the origin, and identifying all neighbors within a cutoff radius. The cards are then assembled into one `feff.inp` per frame.

Output layout, next to the trajectory:

```
<trajectory dir>/exafs_<El>_hole<hole>_de_<vrcorr>_s02_<S02>_rc_<RMAX>/
├── 000000_<index>/feff.inp      # one subdirectory per sampled frame: <frame>_<absorber index>
├── 000250_<index>/feff.inp
├── ...
└── neighborhoods_<index>.xyz    # carved cluster for every sampled frame (extxyz)
```

`<El>` is the absorber's element and `<vrcorr>` is the first `--corrections` value (`0.0` if omitted). A `feff.inp` generated with all defaults for a Zn absorber in ZnO looks like this:

```
* label:md_out/relaxed/exafs_Zn_hole1_de_0.0_s02_1.0_rc_6.0/feff_run000000_0.inp:label
* data_dir:md_out/relaxed/exafs_Zn_hole1_de_0.0_s02_1.0_rc_6.0:data_dir
* frame: 0 :frame  aindex: 0 :aindex
TITLE relaxed frame 0

HOLE 1 1.000000
CONTROL 1 1 1 1 1 1
PRINT   0 0 0 0 0 0

RMAX 6.0000
SCF 6.0 0 30 0.2 1

POTENTIALS
*  IPOT     Z     tag
      0     30     Zn0
      1      8      O
      2     30     Zn

ATOMS
*      X           Y           Z      IPOT    NN-DIST
     0.000000    0.000000    0.000000    0    0.000000
     0.000000    1.876388    0.622917    1   1.977083
     ...
END
```

## 2. Workflow defaults and how to change them

### 2.1 Configurable parameters

The Python keyword, MCP tool argument, and SciLink tool argument all have the same name.

| FEFF card / setting | Default | CLI option | Python / MCP / SciLink argument |
|---|---|---|---|
| Absorbing atom | required | `-i`, `--target-atom` | `target_atom` |
| `HOLE` edge index | `1` (K edge) | `--hole` | `hole` |
| S₀² (second `HOLE` value) | `1.0` | `--s02` | `s02` |
| `RMAX` (path cutoff, Å) | `6.0` | `--rmax` | `rmax` |
| `SCF` | `6.0 0 30 0.2 1` | `--scf "<values>"` | `scf` |
| `CONTROL` | `1 1 1 1 1 1` | `--control "<values>"` | `control` |
| `CORRECTIONS` | omitted | `--corrections "<vrcorr> <vicorr>"` | `corrections` (`None` omits the card) |
| `PRINT` | `0 0 0 0 0 0` | not available | Python only: `print_flags` |
| Frame stride | every 250th frame | `--step-size` | `step_size` |
| First sampled frame | `0` | `--sampling-start` | `sampling_start` |

The Python function has no defaults for `hole` and `rmax`. Pass them explicitly (the CLI, MCP, and SciLink defaults are `1` and `6.0`).

Examples. CLI, for an L3 edge with a larger path cutoff and a fixed E₀ correction:

```bash
mlmd-exafs feff-input \
    -f md_out/relaxed/relaxed.xyz \
    --target-atom 0 \
    --hole 4 \
    --rmax 8.0 \
    --scf "5.5 0 30 0.05 10" \
    --s02 0.9 \
    --corrections "2.0 0.0" \
    --step-size 100 \
    --sampling-start 1000
```

Python:

```python
from mlmd_exafs.feff import generate_feff_inputs_from_trajectory

gen = generate_feff_inputs_from_trajectory(
    "md_out/relaxed/relaxed.xyz", target_atom=0, hole=4, rmax=8.0,
    scf="5.5 0 30 0.05 10", s02=0.9, corrections="2.0 0.0",
    step_size=100, sampling_start=1000,
)
```

MCP / SciLink: pass the same names as tool arguments, e.g. `{"trajectory_path": "...", "target_atom": 0, "hole": 4, "rmax": 8.0}`.

### 2.2 Fixed behavior

These are not options. Changing them requires editing `mlmd_exafs/feff.py` (`carve_out`, `_build_feff_inp`) or editing the generated `feff.inp` files before `run-feff`.

| Behavior | Value |
|---|---|
| Cluster (carve) radius | `RMAX + 2.5` Å |
| Hydrogen | Removed from the cluster |
| Potentials | Absorber is potential 0; one potential per element in the cluster, numbered in order of atomic number |
| Cards never written | `EXAFS`, `FMS`, `DEBYE`, `EDGE`, `S02`, `XANES`, `NLEG`, `CRITERIA` (FEFF defaults apply) |

## 3. Cards the workflow does not write

- **`FMS`.** Without it, χ(k) comes from the path expansion, which is the recommended method for EXAFS. Full multiple scattering loses accuracy at high k. FMS is still used internally by the `SCF` step to build the potentials; that is intended.
- **`DEBYE`.** Without it, FEFF applies σ² = 0 to every path. In MLMD-EXAFS all thermal and static disorder comes from averaging χ(k) over MD snapshots. Do not add a `DEBYE` card: it would count the disorder twice and damp the spectrum too much.
- **`EDGE`.** The workflow uses the older `HOLE` card, which FEFF9 still accepts. `HOLE` also carries S₀².
- **`EXAFS`, `NLEG`, `CRITERIA`.** FEFF's defaults for the k range, number of legs per path, and path-importance filters are used. The number of paths grows quickly with `RMAX`.

## 4. Edge and `HOLE` card selection

Workflow default: `--hole 1` (K edge). Change with `--hole` / `hole`.

| Element type | HOLE index | Edge | Notes |
|-------------|-----------|------|-------|
| 3d metals (Ti-Zn) | 1 | K | Standard choice |
| 4d metals (Zr-Cd) | 4 | L3 | Higher cross-section than K |
| 5d metals (Hf-Hg) | 4 | L3 | K-edge too high for most beamlines |
| Lanthanides | 4 | L3 | K accessible but L3 preferred |
| Light elements (C-Si) | 1 | K | Only option |

The HOLE index must match the edge that was measured. The other indices are 2 = L1 and 3 = L2.

## 5. S₀²

Workflow default: `--s02 1.0`, written as the second value of the `HOLE` card. Change with `--s02` / `s02`.

Typical values are 0.8–1.0. In MLMD-EXAFS, S₀² scales every snapshot equally and nothing downstream fits it: `lcf` has no free amplitude, and its weights must sum to 1. Use a physically reasonable value, or the value from a path fit of a reference compound, and use the same value for every LCF standard. A value below 0.1 makes FEFF estimate S₀² from atomic overlap integrals.

## 6. `CONTROL` card by calculation goal

Workflow default: `--control "1 1 1 1 1 1"`. Change with `--control` / `control`.

The six flags switch the FEFF modules on or off, in order: potentials (`pot`), phase shifts (`xsph`), full multiple scattering (`fms`), path enumeration (`path`), path XAFS parameters (`genfmt`), and χ assembly (`ff2x`).

| Goal | CONTROL | Notes |
|------|---------|-------|
| Full calculation | `1 1 1 1 1 1` | All modules: potentials, phases, FMS, paths, FEFF, chi |
| EXAFS only (skip XANES) | `1 1 0 1 1 1` | Skip module 3 (FMS, used for XANES) |
| Re-run chi only | `0 0 0 0 0 1` | Use existing potentials/phases (E0 fitting) |
| Potentials only | `1 0 0 0 0 0` | Check SCF convergence |

In the workflow:

- Every snapshot has a different geometry, so every snapshot needs its own potentials and phase shifts. Keep the first two flags on.
- The partial reruns (`0 0 0 0 0 1`, ...) need the intermediate files from a previous run. `run-feff` deletes these by default, so run it with `--no-cleanup` (`cleanup=False`) if you plan to rerun modules. E₀ is fitted after averaging (`fit-e0`, `lcf`), so rerunning `ff2x` is not needed for E₀.

## 7. `RMAX` selection

Workflow default: `--rmax 6.0` Å. Change with `--rmax` / `rmax`. FEFF9 treats `RMAX` as `RPATH`: the maximum half path length.

| Analysis goal | RMAX (Å) | Notes |
|--------------|----------|-------|
| First shell only | 4.5 | Fast, adequate for coordination number |
| Standard 2-4 shells | 6.0 | Good balance of information and speed |
| Extended with heavy scatterers | 8.0 | Needed for multiple-scattering paths |

- The number of paths grows roughly exponentially with `RMAX`, and the cost is paid once per snapshot. Start small, inspect the result, and increase gradually.
- `RMAX` should cover at least the R range you will compare with experiment (the upper end of the Fourier-transform window in `convergence` or your own analysis).
- The carved cluster is `RMAX + 2.5` Å, so a larger `RMAX` needs a larger MD cell.

## 8. `SCF` card parameters

Workflow default: `--scf "6.0 0 30 0.2 1"`. Change with `--scf` / `scf`.

The values are `rfms1 lfms1 nscmt ca nmix`: the cluster radius for the self-consistent potentials (Å), `0` for a solid or `1` for a molecule, the maximum number of iterations, the convergence accelerator, and the number of mixing iterations.

| System type | SCF card | Notes |
|------------|----------|-------|
| Standard metals/oxides | `6.0 0 30 0.2 1` | 6 Å radius, 30 iterations |
| f-electron systems | `5.5 0 30 0.05 10` | Improves run stability |
| Molecular systems | `4.0 0 30 0.2 1` | Smaller cluster radius adequate |

- Self-consistent potentials have a small effect on the EXAFS oscillations but give a more accurate E₀ and more reliable phase shifts. Keep `SCF` on.
- The SCF radius should include at least the first one or two coordination shells.
- For isolated molecules or clusters in vacuum, set `lfms1` to `1`.

## 9. `CORRECTIONS` card

Workflow default: omitted. Add with `--corrections "<vrcorr> <vicorr>"` / `corrections="<vrcorr> <vicorr>"`.

- Omit initially and fit the E₀ shift post-hoc against experimental data. In MLMD-EXAFS this is done after averaging, by `mlmd-exafs average --exp-file`, `mlmd-exafs fit-e0`, or per standard in `mlmd-exafs lcf`.
- With SCF: expect ±1 eV shift
- Without SCF: expect ±3 eV shift
- Format: `CORRECTIONS vrcorr vicorr` (e.g., `CORRECTIONS 0.0 0.0`). `vrcorr` shifts the Fermi level (E₀) and `vicorr` adds broadening, both in eV.
- If you apply a known shift with `vrcorr`, do not fit E₀ for the same shift again afterwards.
- `vrcorr` appears in the output directory name (`_de_<vrcorr>`); `vicorr` does not.

## 10. The carved cluster

For each sampled frame, `carve_out`:

1. Builds a supercell with an odd number of repeats (at least 3) along each cell vector, large enough to hold a sphere of `RMAX + 2.5` Å.
2. Translates the absorber to the origin and wraps positions.
3. Keeps every non-hydrogen atom within `RMAX + 2.5` Å of the absorber. H atoms are very weak scatterers, so their contribution to EXAFS is negligible. The 2.5 Å margin keeps paths shorter than `RMAX` away from the edge of the cluster.
4. Assigns potentials: 0 for the absorber, then one per element present. Atoms of the absorber's element other than the absorber get their own potential. The default FEFF array limits (`nphx`) are enough unless a snapshot contains an unusually large number of distinct elements (see `feff9-setup.md` §4.4).

Notes:

- If the supercell cannot hold the sphere, `feff-input` stops with `Expanded supercell not large enough for rmax=... A`. Use a larger MD cell or a smaller `--rmax`.
- `--target-atom` is the same atom index in every frame. Check that it is the intended absorber in `neighborhoods_<index>.xyz`.
- For a structure with several inequivalent absorbing sites, run `feff-input` once per site index.

## 11. Changing settings and rerunning

- Changing any card means regenerating the inputs and rerunning FEFF for every snapshot.
- The output directory name encodes only the absorber element, `--hole`, `vrcorr`, `--s02`, and `--rmax`. It does not encode `--scf`, `--control`, `vicorr`, `--step-size`, `--sampling-start`, or `--target-atom`.
  - Rerunning `feff-input` with a change to one of these writes into the *same* directory. Existing `feff.inp` files for the same frames are overwritten, but snapshot directories from the previous run (with their `chi.dat`) are left in place, and `average` includes every subdirectory with a `chi.dat`. Move or delete the old `exafs_*` directory first.
  - Two absorber indices of the same element also share a directory (subdirectories `<frame>_<index>`), so `average` combines both sites into one spectrum. That is correct for equally occupied sites. To keep the sites separate, for example as separate LCF standards, move each site's subdirectories into its own directory before `run-feff`.
- All LCF standards should be computed with the same `--hole`, `--rmax`, `--scf`, `--s02`, and `--corrections`, so differences between standards reflect the structures rather than the settings.

## 12. Common `feff.inp` pitfalls in the workflow

| Pitfall | Why it matters | What to do |
|---|---|---|
| Adding a `DEBYE` card to generated inputs | Disorder is already in the MD average; σ² is counted twice | Leave σ² to the MD (§3) |
| `--hole` does not match the measured edge | Wrong edge, wrong spectrum | Check `HOLE` card selection (§4) |
| Wrong `--target-atom` | The index is fixed across frames; a wrong index simulates the wrong absorber | Check `neighborhoods_<index>.xyz` |
| Unrealistic `--s02` | LCF has no free amplitude, so an S₀² mismatch biases the weights | Use a physical value (§5) |
| Correcting E₀ twice | `--corrections` shifts E₀ inside FEFF, and `fit-e0`/`lcf` shift it again | Use one or the other (§9) |
| `RMAX` too small for the analysis range | Paths beyond `RMAX` are missing from χ(k) | Match `RMAX` to the R range you compare (§7) |
| `RMAX` too large | Path count and FEFF cost grow quickly, per snapshot | Increase gradually (§7) |
| Rerunning into an existing `exafs_*` directory | Stale snapshots from the old settings are averaged in | Move or delete the old directory (§11) |
| Standards with different card settings | Differences reflect settings, not structures | Use identical settings for all standards (§11) |
| Partial `CONTROL` rerun after cleanup | Intermediate files were deleted by `run-feff` | Run `run-feff --no-cleanup` first (§6) |

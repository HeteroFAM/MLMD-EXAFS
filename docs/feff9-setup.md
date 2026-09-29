# Guide: Obtaining, Licensing, and Compiling FEFF9

This guide is based on the [*FEFF9.6 User's Guide* (version 9.6.4, updated February 2, 2013)](https://feff.phys.washington.edu/feff/Docs/feff9/feff90/feff90_users_guide.pdf). Some details may have changed since then; check the [FEFF Project website](https://feff.phys.washington.edu/) for current information.

MLMD-EXAFS runs FEFF through a single executable, by default
`/share/feff/feff90_binaries/feff.x`. This guide builds FEFF from source into that
layout. Section 6 covers how to use a different location.

## 1. Overview

- FEFF is developed by The FEFF Project, Department of Physics, University of Washington, Seattle, WA.
- FEFF9 is written in Fortran 90.
- FEFF is not a Python package. `pip install mlmd-exafs` does not install it. You must obtain, compile, and point the workflow at it yourself.

## 2. Licensing and Purchasing

### 2.1 A license is required

The full FEFF9 code is copyright-protected software. You must obtain a license from the University of Washington Office of Technology Transfer to use it. Both academic/non-profit and commercial licenses are available.

### 2.2 How to get a license

- A FEFF9 license can purchased here: [https://els2.comotion.uw.edu/product/feff](https://els2.comotion.uw.edu/product/feff).
- You will receive a license code. The code gives you access to the [download section](https://feff.phys.washington.edu/feffproject-feff-download.html), where you can download the FEFF source code.

### 2.3 Citation requirement

If you use FEFF or its results in published work, you should cite it:

> J.J. Rehr, J.J. Kas, F.D. Vila, M.P. Prange, K. Jorissen, "Parameter-free calculations of X-ray spectra with FEFF9," *Phys. Chem. Chem. Phys.* **12**, 5503–5513 (2010).

## 3. System Requirements

- **Memory:** FEFF typically needs 250 MB – 2 GB of RAM per calculation. `mlmd-exafs run-feff` runs up to `--max-workers` (default 32) FEFF jobs at once, so size `--max-workers` to your RAM. For example, 32 jobs at 1 GB each need about 32 GB.
- **C shell (csh):** Required. The build script (`Compile`), the `feff.x` driver it generates, and `scripts/run_feff.csh` are all csh scripts. Some Linux distributions (e.g., Ubuntu) ship without csh. Check with:

  ```bash
  which csh              # install with e.g. `sudo apt install csh` or `tcsh` if missing
  ```

- **Fortran 90 compiler:** `gfortran` or Intel `ifx`/`ifort`. FEFF does not link against any external libraries.

  ```bash
  which gfortran ifx
  ```

## 4. Compiling FEFF from Source

The workflow expects this layout (the default everywhere in MLMD-EXAFS):

```
/share/feff/
├── feff90/              # source: <module>_tot.f90 files + Compile script
└── feff90_binaries/     # compiled module executables + feff.x driver
    ├── rdinp  dmdw  atomic  pot  ldos  screen  xsph  fms
    ├── mkgtr  path  genfmt  ff2x  sfconv  eels  dym2feffinp
    └── feff.x           # csh driver that runs the modules in order
```

`feff.x` is the file the workflow calls. It is a csh script that runs each FEFF module in turn, using absolute paths, from the current working directory. The modules read `feff.inp` from that directory and write `chi.dat` and the other outputs next to it.

If you use a different prefix, replace `/share/feff` below and see [Section 6](#6-pointing-the-workflow-at-your-feff-binary).

### 4.1 Option A: Modular source (`<module>_tot.f90` + `Compile`)

This is how the default `/share/feff` install is built. The modular source has one self-contained `<module>_tot.f90` file per FEFF module and a csh `Compile` script.

1. Put the source in `/share/feff/feff90`:

   ```bash
   mkdir -p /share/feff/feff90
   # copy/unpack the modular source (*_tot.f90 and Compile) into it
   cd /share/feff/feff90
   ```

2. Edit `Compile` to set the compiler. It defines two compiler lines, but only the one named `F77` is used by the build loop:

   ```csh
   set noF77 = 'ifx -stand f95 -fp-model precise -assume byterecl -assume buffered_io -assume old_maxminloc -fpe0 -traceback -O2'
   set F77 = 'gfortran -std=legacy -O3 -ff2c'
   ```

   To build with Intel `ifx`, swap the two variable names so the `ifx` line is `F77`.

3. Run it:

   ```bash
   csh ./Compile
   ```

   `Compile` does the following:

   - creates `../feff90_binaries` (i.e. `/share/feff/feff90_binaries`) if it does not exist.
   - compiles each module (`rdinp pot ldos xsph fms path genfmt mkgtr atomic ff2x screen sfconv eels dmdw dym2feffinp`) into it.
   - writes `feff.x` there, containing the absolute path of every module.

   The directory relationship is fixed: binaries always go to the sibling `feff90_binaries` directory of wherever you run `Compile`.

4. `Compile` ends with `sudo chmod 777 feff.x`. You do not need `sudo` or mode 777 if you own the directory. If `sudo` is not available, the chmod fails, and you can run this instead:

   ```bash
   chmod 755 /share/feff/feff90_binaries/feff.x
   ```

5. Because `feff.x` contains absolute paths, moving `feff90_binaries` after building breaks it. If you relocate it, re-run `Compile` from the new location or edit the paths in `feff.x`.

### 4.2 Option B: Full source tree (`feff90.tar.gz`)

1. Unpack the archive and go to `feff90/src`.
2. Open `Compiler.mk` and set the Fortran 90 compiler and its compilation options.
3. Build the serial version:

   ```bash
   make
   ```

4. The executables appear in `feff90/bin/Seq`. This build does not create the `feff.x` driver the workflow calls, so create one yourself. Put it in the same layout as Option A:

   ```bash
   mkdir -p /share/feff/feff90_binaries
   cp feff90/bin/Seq/* /share/feff/feff90_binaries/
   cd /share/feff/feff90_binaries
   echo '#!/bin/csh -f' > feff.x
   for m in rdinp dmdw atomic pot ldos screen xsph fms mkgtr path genfmt ff2x sfconv eels; do
       echo "$PWD/$m" >> feff.x
   done
   chmod 755 feff.x
   ```

   The module order above matches the driver `Compile` generates. Only include modules that exist in `bin/Seq`.

5. Always run `make clean` when switching between serial and MPI builds. Otherwise the executables may crash.
6. With a source-tree install, the example calculations are in `feff90/examples`.

### 4.3 Compiler notes

- The FEFF developers built with Intel Fortran. gfortran, g95, pgf90, and other compilers may also work. The default `/share/feff` build uses `gfortran -std=legacy -O3 -ff2c`.
- Recommended approach:
  1. Start with a conservative build: all optimization disabled and maximum floating-point safety flags.
  2. Build a faster version and check its results against the conservative build.
  3. Test with a case that uses the SCF and FMS cards, to confirm the FMS routines compiled safely. The workflow always writes an `SCF` card (`mlmd-exafs feff-input --scf`, default `"6.0 0 30 0.2 1"`), so this path is exercised in every run.
- On some machines you may need to increase the stack size. Set it in the shell you launch the workflow from; FEFF jobs started by `mlmd-exafs run-feff` or `run_feff.csh` inherit it:

  ```bash
  ulimit -s unlimited          # bash, before `mlmd-exafs run-feff`
  limit stacksize unlimited    # csh/tcsh, before `source scripts/run_feff.csh`
  ```

### 4.4 Array dimensions (only if needed)

- The two main array sizes, `nclusx` and `lx`, are set automatically at run time.
- Other limits are fixed at compile time. These include the maximum number of potentials (`nphx`) and spin states (`nspx`).
- To change them, edit `feff90/src/COMMON/m_dimsmod.f90` (full source tree) and recompile.
- After changing these values, intermediate files from earlier runs become unusable. Rerun the affected calculations.
- `mlmd-exafs feff-input` writes one potential per unique element in the carved cluster, plus the absorber. The default `nphx` is enough unless a snapshot contains an unusually large number of distinct elements.

### 4.5 MPI (parallel) builds

MLMD-EXAFS gets its parallelism by running many serial FEFF jobs at once, one per MD snapshot (`--max-workers` in `run-feff`, `max_num_processes` in `run_feff.csh`). Build the serial version. An MPI build (`make mpi`, executables in `feff90/bin/MPI`) is not used by the workflow scripts.

## 5. Verify the Build

1. Check that the driver exists and is executable:

   ```bash
   ls -l /share/feff/feff90_binaries/feff.x
   head -3 /share/feff/feff90_binaries/feff.x      # should list absolute module paths
   ```

2. Run a small test the same way the workflow does, from inside a job directory with `feff.inp` as the argument:

   ```bash
   mkdir -p /tmp/feff_test && cd /tmp/feff_test
   cat > feff.inp <<'EOF'
   TITLE Cu test
   HOLE 1 1.0
   CONTROL 1 1 1 1 1 1
   PRINT 0 0 0 0 0 0
   RMAX 3.0
   POTENTIALS
    0 29 Cu
    1 29 Cu
   ATOMS
    0.0    0.0    0.0   0
    1.805  1.805  0.0   1
   -1.805 -1.805  0.0   1
    1.805  0.0    1.805 1
   -1.805  0.0   -1.805 1
    0.0    1.805  1.805 1
    0.0   -1.805 -1.805 1
   END
   EOF
   /share/feff/feff90_binaries/feff.x feff.inp > feff.out 2>&1; echo "rc=$?"
   tail -1 feff.out     # " Done with module 6: DW + final sum over paths."
   head -1 chi.dat      # FEFF version banner
   ```

   A zero return code and a `chi.dat` mean the build works with the workflow. `mlmd-exafs run-feff` treats a non-zero return code as a failed job and lists it under `failed`.

3. With the full source tree, you can also run the examples in `feff90/examples` and compare against their reference outputs (`referencexmu.dat`, `REFERENCE.zip`).

## 6. Pointing the Workflow at Your FEFF Binary

Every entry point defaults to `/share/feff/feff90_binaries/feff.x`. If you built FEFF there, no configuration is needed. Otherwise, pass the path to `feff.x` explicitly on each call. The workflow does not read an environment variable or search `PATH`.

| Entry point | How to set the FEFF path |
|-------------|--------------------------|
| CLI | `mlmd-exafs run-feff -d <exafs_dir> --feff-bin /path/to/feff90_binaries/feff.x` |
| Python | `run_feff_batch(dir, feff_bin="/path/to/feff90_binaries/feff.x")` |
| MCP server | `mlmd_run_feff` tool argument `feff_bin` |
| SciLink plug-in | `mlmd_run_feff` tool argument `feff_bin` |
| csh script | Edit `set FEFF_BIN=...` at the top of `scripts/run_feff.csh` (the script sets it unconditionally, so an environment variable does not override it) |

Example:

```bash
mlmd-exafs run-feff \
    -d md_out/relaxed/exafs_Zn_hole1_de_0.0_s02_1.0_rc_6.0 \
    --feff-bin $HOME/feff/feff90_binaries/feff.x \
    --max-workers 16
```

Requirements for the path:

- **Point it at the `feff.x` driver**, not at the `feff90_binaries` directory or a single module such as `rdinp`. The workflow runs `<feff_bin> feff.inp` once per snapshot directory and needs all modules to run.
- **Use an absolute path.** `run-feff` checks that the file exists relative to the directory you launch from, but runs FEFF from inside each snapshot directory. A relative path like `./feff.x` passes the check and then fails with `FileNotFoundError` when the jobs start.
- If the path is wrong, `run-feff` stops before running anything with: `FEFF binary not found at '<path>'. Pass --feff-bin with the correct path.`

What the workflow does with the binary:

- For every subdirectory of the `exafs_*` directory that contains a `feff.inp`, it runs `feff.x feff.inp` in that subdirectory, with stdout and stderr captured to `feff.out`.
- When all jobs finish, it deletes FEFF's scratch files (`*.bin`, `log*.dat`, `paths.dat`, `xmu.dat`, ...). It keeps `feff.inp`, `feff.out`, `chi.dat` and `neighborhoods_*.xyz`. Pass `--no-cleanup` (or `cleanup=False`) to keep everything. This is useful when debugging a FEFF build. For the csh script, `set CLEANUP=0` before sourcing it.

## 7. Troubleshooting and Support

| Symptom | Likely cause / fix |
|---------|--------------------|
| `FEFF binary not found at ...` | Wrong `--feff-bin`; point it at the absolute path of `feff.x` (Section 6) |
| `FileNotFoundError` once jobs start | Relative `--feff-bin` path (use an absolute path), or csh not installed (`feff.x` starts with `#!/bin/csh -f`; see Section 3) |
| `PermissionError` once jobs start | `feff.x` not executable; `chmod 755 feff.x` (Section 4.1, step 4) |
| Every job fails, `feff.out` shows `Command not found` for a module | `feff90_binaries` moved after `Compile`; re-run `Compile` or fix the paths in `feff.x` |
| Jobs crash in `fms`/`pot` with a segmentation fault | Stack size; `ulimit -s unlimited` before launching (Section 4.3) |
| Machine runs out of memory during `run-feff` | Lower `--max-workers` |
| `run-feff` succeeds but `average` skips snapshots | FEFF stopped before writing `chi.dat`; rerun with `--no-cleanup` and inspect that snapshot's `feff.out` and `log*.dat` |
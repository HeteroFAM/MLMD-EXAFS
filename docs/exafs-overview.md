# EXAFS: An Overview of Simulation and Interpretation

This overview draws on [B. Ravel, "Quantitative EXAFS Analysis," in *X-Ray Absorption and X-Ray Emission Spectroscopy: Theory and Applications* (Wiley, 2016)](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=915832) and the [*FEFF9.6 User's Guide*](https://feff.phys.washington.edu/feff/Docs/feff9/feff90/feff90_users_guide.pdf) to explain concepts and general strategies for quantitative EXAFS analysis.

## 1. What EXAFS Is

Extended X-ray Absorption Fine Structure (EXAFS) is the oscillatory structure in an X-ray absorption spectrum well above an absorption edge. Analyzing it reveals the local structure around the absorbing atom:

- interatomic distances (R)
- mean-square deviations in distance (σ², the Debye–Waller-type disorder term)
- coordination numbers (N)
- the species of coordinating atoms

Key strengths:

- *No symmetry or periodicity assumed.* Neither the theory nor its interpretation requires them, so EXAFS works equally well for crystals, glasses, liquids, solutions, and biological samples.
- *Element specific.* The absorption process photo-excites a deep core electron of one particular element, so EXAFS works even in dilute systems.
- *Broad applicability.* It is used in life science, chemistry, materials science, geology, and more.

## 2. XANES vs. EXAFS

From a real-space multiple-scattering viewpoint, there is no fundamental physical boundary between the near-edge region (XANES) and EXAFS. In practice, the EXAFS region is the part of the spectrum where EXAFS analysis can be done reliably.

| Definition | Where EXAFS begins | Notes |
|---|---|---|
| Strict mathematical criterion (convergence of a matrix in the multiple-scattering formalism) | ~16 eV above the edge in Cu metal (~2 Å⁻¹) | Rigorous but not very practical |
| Practical rule of thumb | 25–35 eV above the edge (~2.5–3 Å⁻¹) | Path expansion converges; results are insensitive to potential details; background removal is stable |

## 3. From Measured Spectrum to χ(k) and χ(R)

The basic data-processing steps have been used since the 1970s:

1. *Background removal.* Subtract the slowly varying atomic background from the measured absorption μ(E) to isolate the fine structure χ(E).
2. *Convert to wavenumber.* Change the energy axis to photoelectron wavenumber:

   $$k = \frac{\sqrt{2m_e (E - E_0)}}{\hbar}$$

3. *Weight by kⁿ.* Multiply χ(k) by kⁿ (typically n = 1, 2, or 3) so the oscillations are similar in size across the whole data range.
4. *Fourier transform.* The complex transform χ(R) resembles a pair distribution function, but it is not one. Peaks are shifted and broadened by phase shifts and the finite k range.

> Caution: Peaks in |χ(R)| often overlap. In BCC iron, for example, the first two shells (8 Fe at ~2.48 Å and 6 Fe at ~2.86 Å) overlap heavily, even though the structure is simple and well ordered. Older methods that isolate a single shell by Fourier filtering fail in such cases. This is a major reason modern analysis relies on theoretical models.

## 4. The EXAFS Equation

Each scattering path Γ (single or multiple scattering) contributes:

$$\chi_\Gamma(k) = \frac{S_0^2 N_\Gamma F_\Gamma(k)}{2kR_\Gamma^2}\, e^{-2k^2\sigma_\Gamma^2}\, e^{-R_\Gamma/\lambda_\Gamma(k)}\, \sin\!\big(2kR_\Gamma + \delta_\Gamma(k)\big)$$

The total theoretical signal is the sum over paths:

$$\chi(k) = \sum_\Gamma \chi_\Gamma(k)$$

| Term | Meaning | Source |
|---|---|---|
| $F_\Gamma(k)$ | Effective scattering amplitude | Theory |
| $\delta_\Gamma(k)$ | Total phase shift (central + scattering atoms) | Theory |
| $\lambda_\Gamma(k)$ | Photoelectron mean free path | Theory |
| $S_0^2$ | Amplitude reduction factor (many-body/intrinsic losses) | Modeled / fitted |
| $N_\Gamma$ | Path degeneracy (e.g., coordination number) | Modeled / fitted |
| $\sigma_\Gamma^2$ | Mean-square deviation in half-path length | Modeled / fitted |
| $R_\Gamma = R_{0,\Gamma} + \Delta R_\Gamma$ | Half-path length (starting value + fitted shift) | Modeled / fitted |
| $E_0$ | Aligns the theory energy grid with the data | Fitted |

*Adapted from Table 11.1 of [Ravel (2016)](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=915832).*

> *Notation note:* Factor conventions differ between references. The FEFF User's Guide, for example, writes the prefactor as $N F_{\text{eff}}/(kR^2)$ and the loss term as $\exp(-2r/\lambda)$. The physics is the same; the factors are absorbed into how each quantity is defined.

What the equation explains:
- Oscillations come from the sine term, which depends on distance R.
- Locality comes from the mean-free-path term and the 1/R² falloff, which suppress distant contributions.
- Thermal and static disorder damp the signal at high k through the $e^{-2k^2\sigma^2}$ term.

## 5. Simulating EXAFS: The Theory

### 5.1 Why theoretical standards?

Early quantitative EXAFS was empirical. It used measured reference compounds, log-ratio methods, or Fourier-filtered shells. These approaches have two weaknesses:
- the shell of interest must be isolable in R-space
- a suitable measured standard must exist

That rules out metastable, in situ, operando, doped, mixed-phase, or surface-sorbed samples, and unusual absorber/scatterer pairs.

Theoretical scattering factors remove these limitations. They also provide a rigorous way to include multiple scattering and more distant shells.

### 5.2 Main theory codes

By the mid-1990s, three user-oriented packages dominated:

| Code | Approach | Notes |
|---|---|---|
| EXCURVE | Exact curved-wave theory | Originally limited to single, double, and triple scattering because of computational cost. The first EXAFS code in commercial software with a GUI. Early work on constrained fitting and EXAFS/diffraction corefinement |
| GNXAS | *n-body decomposition*. Computes scattering to all orders within each sub-cluster of n atoms (γ₂, γ₃, …) | Free of cost (registration required). Theory and fitting are tightly integrated |
| FEFF | *Path expansion*. Enumerates individual single- and multiple-scattering paths | Most widely adopted. Its output feeds many independent analysis programs |

All three represent the signal as a sum of geometric contributions and use muffin-tin scattering potentials.

### 5.3 The muffin-tin potential

- Each atom is represented by a spherical potential that touches, or slightly overlaps, its neighbors. The interstitial region between spheres is treated as flat, with uniform charge density.
- This is a crude picture of a chemical bond. Far above the edge, however, the photoelectron has enough kinetic energy to be mostly insensitive to potential details. That is why the muffin-tin approximation works well for EXAFS, even though XANES often needs non-muffin-tin potentials.
- Curved-wave treatment of the photoelectron is essential. The early plane-wave approximation performs poorly when the signal is dominated by low-k data.

### 5.4 Multiple-scattering formalism (in brief)

Absorption follows Fermi's golden rule. It is solved in practice using the Green's function (propagator) $\mathbb{G}$ for the muffin-tin Hamiltonian:

$$\mathbb{G} = G_0 + G_0 t G_0 + G_0 t G_0 t G_0 + \cdots$$

- $G_0$ is the free-electron propagator, and $t$ is the single-atom scattering matrix.
- The second term covers all single-scattering events, the third all double-scattering events, and so on.
- Each theory code expands these terms into scattering configurations in its own way. That choice is the main difference between the codes.

### 5.5 The FEFF path expansion and pathfinder

- *Path enumeration.* FEFF first finds all single-scattering (SS) paths. It then builds double-scattering (DS) paths by extending each SS path to every other atom, and continues to higher orders. Consecutive scattering from the same atom is not allowed, and paths are truncated at eight legs.
- *Sorting and degeneracy.* Paths are sorted by half-path length (half the sum of the leg lengths). They are then checked for degeneracy, including equivalent geometry, time-reversal symmetry, and mirror symmetry. Polarization effects can also be included.
- *Importance filtering.* Each path gets an importance factor based on the integrated magnitude of its χ(k). Paths with little spectral weight are discarded. In FEFF9, a fast plane-wave criterion filters paths during the search, and a curved-wave criterion is applied afterward.
- *Practical truncation.* The fitted R range rarely exceeds about 6 Å, and paths with sharp scattering angles are weak. So the sum usually includes only a few to a few dozen significant paths.

> *Path expansion vs. n-body decomposition:* FEFF calculates each path individually. GNXAS calculates their sum for each n-atom configuration. Individual paths mean more terms to handle, but they allow a more careful treatment of σ² for multiple-scattering paths.

### 5.6 The input structure matters

- Theory codes need a list of atomic species and Cartesian coordinates. This can come from crystal structures, molecular structures, or first-principles or molecular-dynamics output.
- Physically reasonable distances give good muffin-tin potentials and defensible results. Unphysical separations make the calculation unusable.
- Simplified first-shell models (for example, a rock-salt cluster built from a guessed absorber–scatterer distance) work well for the first shell. Used for more distant shells, they produce unphysical muffin-tin radii and significant errors.
- FEFF can also read `.cif` files, and third-party tools such as ATOMS can generate input from crystallographic data.

## 6. Simulation Notes Specific to EXAFS calculations with FEFF9

MLMD-EXAFS does not run one FEFF calculation on one static structure. It samples snapshots from a molecular-dynamics (MD) trajectory, runs FEFF on the local cluster around the absorber in each snapshot, and averages the resulting χ(k):

```
relax → md → feff-input → run-feff (+ cleanup) → average (+ fit-e0) → plot / convergence → lcf
```

This changes how several of the usual FEFF choices apply. The exact `feff.inp` cards the workflow writes, their defaults, and how to change them are in [`feff-input-recs.md`](feff-input-recs.md).

### 6.1 FEFF's calculation chain

- FEFF runs as a chain of modules: potentials (`pot`), then phase shifts (`xsph`), then path enumeration (`path`), then per-path XAFS parameters (`genfmt`), then assembly of χ(k) (`ff2x`).
- Potentials and `genfmt` are slow. `ff2x` is fast. For a single static structure, Debye–Waller factors, S₀², or E₀ shifts can be explored by rerunning only `ff2x`.
- In MLMD-EXAFS every snapshot has a different geometry, so the potentials and phase shifts are recalculated for every snapshot. The cost of the whole chain is paid once per snapshot.
- *Path expansion vs. FMS.* The path expansion is the recommended method for extended spectra, and it is what the workflow uses. Full multiple scattering (FMS) loses accuracy at high energy.
- *Path length.* The number of paths grows exponentially with path length. Start with a short cutoff and increase it gradually.
- *Self-consistent potentials* (`SCF`) have a small effect on EXAFS itself, but they give a more accurate E₀ and more reliable phase shifts. The workflow uses them by default.

### 6.2 The input cluster

Each snapshot is turned into a finite cluster centered on the absorber, cut from the periodic MD cell. The cluster extends a margin beyond the longest path, so no path runs into the edge of the cluster. Hydrogen atoms are left out; they are very weak scatterers, so their contribution to EXAFS is negligible. The input structure matters as described in §5.6, with the difference that the structure here comes from the MLIP rather than from a crystal structure or a guessed model.

### 6.3 Disorder from MD sampling

- *Where σ² comes from.* In the EXAFS equation (§4), σ² describes the damping caused by the spread of absorber–scatterer distances. FEFF can estimate it from a model such as the correlated Debye model, which works best for homogeneous systems and can be off by a factor of two or more in heterogeneous or anisotropic ones. MLMD-EXAFS does not use these models. It calculates each snapshot with σ² = 0 and averages the spectra over an NVT MD trajectory. The spread of distances in the trajectory then produces the damping directly, including anharmonic and non-Gaussian effects that a single σ² per path cannot represent.
- *Classical nuclei.* MD treats the nuclei classically, so zero-point motion is missing. At low temperature, or for bonds to light atoms, the simulated disorder is too small and the high-k amplitude is too large.
- *The potential sets the structure.* Bond lengths, and therefore the phase of every path, come from the machine-learned interatomic potential (MLIP). An error of a few hundredths of an Å in a bond length shifts the phase noticeably at high k.
- *Sampling.* Early frames, before the system reaches the target temperature, bias the average and should be skipped. Closely spaced frames are correlated, so they add cost without adding information.
- *Convergence.* The average is converged when adding snapshots no longer changes k²χ(k) or |χ(R)|. The spread of the snapshot spectra (standard deviation and standard error of the mean) describes MD sampling. It is not an error bar on the theory.

### 6.4 E₀ and S₀²

- *E₀.* FEFF places E₀ from its own potentials, typically a few eV away from the experimental edge. In MLMD-EXAFS, E₀ is fitted after averaging, against the experimental spectrum (§7.6), using the Artemis/IFEFFIT sign convention. A known shift can instead be applied inside FEFF, but the same shift should not be applied twice.
- *S₀².* S₀² scales every snapshot equally and is not fitted afterwards, because LCF has no free amplitude (§7.6). It should be set to a physically reasonable value (typically 0.7–1.0, §7.5) or to the value from a path fit of a reference compound.

> *Licensing note:* A free, redistributable version of FEFF6 suitable for EXAFS analysis has been available since 2002. Later versions of FEFF require a license fee. MLMD-EXAFS requires FEFF9.

## 7. Fitting Theory to Data

There are two broad ways to compare theory with data:

- *Path fitting* (§7.1–7.5). Adjust the EXAFS-equation parameters (S₀², E₀, ΔR, σ², N) of individual FEFF paths until the sum matches the data. Artemis/IFEFFIT and Larch work this way.
- *Linear combination fitting (LCF)* (§7.6). Treat several complete theoretical spectra as fixed "standards" and fit only how much of each is present. This is the method MLMD-EXAFS uses (`mlmd-exafs lcf`, `mlmd_exafs.lcf.run_lcf`).

### 7.1 The fitting metric

Many FEFF-based packages use Levenberg–Marquardt non-linear least squares.

- Difference function in k-space:

  $$f(k_i) = \chi(k_i|\text{data}) - \chi(k_i|\text{theory})$$

- In R-space, the difference uses both the real and imaginary parts of the Fourier-transformed data and theory.
- Fitting metric:

  $$\chi^2 = \frac{N_{idp}}{N}\sum_{i=1}^{N}\left(\frac{f_i}{\epsilon_i}\right)^2$$

  Here $\epsilon_i$ is the measurement uncertainty and N is the number of data points.
- Independent points. $N_{idp}$ is the information content of the data, estimated from the Nyquist criterion as:

  $$N_{idp} \approx \frac{2\,\Delta k\,\Delta R}{\pi}$$

  Data are heavily oversampled: N is typically a few hundred, while $N_{idp}$ is often only around 10–20.

> The letter χ does double duty. χ² is the fitting metric, while χ(k) and χ(R) are the EXAFS signal.

### 7.2 Constraints encode prior knowledge

Floating N, S₀², σ², E₀, and ΔR independently for every path quickly exceeds the information content of the data. Constraints are essential beyond the first shell. In a constrained model, the EXAFS-equation parameters are *written in terms of* a smaller set of fit parameters.

| Constraint type | Example |
|---|---|
| Central-atom properties | One S₀² and one E₀ shared by all paths |
| Lattice expansion (cubic systems) | $\Delta R_\Gamma = \alpha \cdot R_{0,\Gamma}$ for every path, using a single fitted α |
| Thermal models | σ² expressed through an Einstein temperature. For monatomic crystals, all σ² values can come from a single Debye temperature |
| Collinear multiple scattering | σ² of collinear MS paths expressed through σ² of related SS paths, which cuts many paths to two σ² parameters |
| Geometry | Second-shell distances computed trigonometrically from a fitted first-shell distance in a rigid ring |
| Stoichiometry / coordination | Coordination numbers tied to each other or to independently measured compositions |

### 7.3 Judging fit quality

*R-factor.* This is the fractional misfit: the sum of squared residuals divided by the sum of squared data. It tells you how closely the fit overplots the data, not whether the fit is physically sensible.

*Reduced chi-square*:

$$\chi^2_\nu = \frac{\chi^2}{\nu}, \qquad \nu = N_{idp} - N_{var}$$

- *Its main value is comparing models.* Suppose you add a parameter, such as separate σ² values for two split oxygen shells in hematite. If $\chi^2_\nu$ drops significantly, the data support the extra parameter. If it barely changes or rises, they do not.
- *It is a poor absolute measure.* The uncertainty $\epsilon$ is usually estimated from the noise in |χ(R)| between 15 and 25 Å. That estimate assumes statistical noise dominates. At synchrotrons, systematic and instrumental errors, plus errors in the theory, usually dominate instead. So ε comes out too small, and $\chi^2_\nu$ comes out too large, even for good fits.

Worked example: gold foil, first shell

| Quantity | Value |
|---|---|
| N | fixed at 12 |
| S₀² | 0.85(4) |
| E₀ | 5.56(31) eV |
| ΔR | −0.023(3) Å |
| σ² | 0.00824(34) Å² |
| R-factor | 0.0070 (below 1% misfit) |
| χ²_ν | 754 |

This is a clearly good fit, despite its very large $\chi^2_\nu$.

### 7.4 Error bars and correlations

- *Error bars.* Uncertainties come from the diagonal of the fit's covariance matrix, scaled by $\sqrt{\chi^2_\nu}$. If the fit is defensible, these scaled values are reasonable 1σ uncertainties.
- *Correlations are normal.* Amplitude-type parameters (S₀² and σ²) and phase-type parameters (E₀ and ΔR) are usually strongly correlated. In the gold example, S₀² and σ² were more than 80% correlated, and E₀ and ΔR were nearly 70% correlated.

### 7.5 Sanity checks: is the result defensible?

A defensible fit has a small R-factor and $\chi^2_\nu$, and physically sensible parameters:

- S₀² should not be much smaller than ~0.7 or much larger than ~1.0.
- ΔR should not produce unphysical interatomic distances.
- σ² should not be negative.
- E₀ shifts should not be so large that the fit may have fallen into a false minimum.

### 7.6 Linear combination fitting of simulated spectra (MLMD-EXAFS)

LCF answers a different question from path fitting. Path fitting asks which structural parameters of one model best explain the data. LCF asks which mixture of *candidate structures* best explains the data. Examples of candidate structures are different adsorption sites, protonation states, or substitution sites. LCF is the spectrum-level counterpart of §8.1. Each environment gets its own theory calculation, but whole spectra are combined instead of paths.

#### The standards

Each standard $\chi_j(k)$ is a complete theoretical spectrum for one candidate structure, typically the MD-averaged `<savefile>-chi_avg.dat` from `mlmd-exafs average`. Averaging FEFF spectra over MD snapshots builds thermal and static disorder directly into each standard, so σ² is not a fit parameter. N, R, and σ² are all fixed by the simulated structure and dynamics. The only free parameters are one energy shift per standard and the mixing weights.

Experimental and standard files are read as two columns, k (Å⁻¹) and χ(k), from `.dat` (whitespace-delimited) or `.csv` (comma-delimited) files. The experimental spectrum is restricted to $[k_{min}, k_{max}]$. Standards are linearly interpolated onto the experimental k grid.

#### Step 1: an independent ΔE₀ for each standard

Each standard is shifted in energy using the Artemis/IFEFFIT sign convention. A shifted standard is evaluated at

$$k_q^2 = k_{exp}^2 - 0.262468\,\Delta E_0 \qquad (k \text{ in Å}^{-1},\ E \text{ in eV})$$

so a positive ΔE₀ moves the theory edge up in energy. Points where $k_q$ falls outside the standard's k range are discarded. A shift is accepted only if the shifted standard still covers at least `min_valid_frac` (default 95%) of the experimental points.

ΔE₀ is found for each standard *on its own*. A bounded one-dimensional minimization over $[E_{0,min}, E_{0,max}]$ (default ±20 eV) minimizes the chosen metric between $k^n\chi_{exp}$ and $k^n\chi_j$ (default n = 2). By default (`scale_for_shift`), a temporary non-negative amplitude is fitted at each trial ΔE₀:

$$a = \max\!\left(0,\ \frac{\sum_i y_i s_i}{\sum_i s_i^2}\right)$$

Here $y = k^n\chi_{exp}$ and $s = k^n\chi_j(k_q)$. This way the phase of the oscillations, not their amplitude, drives the shift. The amplitude is discarded afterwards.

#### Step 2: weights (and ΔE₀) for every combination

Every combination of 1 to `max_components` standards (default 3) is fitted:

$$k^n\chi_{fit}(k) = \sum_{j \in \text{combo}} w_j\, k^n\chi_j(k_q^{(j)}), \qquad w_j \ge 0,\quad \sum_j w_j = 1$$

`e0_mode` sets how ΔE₀ is treated:

- `joint` (default): the n weights and the n ΔE₀ values (one per standard) are refined together by constrained minimization of the metric (SLSQP), with ΔE₀ bounded by $[E_{0,min}, E_{0,max}]$.
- `joint_shared`: the n weights and one ΔE₀ shared by the whole combination are refined together, like Athena's single E₀ shift.
- `fixed`: each ΔE₀ is kept at its Step 1 value, and only the weights are fitted (SLSQP, starting from equal weights).

A single standard has $w = 1$, so only its ΔE₀ is fitted. The metric is multimodal in ΔE₀ because of the EXAFS oscillations, so the joint fits are started from the Step 1 ΔE₀ values with the `fixed` weights, and from `n_starts` − 1 further points with every ΔE₀ offset by ±`e0_start_step`, ±2·`e0_start_step`, ... eV (defaults 5 and 2.5 eV). The best result is kept. Because the `fixed` solution is one of the starts, `joint` never fits worse than `fixed` over the same k points.

The metric of one joint fit is always evaluated on the same k points: those where every standard of the combination stays inside its k range at both ΔE₀ bounds. If these cover less than `min_valid_frac` of the data, the points valid at the Step 1 values are used, and each ΔE₀ range is narrowed so that none of those points can leave its standard's k range.

1σ uncertainties of the weights and ΔE₀ values are estimated from the Jacobian $J$ of the k-weighted residual, $\text{cov} = \chi^2_\nu (J^TJ)^{-1}$. They are NaN for weights at 0 or 1 and for ΔE₀ values at a bound. Since no measurement uncertainty ε is used and the points are oversampled, treat them as relative, not absolute, error bars.

Because FEFF's χ(k) is normalized per absorbing atom, and the weights sum to 1, each $w_j$ can be read as the fraction of absorbers in environment j.

All combinations are ranked by the chosen metric. With M standards the number of fits is $\sum_{n=1}^{\text{max}} \binom{M}{n}$. For example, 13 standards and up to 3 components give 377 fits.

#### The metrics

No measurement uncertainty ε is available, so all metrics use unweighted residuals $r_i = y_i - y_{fit,i}$ of the k-weighted spectra over the N valid data points:

| Metric | Definition |
|---|---|
| `chi2` | $\sum_i r_i^2$ |
| `redchi` (default) | $\chi^2 / (N - N_{par})$, with $N_{par} = (n-1) + N_{E_0}$ for n standards: n−1 independent weights plus $N_{E_0}$ = n (`joint`), 1 (`joint_shared`) or 0 (`fixed`) ΔE₀ values |
| `rmsd` | $\sqrt{\frac{1}{N}\sum_i r_i^2}$ |
| `rfactor` | $\sum_i r_i^2 / \sum_i y_i^2$, the fractional misfit of §7.3 |

#### How LCF differs from path fitting

| | Path fitting (§7.1–7.5) | MLMD-EXAFS LCF |
|---|---|---|
| Free parameters | S₀², E₀, ΔR, σ², N per path (constrained) | One ΔE₀ per standard, plus mixing weights |
| Disorder (σ²) | Fitted or modeled | Built into each standard by MD averaging |
| E₀ | Usually one shared E₀ | One ΔE₀ per standard refined with the weights (`joint`), one shared ΔE₀ (`joint_shared`), or per-standard values fixed from Step 1 (`fixed`) |
| Overall amplitude | S₀² fitted | None. S₀² is whatever was set in each `feff.inp` (`--s02`), and the weights must sum to 1 |
| χ²_ν denominator | $N_{idp} - N_{var}$ | $N - N_{par}$, with N the number of data points |
| Error bars | From the covariance matrix | From $\chi^2_\nu (J^TJ)^{-1}$, without ε |

#### Interpreting LCF results

- *Compare metrics only within one run.* `redchi` is computed on raw, oversampled data points without ε. Its absolute value cannot be compared with χ²_ν from Artemis/IFEFFIT, and it changes with k range and k-weight. Use it only to rank combinations fitted to the same data, over the same k range, with the same k-weight.
- *The complexity penalty is weak.* Because N (typically a few hundred) is much larger than $N_{par}$, adding a component barely changes the denominator. With N = 210, going from 1 to 3 components changes it from 209 to 205, about 2%. A combination with more standards will therefore usually rank first whenever it lowers χ² at all. Look at how much the metric actually improves. Also check whether the data's information content, $N_{idp} \approx 2\Delta k\Delta R/\pi$ (§7.1), supports that many components. Treat weights close to 0 as absent.
- *Weights assume matched amplitudes.* There is no free scale factor. If S₀², normalization, or the simulated disorder make the standards too strong or too weak overall, the mismatch shows up in the residual, and it can bias which standards are chosen.
- *ΔE₀ values need scrutiny.* In `fixed` mode each ΔE₀ is fitted as if that standard alone explained the data, which for a real mixture is an approximation. `joint` removes that approximation, but n free shifts also let standards absorb misfit by sliding in energy. `joint_shared` is the more constrained choice when all standards describe the same absorber. Physically, standards for the same absorber should need similar shifts, so widely different ΔE₀ values deserve scrutiny. A ΔE₀ at or near the search bound (e.g. −19.99 eV) almost certainly means a false minimum (§7.5), and that standard does not match the data.
- *Uniqueness is not guaranteed.* Standards with similar spectra can trade weight with each other, and fits of similar quality can have very different compositions. Inspect the top-ranked fits in `lcf_results.csv`, not only the best one.

#### Single-standard E₀ alignment (`fit-e0`)

`mlmd-exafs fit-e0` (also run by `mlmd-exafs average --exp-file`) aligns one simulated spectrum to the data, without mixing:

- It uses the same sign convention: the simulated grid is shifted to $k'^2 = k^2 + E_0/3.81$.
- It minimizes the mean squared difference in k²χ over $[k_{min}, k_{max}]$.
- It searches E₀ on a grid over ±10 eV, then refines the best point by golden-section search.

For a single spectrum, it gives essentially the same ΔE₀ as a one-component LCF with `scale_for_shift` off.

## 8. Extending the Fit

LCF in MLMD-EXAFS fits one experimental spectrum, at one k-weight, with a set of standards. The fit is extended by building better standards or by running it more than once, not by adding free parameters.

### 8.1 Multiple environments: building the set of standards

Some samples have the absorber in more than one environment:
- mixed-phase samples
- multiple crystallographic or adsorption sites
- several protonation, oxidation, or substitution states

In path fitting, you run the theory separately for each environment and sum the paths. In MLMD-EXAFS, each environment becomes one LCF standard instead:

1. Build one structure per candidate environment, and simulate and average each one separately.
2. A structure with several inequivalent absorbing sites can be treated either way. If the sites are equally occupied, average all sites into one standard. Otherwise, keep one standard per site and let LCF find the weights.
3. Fit all the standards to the data together.

All standards should be computed with the same FEFF settings. Otherwise differences between standards reflect the settings rather than the structures. See [`feff-input-recs.md`](feff-input-recs.md) §11 for how to set up per-site runs and keep settings consistent.

### 8.2 Multiple data sets

`lcf` fits a single experimental file. There is no simultaneous fit to several data sets with shared weights or ΔE₀ values. What can be done:

- *Temperature series.* Run the MD at each measured temperature and build standards for each. Fit each data set with the standards at its own temperature. MD-averaged standards already include the change in thermal disorder, which path fitting would describe with an Einstein or Debye temperature.
- *Different samples or conditions.* Fit each data set separately with the same standards, k range, and k-weight, then compare the weights. Consistent trends in the weights across the series are more convincing than any single fit.
- *Different absorption edges.* Build separate standards for each absorbing element or edge and fit each edge separately. The weights for the same environment should agree between edges, which is a useful consistency check.

### 8.3 k-weight and k range

- Higher kⁿ emphasizes high-k data, where the spectrum is most sensitive to disorder and to the details of the MD structure.
- Lower kⁿ emphasizes low-k data, where the spectrum is most sensitive to E₀ and to the choice of potentials.
- `lcf --k-weight` (default 2) uses one k-weight for both the ΔE₀ search and the combination fits. Repeat the fit at k-weights 1, 2, and 3, and over slightly different `--kmin`/`--kmax`. A composition that holds up across these is robust; one that changes is not well determined by the data.
- `fit-e0` always compares k²χ.

## 9. Common Pitfalls

General EXAFS analysis:

| Pitfall | Why it matters |
|---|---|
| Too many free parameters | Exceeds $N_{idp}$, giving unreliable, highly correlated results |
| Relying on Fourier-filtered single shells | Shells beyond the first usually overlap in R |
| Unphysical input structures | Produce bad muffin-tin potentials and wrong scattering factors |
| Using a simplified first-shell model for distant shells | Gives unphysical muffin-tin radii and large errors |
| Reading χ²_ν as an absolute measure of fit quality | ε is usually underestimated, so χ²_ν is inflated |
| Ignoring parameter sanity checks | A low R-factor can hide unphysical results |
| Large E₀ shifts | Can indicate a false minimum |
| Using FMS for EXAFS in FEFF | FMS is inaccurate at high k; use the path expansion |

Specific to MLMD-EXAFS:

| Pitfall | Why it matters | What to do |
|---|---|---|
| Adding FEFF's own Debye–Waller model to the snapshots | Disorder is already in the MD average; σ² is counted twice and the amplitude is too small | Leave σ² to the MD (§6.3) |
| Sampling equilibration frames | Early frames are not yet at the target temperature and bias the average | Skip equilibration; check the MD log |
| Too few or too closely spaced snapshots | The average is noisy, or FEFF time is spent on correlated frames | Check convergence of the average (§6.3) |
| Low temperatures or bonds to light atoms | Classical MD misses zero-point motion, so disorder is too small | Treat high-k amplitude with caution; compare with a reference compound |
| Trusting the MLIP geometry without checking | Bond-length errors shift the phase at high k and bias both ΔE₀ and the weights | Compare relaxed bond lengths with known structures |
| Standards computed with different FEFF settings | Differences between standards reflect settings, not structures | Use identical settings for all standards (§8.1) |
| Unrealistic S₀² | LCF has no free amplitude, so an S₀² mismatch biases the weights | Use a physical S₀² (§6.4) |
| Correcting E₀ twice | A shift applied inside FEFF is shifted again by the E₀ fit | Apply the shift in one place only (§6.4) |
| Mixing sign conventions for ΔE₀ | Some codes use the opposite sign | MLMD-EXAFS follows Artemis/IFEFFIT: positive ΔE₀ moves the theory edge up (§7.6) |
| ΔE₀ at the search bound | The minimizer found no real minimum; that standard does not match the data | Drop the standard or check its structure (§7.6) |
| Taking the top-ranked LCF combination at face value | The reduced χ² barely penalizes extra components, and similar standards trade weight | Compare the top-ranked fits, and repeat at other k-weights and k ranges (§8.3) |
| Comparing the reduced χ² across LCF runs | It depends on the k range, k-weight, and number of points | Compare only within one run (§7.6) |

Pitfalls specific to generating `feff.inp` files are listed in [`feff-input-recs.md`](feff-input-recs.md) §12.

## 10. Key References

- J.J. Rehr, J.J. Kas, F.D. Vila, M.P. Prange, K. Jorissen, "Parameter-free calculations of X-ray spectra with FEFF9," *Phys. Chem. Chem. Phys.* **12**, 5503–5513 (2010).
- J.J. Rehr and R.C. Albers, "Theoretical approaches to x-ray absorption fine structure," *Rev. Mod. Phys.* **72**, 621 (2000).
- S.I. Zabinsky *et al.*, "Multiple-scattering calculations of x-ray-absorption spectra," *Phys. Rev. B* **52**, 2995 (1995).
- A. Filipponi, A. Di Cicco, C.R. Natoli, "X-ray absorption spectroscopy and n-body distribution functions in condensed matter. I. Theory," *Phys. Rev. B* **52**, 15122 (1995).
- M. Newville, "IFEFFIT: interactive XAFS analysis and FEFF fitting," *J. Synchrotron Rad.* **8**, 322 (2001).
- M. Newville, "Fundamentals of XAFS," *Rev. Mineral. Geochem.* **78**, 33–74 (2014). Recommended for a derivation of the EXAFS equation.
- B. Ravel, "Quantitative EXAFS Analysis," in *X-Ray Absorption and X-Ray Emission Spectroscopy*, J.A. van Bokhoven and C. Lamberti (eds.), Wiley (2016).
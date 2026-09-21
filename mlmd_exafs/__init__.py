"""MLMD-EXAFS: MLIP molecular dynamics + FEFF EXAFS simulation workflow.

A standalone toolkit for computing theoretical EXAFS chi(k) spectra from
crystal structures:

    cell relaxation -> NVT molecular dynamics (universal MLIP force engine)
    -> FEFF input generation from trajectory snapshots -> batch FEFF
    -> configurational averaging of chi(k) -> k-space / R-space plotting.

The MD force engine can be any of several universal machine-learning
interatomic potentials (CHGNet, MACE, ORB, UMA, SevenNet); FEFF9 computes
the scattering paths for each sampled snapshot.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]

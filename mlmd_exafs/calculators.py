"""MLIP calculator factory.

Builds an ASE-compatible calculator for the requested universal machine-learning
interatomic potential backend. Each backend is imported lazily so that only the
package for the chosen backend needs to be installed.
"""

from __future__ import annotations

from pathlib import Path

BACKENDS = ("chgnet", "mace", "uma", "orb", "sevennet")


def build_calculator(
    backend: str,
    device: str = "cpu",
    model: str | None = None,
    head: str = "omat",
    modal: str = "mpa",
    checkpoint: str | None = None,
):
    """Return an ASE calculator for the named MLIP backend.

    Parameters
    ----------
    backend : str
        One of ``chgnet``, ``mace``, ``uma``, ``orb``, ``sevennet``.
    device : str
        ``cpu`` or ``cuda``. Ignored by CHGNet's default loader path (it picks
        a device automatically) but honored by the others.
    model : str, optional
        Model name/checkpoint for the backend. Falls back to a sensible default
        per backend when omitted.
    head : str
        UMA task head (``oc20``, ``omat``, ``omol``, ``odac``, ``omc``).
    modal : str
        Dataset modality for ORB / SevenNet (``mpa`` or ``omat24``).
    checkpoint : str, optional
        Path to a local model checkpoint file, e.g. a fine-tuned model. Loaded
        in place of the pretrained model; ``model`` must then be omitted. The
        file must be in the backend's own checkpoint format:

        - ``chgnet``: CHGNet trainer checkpoint (``*.pth.tar``), loaded with
          ``CHGNet.from_file``.
        - ``mace``: MACE model file (``*.model``).
        - ``uma``: fairchem checkpoint (``*.pt``), loaded with
          ``load_predict_unit``.
        - ``orb``: ORB state dict (``*.ckpt``) for the ``orb-v3-conservative-inf``
          architecture, e.g. from orb-models' ``finetune.py``.
        - ``sevennet``: SevenNet checkpoint (``*.pth``), e.g.
          ``checkpoint_best.pth``.

    Returns
    -------
    ase.calculators.calculator.Calculator
    """
    backend = backend.lower()

    if checkpoint is not None:
        if model is not None:
            raise ValueError("Pass either model or checkpoint, not both.")
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint!r}.")
        checkpoint = str(Path(checkpoint).resolve())

    if backend == "chgnet":
        from chgnet.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator

        chgnet = CHGNet.from_file(checkpoint) if checkpoint else CHGNet.load()
        return CHGNetCalculator(chgnet, use_device=device)

    if backend == "mace":
        from mace.calculators import mace_mp

        return mace_mp(
            model=checkpoint or model or "medium-omat-0",
            device=device,
            default_dtype="float32",
        )

    if backend == "uma":
        from fairchem.core import FAIRChemCalculator, pretrained_mlip

        if checkpoint:
            predictor = pretrained_mlip.load_predict_unit(checkpoint, device=device)
        else:
            predictor = pretrained_mlip.get_predict_unit(
                model or "uma-m-1p1", device=device
            )
        return FAIRChemCalculator(predictor, task_name=head)

    if backend == "orb":
        import torch._dynamo

        torch._dynamo.config.suppress_errors = True
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        weights = {"weights_path": checkpoint} if checkpoint else {}
        orb_model = pretrained.orb_v3_conservative_inf_omat(
            device=device, precision="float32-highest", **weights
        )
        return ORBCalculator(orb_model, device=device)

    if backend == "sevennet":
        from sevenn.calculator import SevenNetCalculator

        return SevenNetCalculator(
            model=checkpoint or model or "7net-mf-ompa", modal=modal
        )

    raise ValueError(
        f"Unknown backend {backend!r}. Choose one of: {', '.join(BACKENDS)}."
    )

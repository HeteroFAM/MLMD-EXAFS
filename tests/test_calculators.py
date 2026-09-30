import sys
import types

import pytest

from mlmd_exafs.calculators import BACKENDS, build_calculator


def test_backends_tuple():
    assert BACKENDS == ("chgnet", "mace", "uma", "orb", "sevennet")


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        build_calculator("not-a-backend")


def test_model_and_checkpoint_are_exclusive(tmp_path):
    ckpt = tmp_path / "m.model"
    ckpt.write_text("x")
    with pytest.raises(ValueError, match="either model or checkpoint"):
        build_calculator("mace", model="small", checkpoint=str(ckpt))


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_calculator("mace", checkpoint=str(tmp_path / "missing.model"))


@pytest.fixture
def fake_mace(monkeypatch):
    calls = []

    def mace_mp(**kwargs):
        calls.append(kwargs)
        return "mace-calc"

    calculators = types.ModuleType("mace.calculators")
    calculators.mace_mp = mace_mp
    monkeypatch.setitem(sys.modules, "mace", types.ModuleType("mace"))
    monkeypatch.setitem(sys.modules, "mace.calculators", calculators)
    return calls


def test_mace_default_model(fake_mace):
    assert build_calculator("MACE") == "mace-calc"
    assert fake_mace[0]["model"] == "medium-omat-0"
    assert fake_mace[0]["device"] == "cpu"


def test_mace_checkpoint_passed_as_absolute_path(fake_mace, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ft.model").write_text("x")
    build_calculator("mace", checkpoint="ft.model")
    assert fake_mace[0]["model"] == str((tmp_path / "ft.model").resolve())


def test_chgnet_checkpoint_uses_from_file(monkeypatch, tmp_path):
    calls = {}

    class FakeCHGNet:
        @staticmethod
        def from_file(path):
            calls["from_file"] = path
            return "model"

        @staticmethod
        def load():
            calls["load"] = True
            return "pretrained"

    def FakeCalc(model, use_device):
        return (model, use_device)

    model_mod = types.ModuleType("chgnet.model")
    model_mod.CHGNet = FakeCHGNet
    dyn_mod = types.ModuleType("chgnet.model.dynamics")
    dyn_mod.CHGNetCalculator = FakeCalc
    monkeypatch.setitem(sys.modules, "chgnet", types.ModuleType("chgnet"))
    monkeypatch.setitem(sys.modules, "chgnet.model", model_mod)
    monkeypatch.setitem(sys.modules, "chgnet.model.dynamics", dyn_mod)

    assert build_calculator("chgnet") == ("pretrained", "cpu")
    ckpt = tmp_path / "c.pth.tar"
    ckpt.write_text("x")
    assert build_calculator("chgnet", checkpoint=str(ckpt), device="cuda") == ("model", "cuda")
    assert calls["from_file"] == str(ckpt.resolve())

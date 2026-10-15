import json
import shutil
from pathlib import Path

import pytest
import yaml

from reel_predictor import paths
from reel_predictor.config import INDEX_NAMES, ConfigError, load_config


def _copy_config(tmp_path: Path) -> Path:
    d = tmp_path / "config"
    shutil.copytree(paths.CONFIG_DIR, d)
    return d


def test_load_default_config():
    cfg = load_config(calibration_file=None)
    assert set(cfg.raw) == {"weights", "roi_mapping", "rules"}
    assert cfg.brain.provider in ("tribe", "heuristic", "mock")
    assert cfg.brain.model == "facebook/tribev2-mini"
    assert cfg.speech.model == "small"
    for name in INDEX_NAMES:
        assert 0 <= cfg.section("indices")[name]["brain_weight"] <= 1
        assert name in cfg.roi_mapping["indices"]
    names = [r["name"] for r in cfg.rules]
    for required in ("attention_drop_no_cut", "weak_hook", "no_face_long", "silence_gap", "text_overload",
                     "cta_missing", "tension_flat", "slow_start"):
        assert required in names
    assert cfg.section("normalization")["hrf_shift_s"] == 0


def test_invalid_weights_raise(tmp_path):
    d = _copy_config(tmp_path)
    w = yaml.safe_load((d / "weights.yaml").read_text())
    w["indices"]["attention"]["brain_weight"] = 2.0
    w["indices"]["attention"]["boosters"]["feature_inexistante"] = 0.1
    del w["retention"]["base_drop"]
    (d / "weights.yaml").write_text(yaml.safe_dump(w, allow_unicode=True))
    with pytest.raises(ConfigError) as e:
        load_config(d, calibration_file=None)
    msg = str(e.value)
    assert "brain_weight" in msg and "feature_inexistante" in msg and "base_drop" in msg


def test_invalid_rules_raise(tmp_path):
    d = _copy_config(tmp_path)
    r = yaml.safe_load((d / "rules.yaml").read_text())
    r["rules"][0]["severity"] = "critique"
    (d / "rules.yaml").write_text(yaml.safe_dump(r, allow_unicode=True))
    with pytest.raises(ConfigError):
        load_config(d, calibration_file=None)


def test_unknown_roi_in_mapping_raises(tmp_path):
    d = _copy_config(tmp_path)
    m = yaml.safe_load((d / "roi_mapping.yaml").read_text())
    m["indices"]["attention"]["roi_inexistante"] = 1.0
    (d / "roi_mapping.yaml").write_text(yaml.safe_dump(m, allow_unicode=True))
    with pytest.raises(ConfigError):
        load_config(d, calibration_file=None)


def test_calibration_override(tmp_path):
    cal = tmp_path / "calibration.json"
    cal.write_text(json.dumps({"params": {"base_drop": 0.031, "hook_drop": 0.12, "inconnu": 3}}))
    cfg = load_config(calibration_file=cal)
    assert cfg.retention.base_drop == pytest.approx(0.031)
    assert cfg.retention.hook_drop == pytest.approx(0.12)
    assert cfg.weights["retention"]["base_drop"] == pytest.approx(0.031)
    # les YAML bruts ne sont pas modifiés
    assert cfg.raw["weights"]["retention"]["base_drop"] != pytest.approx(0.031)
    assert cfg.calibration is not None


def test_with_brain_provider():
    cfg = load_config(calibration_file=None)
    m = cfg.with_brain_provider("mock")
    assert m.brain.provider == "mock"
    assert cfg.brain.provider != "mock" or cfg.raw["weights"]["brain"]["provider"] == "mock"
    with pytest.raises(ConfigError):
        cfg.with_brain_provider("inconnu")

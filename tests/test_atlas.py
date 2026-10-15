import numpy as np
import pytest

from reel_predictor import paths
from reel_predictor.brain.atlas import load_atlas, reduce, resolve_labels, roi_masks, roi_matrix
from reel_predictor.config import load_config


@pytest.fixture(scope="module")
def roi_mapping():
    return load_config(calibration_file=None).roi_mapping


def test_atlas_shape():
    a = load_atlas()
    assert a.map_left.size + a.map_right.size == paths.N_VERTICES == 20484
    assert a.vertex_labels.shape == (20484,)


def test_all_labels_resolved(roi_mapping):
    ids = resolve_labels(roi_mapping)
    assert set(ids) == set(roi_mapping["rois"])
    for roi, labels in roi_mapping["rois"].items():
        assert len(ids[roi]) == len(labels)


def test_masks_and_matrix(roi_mapping):
    masks = roi_masks(roi_mapping)
    for roi, m in masks.items():
        assert m.shape == (20484,) and m.sum() > 0, roi
        if roi_mapping.get("hemispheres", "both") == "both":
            assert m[: paths.HEMI_SPLIT].any() and m[paths.HEMI_SPLIT:].any(), roi
    names, W = roi_matrix(roi_mapping)
    assert W.shape == (len(names), 20484)
    np.testing.assert_allclose(W.sum(axis=1), 1.0, rtol=1e-5)
    # une activité constante donne la même valeur dans chaque ROI
    preds = np.full((3, 20484), 2.5, np.float32)
    np.testing.assert_allclose(reduce(preds, W), 2.5, rtol=1e-5)


def test_unknown_label_raises(roi_mapping):
    bad = {**roi_mapping, "rois": {**roi_mapping["rois"], "x": ["Label_inexistant"]}}
    with pytest.raises(ValueError):
        resolve_labels(bad)

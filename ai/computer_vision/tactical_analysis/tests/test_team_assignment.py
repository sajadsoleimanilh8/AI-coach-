"""
Unit tests for jersey-color team assignment.
"""

import os
import sys

import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from ai.computer_vision.tactical_analysis.team_assignment import (
    _cluster_team_labels,
    _kmeans2,
    _margins,
    assign_teams,
    crop_to_feature,
)
from ai.computer_vision.player_tracking.tracker import TrackedDetection


def _solid_crop(bgr, h=40, w=20):
    crop = np.zeros((h, w, 3), dtype=np.uint8)
    crop[:, :] = bgr
    return crop


def test_crop_to_feature_masks_grass_and_skin():
    grass = _solid_crop((0, 180, 0))
    assert crop_to_feature(grass) is None

    mostly_grass = _solid_crop((0, 180, 0), h=40, w=20)
    mostly_grass[0:2, 0:2] = (0, 0, 200)
    assert crop_to_feature(mostly_grass) is None

    red = _solid_crop((0, 0, 200))
    feat = crop_to_feature(red)
    assert feat is not None
    assert feat.shape == (3,)


def test_crop_to_feature_hue_circularity():
    red_bgr = _solid_crop((0, 0, 255))
    feat = crop_to_feature(red_bgr)
    assert feat is not None
    cos_h, sin_h, _ = feat
    assert cos_h > 0.9, f"expected cos_h near 1.0 for a red crop, got {cos_h}"


def test_kmeans2_separates_and_is_deterministic():
    red = crop_to_feature(_solid_crop((0, 0, 200)))
    blue = crop_to_feature(_solid_crop((200, 0, 0)))
    features = np.stack([red] * 8 + [blue] * 8)

    labels1, centroids1 = _kmeans2(features)
    labels2, centroids2 = _kmeans2(features)

    assert np.array_equal(labels1, labels2), "same input must produce same labels (deterministic init)"
    assert len(set(labels1[:8].tolist())) == 1
    assert len(set(labels1[8:].tolist())) == 1
    assert labels1[0] != labels1[8]


def test_margins_bounds():
    red = crop_to_feature(_solid_crop((0, 0, 200)))
    blue = crop_to_feature(_solid_crop((200, 0, 0)))
    features = np.stack([red] * 5 + [blue] * 5)
    labels, centroids = _kmeans2(features)
    margins = _margins(features, centroids, labels)
    assert np.all(margins >= 0.0) and np.all(margins <= 1.0)
    assert np.all(margins > 0.9)


def test_cluster_team_labels_deterministic_across_centroid_order():
    red = crop_to_feature(_solid_crop((0, 0, 200)))
    blue = crop_to_feature(_solid_crop((200, 0, 0)))

    labels_ab = _cluster_team_labels(np.stack([red, blue]))
    labels_ba = _cluster_team_labels(np.stack([blue, red]))
    assert labels_ab[0] != labels_ab[1]
    assert {labels_ab[0], labels_ab[1]} == {"team-home", "team-away"}
    red_label_when_first = labels_ab[0]
    red_label_when_second = labels_ba[1]
    assert red_label_when_first == red_label_when_second


def test_assign_teams_too_few_crops_leaves_everyone_unassigned():
    frames = [
        [
            TrackedDetection(
                frame_number=0, player_id=1, class_name="player", team_id=None,
                team_assignment_confidence=None, x=0, y=0, width=10, height=20, confidence=0.9,
            ),
            TrackedDetection(
                frame_number=0, player_id=2, class_name="goalkeeper", team_id=None,
                team_assignment_confidence=None, x=50, y=0, width=10, height=20, confidence=0.9,
            ),
        ]
    ]
    confidence = assign_teams("D:/this/path/does/not/exist.mp4", frames)
    assert confidence == 0.0
    assert frames[0][0].team_id is None
    assert frames[0][0].team_assignment_confidence is None
    assert frames[0][1].team_id is None
    assert frames[0][1].team_assignment_confidence is None


if __name__ == "__main__":
    test_crop_to_feature_masks_grass_and_skin()
    test_crop_to_feature_hue_circularity()
    test_kmeans2_separates_and_is_deterministic()
    test_margins_bounds()
    test_cluster_team_labels_deterministic_across_centroid_order()
    test_assign_teams_too_few_crops_leaves_everyone_unassigned()
    print("ALL TEAM ASSIGNMENT TESTS PASSED!")

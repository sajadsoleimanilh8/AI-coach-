"""
Unit tests for jersey-color team assignment.
"""


import numpy as np

from ai.computer_vision.player_tracking.tracker import TrackedDetection
from ai.computer_vision.tactical_analysis.team_assignment import (
    _cluster_team_labels,
    _kmeans2,
    _margins,
    assign_teams,
    crop_to_feature,
)


def _solid_crop(bgr, h=40, w=20):
    crop = np.zeros((h, w, 3), dtype=np.uint8)
    crop[:, :] = bgr
    return crop


def test_crop_to_feature_masks_grass_and_skin():
    # Pure grass green (BGR) should be masked out entirely -> unusable crop.
    grass = _solid_crop((0, 180, 0))
    assert crop_to_feature(grass) is None

    # A crop that's mostly grass with a small colored patch too small to
    # clear TEAM_ASSIGNMENT_MIN_USABLE_PIXELS should also come back None.
    mostly_grass = _solid_crop((0, 180, 0), h=40, w=20)
    mostly_grass[0:2, 0:2] = (0, 0, 200)  # a few red pixels, not enough
    assert crop_to_feature(mostly_grass) is None

    # A solid, saturated red crop should produce a usable feature.
    red = _solid_crop((0, 0, 200))
    feat = crop_to_feature(red)
    assert feat is not None
    assert feat.shape == (3,)


def test_crop_to_feature_hue_circularity():
    # Two reds that straddle the OpenCV hue wraparound (H near 0 and H
    # near 179) should both produce a feature with cos_h close to +1 --
    # not a naive-average artifact landing near hue=90 (green/cyan).
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
    # Well-separated, identical-within-cluster points should sit almost
    # exactly on their own centroid.
    assert np.all(margins > 0.9)


def test_cluster_team_labels_deterministic_across_centroid_order():
    red = crop_to_feature(_solid_crop((0, 0, 200)))
    blue = crop_to_feature(_solid_crop((200, 0, 0)))

    labels_ab = _cluster_team_labels(np.stack([red, blue]))
    labels_ba = _cluster_team_labels(np.stack([blue, red]))
    # Whichever index red/blue end up at, the SAME physical colors must
    # resolve to the SAME team-home/team-away labels.
    assert labels_ab[0] != labels_ab[1]
    assert {labels_ab[0], labels_ab[1]} == {"team-home", "team-away"}
    # red's label should be identical regardless of which centroid slot it lands in.
    red_label_when_first = labels_ab[0]
    red_label_when_second = labels_ba[1]
    assert red_label_when_first == red_label_when_second


def test_assign_teams_too_few_crops_leaves_everyone_unassigned():
    # No real video on disk -- _iter_sampled_crops degrades gracefully to
    # yielding nothing, so this exercises the "not enough usable crops"
    # branch without needing a video fixture.
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

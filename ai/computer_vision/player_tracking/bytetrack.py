from __future__ import annotations
from enum import Enum
from typing import List, Tuple
import numpy as np
import scipy.linalg
from scipy.optimize import linear_sum_assignment

class TrackState(Enum):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

class KalmanFilter:
    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(std))
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        innov_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T)) + innov_cov
        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

class STrack:
    shared_kalman = KalmanFilter()
    _count = 0

    def __init__(self, tlwh: np.ndarray, score: float, class_id: int):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        
        self.score = score
        self.class_id = class_id
        self.tracklet_len = 0
        self.state = TrackState.New

        self.track_id = 0
        
    @staticmethod
    def next_id() -> int:
        STrack._count += 1
        return STrack._count

    @staticmethod
    def reset_id_counter() -> None:
        """Call this before tracking a new match/video in a long-running
        process (e.g. a Celery worker handling multiple uploads), otherwise
        player_id keeps climbing across unrelated matches instead of
        starting fresh each time -- _count is a class-level counter, not
        """
        STrack._count = 0

    @property
    def tlbr(self) -> np.ndarray:
        ret = self._tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def activate(self, kalman_filter: KalmanFilter, frame_id: int):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.frame_id = frame_id
        self.is_activated = (frame_id == 1)

    def predict(self):
        if self.state != TrackState.Tracked:
            self.mean[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)

    def update(self, new_track: STrack, frame_id: int):
        self.frame_id = frame_id
        self.tracklet_len += 1
        new_tlwh = new_track._tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self._tlwh = self.xyah_to_tlwh(self.mean[:4])

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def xyah_to_tlwh(xyah: np.ndarray) -> np.ndarray:
        ret = np.asarray(xyah).copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

def bbox_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))

    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    inter_x1 = np.maximum(b1_x1[:, None], b2_x1)
    inter_y1 = np.maximum(b1_y1[:, None], b2_y1)
    inter_x2 = np.minimum(b1_x2[:, None], b2_x2)
    inter_y2 = np.minimum(b1_y2[:, None], b2_y2)

    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    union_area = b1_area[:, None] + b2_area - inter_area
    return inter_area / np.maximum(union_area, 1e-6)

def linear_assignment(cost_matrix: np.ndarray, thresh: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), np.arange(cost_matrix.shape[0]), np.arange(cost_matrix.shape[1])
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matches, unmatched_a, unmatched_b = [], [], []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > thresh:
            unmatched_a.append(r)
            unmatched_b.append(c)
        else:
            matches.append([r, c])

    matches = np.array(matches) if len(matches) > 0 else np.empty((0, 2), dtype=int)
    
    all_rows, all_cols = np.arange(cost_matrix.shape[0]), np.arange(cost_matrix.shape[1])
    unmatched_a.extend(list(set(all_rows) - set(row_ind)))
    unmatched_b.extend(list(set(all_cols) - set(col_ind)))
    
    return matches, np.array(unmatched_a, dtype=int), np.array(unmatched_b, dtype=int)

class BYTETracker:
    def __init__(self, track_thresh: float = 0.5, track_buffer: int = 30, match_thresh: float = 0.8):
        self.track_thresh = track_thresh
        self.high_thresh = track_thresh
        self.low_thresh = 0.1
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        
        self.frame_id = 0
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        self.kalman_filter = KalmanFilter()

    def reset(self) -> None:
        """Start fresh for a new match/video: clears this tracker's own
        state AND the shared STrack ID counter. Safe to call between
        videos in a long-running worker process."""
        self.frame_id = 0
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        STrack.reset_id_counter()

    def update(self, output_results: np.ndarray) -> List[STrack]:
        self.frame_id += 1
        activated_stracks, refind_stracks, lost_stracks, removed_stracks = [], [], [], []

        if len(output_results) == 0:
            for strack in self.tracked_stracks:
                strack.mark_lost()
                self.lost_stracks.append(strack)
            self.tracked_stracks = []
            return []

        scores = output_results[:, 4]
        high_mask = scores >= self.high_thresh
        low_mask = (scores > self.low_thresh) & (scores < self.high_thresh)
        
        dets_high = output_results[high_mask]
        dets_low = output_results[low_mask]

        def _to_stracks(dets):
            return [STrack([d[0], d[1], d[2]-d[0], d[3]-d[1]], d[4], int(d[5])) for d in dets]
        
        detections = _to_stracks(dets_high)
        detections_second = _to_stracks(dets_low)

        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
                
        strack_pool = self._join_tracks(tracked_stracks, self.lost_stracks)
        for strack in strack_pool:
            strack.predict()

        dists = self._iou_distance(strack_pool, detections)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.update(det, self.frame_id)
                refind_stracks.append(track)

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = self._iou_distance(r_tracked_stracks, detections_second)
        matches, u_track_second, u_detection_second = linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            track.update(det, self.frame_id)
            activated_stracks.append(track)

        for it in u_track_second:
            track = r_tracked_stracks[it]
            track.mark_lost()
            lost_stracks.append(track)

        detections_unmatched = [detections[i] for i in u_detection]
        dists = self._iou_distance(unconfirmed, detections_unmatched)
        matches, u_unconfirmed, u_detection_new = linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections_unmatched[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        for inew in u_detection_new:
            track = detections_unmatched[inew]
            if track.score < self.high_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.track_buffer:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self._join_tracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self._join_tracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self._sub_tracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self._sub_tracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)

        return [t for t in self.tracked_stracks if t.is_activated]

    def _join_tracks(self, tlista: List[STrack], tlistb: List[STrack]) -> List[STrack]:
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            if t.track_id not in exists:
                exists[t.track_id] = 1
                res.append(t)
        return res

    def _sub_tracks(self, tlista: List[STrack], tlistb: List[STrack]) -> List[STrack]:
        stracks = {t.track_id: t for t in tlista}
        for t in tlistb:
            if t.track_id in stracks:
                del stracks[t.track_id]
        return list(stracks.values())

    def _iou_distance(self, tracks: List[STrack], detections: List[STrack]) -> np.ndarray:
        if len(tracks) == 0 or len(detections) == 0:
            return np.zeros((len(tracks), len(detections)))
        track_boxes = np.array([track.tlbr for track in tracks])
        det_boxes = np.array([det.tlbr for det in detections])
        ious = bbox_iou(track_boxes, det_boxes)
        return 1.0 - ious
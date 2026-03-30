import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman import KalmanFilter
from .matching import iou
from ReID.extractor import ReIDExtractor


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-6)


# Track Class
class Track:
    def __init__(self, bbox, track_id, feature):
        self.kf = KalmanFilter(bbox)
        self.id = track_id

        self.feature = feature
        self.alpha = 0.9

        self.time_since_update = 0
        self.hits = 1
        self.confirmed = False

        self.state = "active"  # active / lost

    def update(self, bbox, feature):
        self.kf.update(bbox)

        # EMA feature
        self.feature = self.alpha * self.feature + (1 - self.alpha) * feature
        self.feature = self.feature / (np.linalg.norm(self.feature) + 1e-6)

        self.time_since_update = 0
        self.hits += 1
        self.state = "active"

        if self.hits >= 3:
            self.confirmed = True

    def predict(self):
        self.time_since_update += 1

        if self.time_since_update > 5:
            self.state = "lost"

        return self.kf.predict()


# Tracker Class
class Tracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 0
        self.extractor = ReIDExtractor()

        self.max_age = 30
        self.match_threshold = 0.5

    # Recover lost track
    def _recover_lost_track(self, feature):
        best_track = None
        best_score = 0.7

        for t in self.tracks:
            if t.state != "lost":
                continue

            score = cosine(t.feature, feature)

            if score > best_score:
                best_score = score
                best_track = t

        return best_track

    # Update
    def update(self, detections, frame):

        # 1. Predict
        for t in self.tracks:
            t.predict()

        # 2. Extract features
        det_features = [
            self.extractor.extract(frame, d)
            for d in detections
        ]

        # 3. If no tracks yet
        if len(self.tracks) == 0:
            for i, det in enumerate(detections):
                self.tracks.append(
                    Track(det, self.next_id, det_features[i])
                )
                self.next_id += 1

            return [(detections[i], i) for i in range(len(detections))]

        # 4. Cost matrix
        cost_matrix = np.zeros((len(self.tracks), len(detections)))

        for i, t in enumerate(self.tracks):
            for j, d in enumerate(detections):

                iou_score = iou(t.kf.state, d)
                app_score = cosine(t.feature, det_features[j])

                score = 0.2 * iou_score + 0.8 * app_score

                cost_matrix[i, j] = 1 - score

        # 5. Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_dets = set()
        results = []

        # 6. Apply matches
        for r, c in zip(row_ind, col_ind):

            score = 1 - cost_matrix[r, c]

            if score < self.match_threshold:
                continue

            self.tracks[r].update(detections[c], det_features[c])
            assigned_dets.add(c)

            if self.tracks[r].confirmed:
                results.append((detections[c], self.tracks[r].id))

        # 7. Unmatched detections
        for j, det in enumerate(detections):

            if j in assigned_dets:
                continue

            feat = det_features[j]

            recovered = self._recover_lost_track(feat)

            if recovered is not None:
                recovered.update(det, feat)
                recovered.state = "active"

                if recovered.confirmed:
                    results.append((det, recovered.id))

                continue

            self.tracks.append(
                Track(det, self.next_id, feat)
            )
            self.next_id += 1

        # 8. Remove dead tracks
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update < self.max_age
        ]

        return results
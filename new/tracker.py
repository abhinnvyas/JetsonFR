import math
import cv2
from collections import OrderedDict


class Track:
    def __init__(self, track_id, bbox):
        self.id = track_id
        self.bbox = bbox
        self.disappeared = 0
        self.name = None
        self.recognized = False
        self.unknown_count = 0

    @property
    def centroid(self):
        x1, y1, x2, y2 = self.bbox
        return (
            (x1 + x2) // 2,
            (y1 + y2) // 2,
        )


class CentroidTracker:

    def __init__(
        self,
        max_disappeared=15,
        max_distance=100,
    ):
        self.next_track_id = 0
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.tracks = OrderedDict()

    def register(self, bbox, name=None):
        track = Track(self.next_track_id, bbox)
        track.name = name
        track.unknown_count = 0
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1

    def deregister(self, track_id):
        if track_id in self.tracks:
            del self.tracks[track_id]

    @staticmethod
    def _centroid(bbox):
        x1, y1, x2, y2 = bbox
        return (
            (x1 + x2) // 2,
            (y1 + y2) // 2,
        )

    @staticmethod
    def _distance(c1, c2):
        return math.sqrt(
            (c1[0] - c2[0]) ** 2 +
            (c1[1] - c2[1]) ** 2
        )

    def update(self, detections):
        """
        detections = list of (bbox, name) tuples
        [
            ([x1,y1,x2,y2], "Name"),
            ...
        ]
        """

        if len(detections) == 0:

            remove = []

            for track_id, track in self.tracks.items():
                track.disappeared += 1

                if track.disappeared > self.max_disappeared:
                    remove.append(track_id)

            for track_id in remove:
                self.deregister(track_id)

            return self.tracks

        if len(self.tracks) == 0:

            for bbox, name in detections:
                self.register(bbox, name)

            return self.tracks

        detection_centroids = [
            self._centroid(bbox)
            for bbox, name in detections
        ]

        used_tracks = set()
        used_detections = set()

        track_ids = list(self.tracks.keys())

        for track_id in track_ids:

            track = self.tracks[track_id]

            best_idx = -1
            best_distance = 1e9

            for idx, centroid in enumerate(detection_centroids):

                if idx in used_detections:
                    continue

                dist = self._distance(
                    track.centroid,
                    centroid
                )

                if dist < best_distance:
                    best_distance = dist
                    best_idx = idx

            if (
                best_idx != -1
                and best_distance < self.max_distance
            ):

                bbox, name = detections[best_idx]
                track.bbox = bbox
                track.disappeared = 0

                # Name update logic with leak protection & flicker prevention
                if name != "Unknown":
                    track.name = name
                    track.unknown_count = 0
                else:
                    if track.name is not None and track.name != "Unknown":
                        track.unknown_count += 1
                        if track.unknown_count > 5:
                            track.name = "Unknown"
                    else:
                        track.name = "Unknown"

                used_tracks.add(track_id)
                used_detections.add(best_idx)

            else:

                track.disappeared += 1

        remove = []

        for track_id, track in self.tracks.items():

            if track.disappeared > self.max_disappeared:
                remove.append(track_id)

        for track_id in remove:
            self.deregister(track_id)

        for idx, (bbox, name) in enumerate(detections):

            if idx not in used_detections:
                self.register(bbox, name)

        return self.tracks


class TemplateTracker:
    def __init__(self, frame, bbox, padding=40):
        """
        bbox is [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        
        # Crop template with boundary protection
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        
        self.template = frame[y1:y2, x1:x2].copy()
        self.bbox = [x1, y1, x2, y2]
        self.padding = padding

    def update(self, frame):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, self.bbox)
        tw = x2 - x1
        th = y2 - y1
        
        if tw <= 0 or th <= 0 or self.template.size == 0:
            return False, self.bbox

        # Define search region around the last known position
        sx1 = max(0, x1 - self.padding)
        sy1 = max(0, y1 - self.padding)
        sx2 = min(w, x2 + self.padding)
        sy2 = min(h, y2 + self.padding)
        
        search_region = frame[sy1:sy2, sx1:sx2]
        
        # Search region must be larger than or equal to template size
        if search_region.shape[0] < th or search_region.shape[1] < tw:
            return False, self.bbox
            
        # Run template matching
        res = cv2.matchTemplate(search_region, self.template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if max_val > 0.55:  # Similarity threshold
            # Update bbox position
            new_x1 = sx1 + max_loc[0]
            new_y1 = sy1 + max_loc[1]
            self.bbox = [new_x1, new_y1, new_x1 + tw, new_y1 + th]
            return True, self.bbox
        else:
            return False, self.bbox

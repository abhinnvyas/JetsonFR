import math
from collections import OrderedDict


class Track:
    def __init__(self, track_id, bbox):
        self.id = track_id
        self.bbox = bbox
        self.disappeared = 0
        self.name = None
        self.recognized = False

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
                # Update name if a valid identity was recognized, or if it is currently None
                if name != "Unknown" or track.name is None:
                    track.name = name
                track.disappeared = 0

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

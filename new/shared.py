import threading
import time
import copy


class SharedState:
    """
    Thread-safe shared data between multi-camera pipeline threads.
    """

    def __init__(self):

        self.frame_lock = threading.Lock()
        self.track_lock = threading.Lock()

        # Multi-camera state dictionaries (camera_id -> data)
        self.frames = {}
        self.tracks = {}
        self.track_versions = {}
        self.frame_ids = {}

        self.running = True

        # FPS trackers
        self.capture_fps = {}
        self.last_capture_times = {}
        self.inference_fps = 0.0
        self.display_fps = 0.0

        self.last_inference_time = time.time()
        self.last_display_time = time.time()

    # ---------------------------------------------------
    # Frame functions
    # ---------------------------------------------------

    def set_frame(self, camera_id, frame):
        """
        Called by the capture thread of each camera.
        """
        with self.frame_lock:
            self.frames[camera_id] = frame
            self.frame_ids[camera_id] = self.frame_ids.get(camera_id, 0) + 1

    def get_frame(self, camera_id):
        """
        Called by inference and display threads.
        """
        with self.frame_lock:
            frame = self.frames.get(camera_id)
            if frame is None:
                return None
            return frame.copy()

    # ---------------------------------------------------
    # Tracking results
    # ---------------------------------------------------

    def set_tracks(self, camera_id, tracks):
        """
        Store latest tracking results for a specific camera.
        """
        with self.track_lock:
            self.tracks[camera_id] = copy.deepcopy(tracks)
            self.track_versions[camera_id] = self.track_versions.get(camera_id, 0) + 1

    def get_tracks(self, camera_id):
        """
        Returns a safe copy of tracks for a specific camera.
        """
        with self.track_lock:
            return copy.deepcopy(self.tracks.get(camera_id, []))

    def get_tracks_with_version(self, camera_id):
        """
        Returns a safe copy of tracks along with the version for a specific camera.
        """
        with self.track_lock:
            tracks = copy.deepcopy(self.tracks.get(camera_id, []))
            version = self.track_versions.get(camera_id, 0)
            return tracks, version

    # ---------------------------------------------------
    # FPS counters
    # ---------------------------------------------------

    def update_capture_fps(self, camera_id):
        now = time.time()
        last_time = self.last_capture_times.get(camera_id, now)
        dt = now - last_time
        if dt > 0:
            self.capture_fps[camera_id] = 1.0 / dt
        self.last_capture_times[camera_id] = now

    def update_inference_fps(self):
        now = time.time()
        dt = now - self.last_inference_time
        if dt > 0:
            self.inference_fps = 1.0 / dt
        self.last_inference_time = now

    def update_display_fps(self):
        now = time.time()
        dt = now - self.last_display_time
        if dt > 0:
            self.display_fps = 1.0 / dt
        self.last_display_time = now

    # ---------------------------------------------------
    # Shutdown
    # ---------------------------------------------------

    def stop(self):
        self.running = False

    def is_running(self):
        return self.running

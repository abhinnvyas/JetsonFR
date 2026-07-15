import threading
import time
import copy


class SharedState:
    """
    Thread-safe shared data between:

    Capture Thread
            ↓
      latest_frame

    Inference Thread
            ↓
      latest_tracks

    Display Thread
            ↓
       frame + tracks
    """

    def __init__(self):

        self.frame_lock = threading.Lock()
        self.track_lock = threading.Lock()

        self.latest_frame = None
        self.latest_tracks = []

        self.frame_id = 0
        self.running = True

        self.capture_fps = 0.0
        self.inference_fps = 0.0
        self.display_fps = 0.0

        self.last_capture_time = time.time()
        self.last_inference_time = time.time()
        self.last_display_time = time.time()

    # ---------------------------------------------------
    # Frame functions
    # ---------------------------------------------------

    def set_frame(self, frame):
        """
        Called only by the capture thread.

        Always overwrite the previous frame so inference
        never builds up latency.
        """

        with self.frame_lock:
            self.latest_frame = frame
            self.frame_id += 1

    def get_frame(self):
        """
        Called by inference and display threads.

        Returns a COPY to avoid concurrent modification.
        """

        with self.frame_lock:

            if self.latest_frame is None:
                return None

            return self.latest_frame.copy()

    # ---------------------------------------------------
    # Tracking results
    # ---------------------------------------------------

    def set_tracks(self, tracks):
        """
        Store latest tracking results.

        tracks is expected to be:
        list(track_objects)
        """

        with self.track_lock:
            self.latest_tracks = copy.deepcopy(tracks)

    def get_tracks(self):
        """
        Returns a safe copy of tracks.
        """

        with self.track_lock:
            return copy.deepcopy(self.latest_tracks)

    # ---------------------------------------------------
    # FPS counters
    # ---------------------------------------------------

    def update_capture_fps(self):

        now = time.time()

        dt = now - self.last_capture_time

        if dt > 0:
            self.capture_fps = 1.0 / dt

        self.last_capture_time = now

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

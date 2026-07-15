import cv2
import gi
import time
import threading
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from recognizer import FaceRecognizer
from shared import SharedState
from tracker import CentroidTracker, TemplateTracker
from config import *


# ----------------------------------------------------------
# Initialize GStreamer
# ----------------------------------------------------------

Gst.init(None)


PIPELINE = (
    f'rtspsrc location="{RTSP_URL}" latency={RTSP_LATENCY} ! '
    "rtph264depay ! "
    "h264parse ! "
    "nvv4l2decoder ! "
    "nvvidconv ! "
    "video/x-raw,format=BGRx ! "
    "videoconvert ! "
    "video/x-raw,format=BGR ! "
    "appsink "
    "name=sink "
    "emit-signals=true "
    "max-buffers=1 "
    "drop=true "
    "sync=false"
)


# ----------------------------------------------------------
# Shared Objects
# ----------------------------------------------------------

shared = SharedState()

recognizer = FaceRecognizer()
recognizer.load_database(FACE_DATABASE)

tracker = CentroidTracker(
    max_disappeared=TRACKER_MAX_DISAPPEARED,
    max_distance=TRACKER_MAX_DISTANCE
)


# ----------------------------------------------------------
# Capture Thread
# ----------------------------------------------------------

class CaptureThread(threading.Thread):

    def __init__(self):

        super().__init__(daemon=True)

        self.pipeline = Gst.parse_launch(PIPELINE)

        self.appsink = self.pipeline.get_by_name("sink")

        if self.appsink is None:
            raise RuntimeError("Could not create appsink")

    def run(self):

        print("Starting GStreamer pipeline...")

        self.pipeline.set_state(Gst.State.PLAYING)

        while shared.is_running():

            sample = self.appsink.emit("pull-sample")

            if sample is None:
                continue

            buffer = sample.get_buffer()
            caps = sample.get_caps()

            width = caps.get_structure(0).get_value("width")
            height = caps.get_structure(0).get_value("height")

            success, map_info = buffer.map(Gst.MapFlags.READ)

            if not success:
                continue

            frame = np.frombuffer(
                map_info.data,
                dtype=np.uint8
            ).copy()

            frame = frame.reshape(
                (height, width, 3)
            )

            buffer.unmap(map_info)

            shared.set_frame(frame)
            shared.update_capture_fps()

        print("Stopping Capture Thread...")

        self.pipeline.set_state(Gst.State.NULL)

# ----------------------------------------------------------
# Inference Thread
# ----------------------------------------------------------

class InferenceThread(threading.Thread):

    def __init__(self):

        super().__init__(daemon=True)

        self.frame_interval = 1.0 / INFERENCE_FPS

    def run(self):

        print("Inference thread started")

        while shared.is_running():

            start_time = time.time()

            frame = shared.get_frame()

            if frame is None:
                time.sleep(0.005)
                continue

            try:
                # 1. Run detection and recognition with hybrid downscaled/high-res pipeline
                results = recognizer.recognize(
                    frame,
                    inference_width=INFERENCE_WIDTH,
                    inference_height=INFERENCE_HEIGHT
                )

                # 2. Update centroid tracker with the results (list of (bbox, name))
                tracks = tracker.update(results)

                # 3. Update tracked results in shared state
                shared.set_tracks(list(tracks.values()))
                shared.update_inference_fps()

            except Exception as e:

                print(f"Inference error: {e}")

            elapsed = time.time() - start_time

            sleep_time = self.frame_interval - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

        print("Inference thread stopped")

# ----------------------------------------------------------
# Display Thread
# ----------------------------------------------------------

class DisplayThread(threading.Thread):

    def __init__(self):

        super().__init__(daemon=True)

        self.frame_interval = 1.0 / DISPLAY_FPS
        self.trackers = {}
        self.last_track_version = -1

    def run(self):

        print("Display thread started")

        while shared.is_running():

            start_time = time.time()

            frame = shared.get_frame()

            if frame is None:
                time.sleep(0.005)
                continue

            # Retrieve active tracks along with their version
            tracks, track_version = shared.get_tracks_with_version()

            if track_version != self.last_track_version:
                # 1. Inference thread updated tracks: initialize/correct template trackers
                active_ids = set()
                for track in tracks:
                    if track.disappeared > 0:
                        continue
                    active_ids.add(track.id)
                    # Initialize or re-template the tracker with fresh inference bbox coordinates
                    self.trackers[track.id] = TemplateTracker(frame, track.bbox)
                
                # Cleanup trackers for inactive track IDs
                for tid in list(self.trackers.keys()):
                    if tid not in active_ids:
                        self.trackers.pop(tid, None)
                
                self.last_track_version = track_version
            else:
                # 2. In between inference updates: track faces using cv2.matchTemplate at 30 FPS
                for track in tracks:
                    if track.disappeared > 0:
                        continue
                    t_tracker = self.trackers.get(track.id)
                    if t_tracker is not None:
                        success, bbox = t_tracker.update(frame)
                        if success:
                            track.bbox = bbox

            # Draw tracked faces
            for track in tracks:
                if track.disappeared > 0:
                    continue

                x1, y1, x2, y2 = map(int, track.bbox)
                display_name = f"{track.name or 'Unknown'} (ID: {track.id})"

                if DRAW_BOXES:
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                if DRAW_NAMES:
                    cv2.putText(
                        frame,
                        display_name,
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

            if SHOW_FPS:

                cv2.putText(
                    frame,
                    f"Capture : {shared.capture_fps:.1f}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Inference : {shared.inference_fps:.1f}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Display : {shared.display_fps:.1f}",
                    (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            shared.update_display_fps()

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                shared.stop()
                break

            elapsed = time.time() - start_time

            sleep_time = self.frame_interval - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

        cv2.destroyAllWindows()

        print("Display thread stopped")

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():

    print("=" * 60)
    print("Jetson Face Recognition")
    print("=" * 60)

    capture_thread = CaptureThread()
    inference_thread = InferenceThread()
    display_thread = DisplayThread()

    capture_thread.start()

    # Wait until the first frame arrives
    while shared.get_frame() is None:
        time.sleep(0.05)

    inference_thread.start()
    display_thread.start()

    try:

        while shared.is_running():
            time.sleep(0.5)

    except KeyboardInterrupt:

        print("\nStopping...")

        shared.stop()

    finally:

        capture_thread.join(timeout=2)

        inference_thread.join(timeout=2)

        display_thread.join(timeout=2)

        cv2.destroyAllWindows()

        print("Shutdown complete.")


if __name__ == "__main__":
    main()

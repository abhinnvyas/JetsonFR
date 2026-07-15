import cv2
import gi
import time
import threading
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from recognizer import FaceRecognizer
from shared import SharedState
from tracker import CentroidTracker
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
                # 1. Store original frame resolution
                h_orig, w_orig = frame.shape[:2]

                # 2. Resize frame for faster detection/recognition
                inference_frame = cv2.resize(
                    frame,
                    (INFERENCE_WIDTH, INFERENCE_HEIGHT)
                )

                # 3. Perform face recognition on the resized frame
                results = recognizer.recognize(inference_frame)

                # 4. Scale bounding box coordinates back to original size
                scale_x = w_orig / INFERENCE_WIDTH
                scale_y = h_orig / INFERENCE_HEIGHT

                bboxes = []
                for bbox, name in results:
                    x1, y1, x2, y2 = bbox
                    x1_scaled = int(x1 * scale_x)
                    y1_scaled = int(y1 * scale_y)
                    x2_scaled = int(x2 * scale_x)
                    y2_scaled = int(y2 * scale_y)
                    bboxes.append([x1_scaled, y1_scaled, x2_scaled, y2_scaled])

                # 5. Update centroid tracker with scaled bounding boxes
                tracks = tracker.update(bboxes)

                # 6. Associate names from inference with active tracker items
                for track_id, track in tracks.items():
                    if track.disappeared > 0:
                        continue
                    # Match track.bbox with scaled bboxes to find the recognized name
                    for bbox, (_, name) in zip(bboxes, results):
                        if track.bbox == bbox:
                            # Update name if valid name is recognized or name is currently unset
                            if name != "Unknown" or track.name is None:
                                track.name = name
                            break

                # 7. Update tracked results in shared state
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

    def run(self):

        print("Display thread started")

        while shared.is_running():

            start_time = time.time()

            frame = shared.get_frame()

            if frame is None:
                time.sleep(0.005)
                continue

            # Retrieve active tracks from shared state
            tracks = shared.get_tracks()

            # Draw tracked faces
            for track in tracks:
                # Skip inactive/disappeared tracks
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

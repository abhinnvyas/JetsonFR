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


# ----------------------------------------------------------
# Shared Objects
# ----------------------------------------------------------

shared = SharedState()

recognizer = FaceRecognizer()
recognizer.load_database(FACE_DATABASE)


# ----------------------------------------------------------
# Capture Thread
# ----------------------------------------------------------

class CaptureThread(threading.Thread):

    def __init__(self, camera_id, rtsp_url):

        super().__init__(daemon=True)

        self.camera_id = camera_id

        # Low-latency GStreamer pipeline string with leaky queues and appsink dropping
        self.pipeline_str = (
            f'rtspsrc location="{rtsp_url}" latency={RTSP_LATENCY} drop-on-latency=true ! '
            "queue max-size-buffers=1 leaky=downstream ! "
            "rtph264depay ! "
            "h264parse ! "
            "nvv4l2decoder ! "
            "nvvidconv ! "
            "video/x-raw,format=BGRx ! "
            "queue max-size-buffers=1 leaky=downstream ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink "
            f"name=sink_{camera_id} "
            "emit-signals=true "
            "max-buffers=1 "
            "drop=true "
            "sync=false"
        )

        self.pipeline = Gst.parse_launch(self.pipeline_str)
        self.appsink = self.pipeline.get_by_name(f"sink_{camera_id}")

        if self.appsink is None:
            raise RuntimeError(f"Could not create appsink for camera {camera_id}")

    def run(self):

        print(f"Starting GStreamer pipeline for camera {self.camera_id}...")

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

            frame = frame.reshape((height, width, 3))

            buffer.unmap(map_info)

            shared.set_frame(self.camera_id, frame)
            shared.update_capture_fps(self.camera_id)

        print(f"Stopping Capture Thread for camera {self.camera_id}...")

        self.pipeline.set_state(Gst.State.NULL)


# ----------------------------------------------------------
# Inference Thread
# ----------------------------------------------------------

class InferenceThread(threading.Thread):

    def __init__(self, camera_ids):

        super().__init__(daemon=True)

        self.camera_ids = camera_ids
        self.frame_interval = 1.0 / INFERENCE_FPS

        # A separate CentroidTracker instance for each camera stream
        self.trackers = {
            cam_id: CentroidTracker(
                max_disappeared=TRACKER_MAX_DISAPPEARED,
                max_distance=TRACKER_MAX_DISTANCE
            )
            for cam_id in camera_ids
        }

    def run(self):

        print("Inference thread started")

        while shared.is_running():

            start_time = time.time()

            # Cycle round-robin through all configured camera streams
            for camera_id in self.camera_ids:

                frame = shared.get_frame(camera_id)

                if frame is None:
                    continue

                try:
                    # Run detection and recognition with hybrid downscaled/high-res pipeline
                    results = recognizer.recognize(
                        frame,
                        inference_width=INFERENCE_WIDTH,
                        inference_height=INFERENCE_HEIGHT
                    )

                    # Update this camera's centroid tracker
                    tracks = self.trackers[camera_id].update(results)

                    # Store results in the shared state
                    shared.set_tracks(camera_id, list(tracks.values()))

                except Exception as e:

                    print(f"Inference error on camera {camera_id}: {e}")

            shared.update_inference_fps()

            elapsed = time.time() - start_time

            sleep_time = self.frame_interval - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

        print("Inference thread stopped")


# ----------------------------------------------------------
# Display Thread
# ----------------------------------------------------------

class DisplayThread(threading.Thread):

    def __init__(self, camera_ids):

        super().__init__(daemon=True)

        self.camera_ids = camera_ids
        self.frame_interval = 1.0 / DISPLAY_FPS

        # Dictionary of trackers and tracking versions per camera stream
        self.template_trackers = {cam_id: {} for cam_id in camera_ids}
        self.last_track_versions = {cam_id: -1 for cam_id in camera_ids}

    def run(self):

        print("Display thread started")

        while shared.is_running():

            start_time = time.time()

            # 1. Grab latest frames from all camera feeds
            active_frames = {}
            for camera_id in self.camera_ids:
                frame = shared.get_frame(camera_id)
                if frame is not None:
                    active_frames[camera_id] = frame

            if len(active_frames) == 0:
                time.sleep(0.005)
                continue

            # 2. Update tracking and draw overlays on each active feed
            for camera_id, frame in active_frames.items():

                tracks, track_version = shared.get_tracks_with_version(camera_id)

                if track_version != self.last_track_versions[camera_id]:
                    # Fresh inference tracks available: correct/initialize template trackers
                    active_ids = set()
                    for track in tracks:
                        if track.disappeared > 0:
                            continue
                        active_ids.add(track.id)
                        self.template_trackers[camera_id][track.id] = TemplateTracker(frame, track.bbox)

                    # Remove stale template trackers
                    for tid in list(self.template_trackers[camera_id].keys()):
                        if tid not in active_ids:
                            self.template_trackers[camera_id].pop(tid, None)

                    self.last_track_versions[camera_id] = track_version

                else:
                    # In-between inference frames: update TemplateTrackers at 30 FPS
                    for track in tracks:
                        if track.disappeared > 0:
                            continue
                        t_tracker = self.template_trackers[camera_id].get(track.id)
                        if t_tracker is not None:
                            success, bbox = t_tracker.update(frame)
                            if success:
                                track.bbox = bbox

                # Draw bounding boxes and name labels on this frame
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

                # Render camera performance statistics
                if SHOW_FPS:
                    cap_fps = shared.capture_fps.get(camera_id, 0.0)
                    cv2.putText(
                        frame,
                        f"Cam {camera_id} Cap: {cap_fps:.1f}",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2
                    )
                    cv2.putText(
                        frame,
                        f"Inf: {shared.inference_fps:.1f} | Disp: {shared.display_fps:.1f}",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2
                    )

            # 3. Create and show a grid layout combining all active streams
            grid_frame = self.create_grid(active_frames)

            if grid_frame is not None:
                cv2.imshow(
                    WINDOW_NAME,
                    grid_frame
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

    def create_grid(self, frames):
        """
        Resize and combine multiple camera frames into a single grid frame.
        Supports 1, 2, 3, or 4 camera feeds.
        """
        num_cameras = len(frames)
        if num_cameras == 0:
            return None
        if num_cameras == 1:
            return list(frames.values())[0]

        # Target dimensions for each grid element
        gw, gh = 640, 360
        resized_frames = []
        for cam_id in sorted(frames.keys()):
            resized = cv2.resize(frames[cam_id], (gw, gh))
            resized_frames.append(resized)

        if num_cameras == 2:
            return np.hstack(resized_frames)

        # Pad layout to 4 cells for 3-camera feeds
        while len(resized_frames) < 4:
            resized_frames.append(np.zeros((gh, gw, 3), dtype=np.uint8))

        top_row = np.hstack(resized_frames[:2])
        bottom_row = np.hstack(resized_frames[2:])
        return np.vstack([top_row, bottom_row])


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():

    print("=" * 60)
    print("Jetson Multi-Camera Face Recognition")
    print("=" * 60)

    # Resolve camera indices from the config file RTSP_URLS list
    camera_ids = list(range(len(RTSP_URLS)))

    # Instantiate and start capture thread for each camera
    capture_threads = []
    for camera_id, rtsp_url in enumerate(RTSP_URLS):
        cap_thread = CaptureThread(camera_id, rtsp_url)
        capture_threads.append(cap_thread)
        cap_thread.start()

    # Block until at least one frame is captured from any feed
    print("Waiting for camera feeds to initialize...")
    initialized = False
    while not initialized and shared.is_running():
        for camera_id in camera_ids:
            if shared.get_frame(camera_id) is not None:
                initialized = True
                break
        time.sleep(0.05)

    # Launch inference and window display threads
    inference_thread = InferenceThread(camera_ids)
    display_thread = DisplayThread(camera_ids)

    inference_thread.start()
    display_thread.start()

    try:
        while shared.is_running():
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopping...")
        shared.stop()

    finally:
        for cap_thread in capture_threads:
            cap_thread.join(timeout=2)

        inference_thread.join(timeout=2)
        display_thread.join(timeout=2)

        cv2.destroyAllWindows()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()

import cv2
import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from recognizer import FaceRecognizer
from config import *


# Initialize GStreamer
Gst.init(None)


PIPELINE = (
    f'rtspsrc location="{RTSP_URL}" latency=200 ! '
    "rtph264depay ! "
    "h264parse ! "
    "nvv4l2decoder ! "
    "nvvidconv ! "
    "video/x-raw,format=BGRx ! "
    "videoconvert ! "
    "video/x-raw,format=BGR ! "
    "appsink name=sink emit-signals=true "
    "max-buffers=1 drop=true sync=false"
)


print("Starting GStreamer pipeline:")
print(PIPELINE)


# Create pipeline
pipeline = Gst.parse_launch(PIPELINE)

appsink = pipeline.get_by_name("sink")

if appsink is None:
    raise RuntimeError("Could not create appsink")


# Start pipeline
pipeline.set_state(Gst.State.PLAYING)


print("GStreamer pipeline started")


# Load face database
recognizer = FaceRecognizer()

recognizer.load_database(FACE_DATABASE)


try:

    while True:

        # Get frame from GStreamer
        sample = appsink.emit("pull-sample")

        if sample is None:
            print("No frame received")
            continue


        buffer = sample.get_buffer()
        caps = sample.get_caps()


        width = caps.get_structure(0).get_value("width")
        height = caps.get_structure(0).get_value("height")


        success, map_info = buffer.map(
            Gst.MapFlags.READ
        )

        if not success:
            continue


        # Convert GStreamer buffer to numpy array
        frame = np.frombuffer(
            map_info.data,
            dtype=np.uint8
        ).copy()


        frame = frame.reshape(
            (height, width, 3)
        )


        buffer.unmap(map_info)


        # Face recognition
        results = recognizer.recognize(frame)


        for bbox, name in results:

            x1, y1, x2, y2 = bbox


            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            cv2.putText(
                frame,
                name,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


        cv2.imshow(
            "Face Recognition",
            frame
        )


        # ESC key exits
        if cv2.waitKey(1) == 27:
            break


finally:

    print("Stopping pipeline")

    pipeline.set_state(
        Gst.State.NULL
    )

    cv2.destroyAllWindows()

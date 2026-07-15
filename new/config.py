RTSP_URLS = [
    "rtsp://admin:123456Ai@192.168.1.69:554/snl/live/1/1",
    # You can add more RTSP stream URLs here, e.g.:
    # "rtsp://admin:123456Ai@192.168.1.70:554/snl/live/1/1",
]

FACE_DATABASE = "faces"

SIMILARITY_THRESHOLD = 0.45

# Display FPS (camera)
DISPLAY_FPS = 30

# AI inference FPS
# Lower value = less GPU load
INFERENCE_FPS = 15

# RTSP latency
RTSP_LATENCY = 100

# Draw FPS counter
SHOW_FPS = True

# Window title
WINDOW_NAME = "Face Recognition"

# Draw face boxes
DRAW_BOXES = True

# Draw names
DRAW_NAMES = True

# Inference frame downscaling dimensions
INFERENCE_WIDTH = 640
INFERENCE_HEIGHT = 360

# Centroid tracker configurations
TRACKER_MAX_DISAPPEARED = 15
TRACKER_MAX_DISTANCE = 100


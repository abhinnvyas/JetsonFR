import cv2
import numpy as np
import os

from insightface.app import FaceAnalysis
from insightface.app.common import Face
from config import SIMILARITY_THRESHOLD

class FaceRecognizer:

    def __init__(self):

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider","CPUExecutionProvider"]
        )

        self.app.prepare(ctx_id=0)

        # Print active ONNX Runtime execution providers for debugging performance
        print("[FaceRecognizer] Active execution providers for loaded models:")
        for model_name, model in self.app.models.items():
            if hasattr(model, 'session'):
                print(f"  - Model '{model_name}': {model.session.get_providers()}")
            else:
                print(f"  - Model '{model_name}': Session not available")

        self.known_embeddings = []
        self.known_names = []

    def load_database(self, folder):

        for filename in os.listdir(folder):

            path = os.path.join(folder, filename)

            image = cv2.imread(path)

            if image is None:
                continue

            faces = self.app.get(image)

            if len(faces) == 0:
                continue

            embedding = faces[0].normed_embedding

            self.known_embeddings.append(embedding)

            self.known_names.append(
                os.path.splitext(filename)[0]
            )

        self.known_embeddings = np.array(self.known_embeddings)

    def recognize(self, frame, inference_width=640, inference_height=360):
        # 1. Get original frame size
        h_orig, w_orig = frame.shape[:2]

        # 2. Resize frame for fast face detection
        resized_frame = cv2.resize(frame, (inference_width, inference_height))

        # 3. Detect faces on resized frame
        bboxes, kpss = self.app.det_model.detect(resized_frame, max_num=0, metric='default')

        if bboxes.shape[0] == 0:
            return []

        # 4. Scale detection coordinates and keypoints back to original size
        scale_x = w_orig / inference_width
        scale_y = h_orig / inference_height

        results = []

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4].copy()
            bbox[0] *= scale_x
            bbox[1] *= scale_y
            bbox[2] *= scale_x
            bbox[3] *= scale_y

            det_score = bboxes[i, 4]

            kps = None
            if kpss is not None:
                kps = kpss[i].copy()
                kps[:, 0] *= scale_x
                kps[:, 1] *= scale_y

            # Create Face object in the original coordinate system
            face = Face(bbox=bbox, kps=kps, det_score=det_score)

            # 5. Run other models (alignment & recognition) on the original high-resolution frame
            for model_name, model in self.app.models.items():
                if model_name == 'detection':
                    continue
                model.get(frame, face)

            # 6. Perform similarity matching against known embeddings
            embedding = face.normed_embedding

            if len(self.known_embeddings) == 0:
                name = "Unknown"
            else:
                scores = np.dot(self.known_embeddings, embedding)
                idx = np.argmax(scores)
                score = scores[idx]

                # Diagnostic print to check actual similarity scores
                print(f"[FaceRecognizer] Match score with '{self.known_names[idx]}': {score:.4f} (threshold: {SIMILARITY_THRESHOLD})")

                if score > SIMILARITY_THRESHOLD:
                    name = self.known_names[idx]
                else:
                    name = "Unknown"

            results.append((face.bbox.astype(int), name))

        return results

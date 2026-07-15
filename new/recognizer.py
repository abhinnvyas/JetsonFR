import cv2
import numpy as np
import os

from insightface.app import FaceAnalysis

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

            embedding = faces[0].embedding

            self.known_embeddings.append(embedding)

            self.known_names.append(
                os.path.splitext(filename)[0]
            )

        self.known_embeddings = np.array(self.known_embeddings)

    def recognize(self, frame):

        faces = self.app.get(frame)

        results = []

        for face in faces:

            embedding = face.embedding

            if len(self.known_embeddings) == 0:

                name = "Unknown"

            else:

                scores = np.dot(
                    self.known_embeddings,
                    embedding
                )

                idx = np.argmax(scores)

                score = scores[idx]

                if score > 0.45:
                    name = self.known_names[idx]
                else:
                    name = "Unknown"

            results.append(
                (
                    face.bbox.astype(int),
                    name
                )
            )

        return results

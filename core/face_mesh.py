import os
import cv2
import numpy as np
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = "face_landmarker.task"

def download_model():

    if not os.path.exists(MODEL_PATH):
        print("Downloading face_landmarker.task model (~30 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully!")
    else:
        print("Model file already exists, download skipped.")

class FaceMeshExtractor:

    def __init__(self, model_path=MODEL_PATH):

        download_model()

        base_options = mp_python.BaseOptions(
            model_asset_path=model_path
        )

        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )

        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        print("FaceLandmarker initialized successfully!")

    def process_frame(self, frame):

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        results = self.landmarker.detect(mp_image)

        if not results.face_landmarks:
            return None

        face_landmarks = results.face_landmarks[0]

        landmarks = [
            (lm.x, lm.y, lm.z)
            for lm in face_landmarks
        ]

        return landmarks

    def draw_landmarks(self, frame, landmarks):

        if landmarks is None:
            return frame

        h, w, _ = frame.shape

        for (x, y, z) in landmarks:
            px, py = int(x * w), int(y * h)
            cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)

        return frame

    def release(self):

        self.landmarker.close()


if __name__ == "__main__":
    extractor = FaceMeshExtractor()
    cap = cv2.VideoCapture(0)

    print("Camera initialized. Press 'Q' to exit.")
    print("Successful mesh generation will display green landmarks.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture camera frame.")
            break

        landmarks = extractor.process_frame(frame)

        if landmarks:
            frame = extractor.draw_landmarks(frame, landmarks)
            print(
                f"Landmarks detected: {len(landmarks)} | "
                f"Point[0]: x={landmarks[0][0]:.3f}, "
                f"y={landmarks[0][1]:.3f}, "
                f"z={landmarks[0][2]:.3f}",
                end="\r"
            )
        else:
            cv2.putText(
                frame, "Face Not Found", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )

        cv2.imshow("FaceAuthSystem — Face Mesh Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    extractor.release()
    cv2.destroyAllWindows()
    print("\nTesting completed.")
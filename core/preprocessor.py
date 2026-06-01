import os
import cv2
import numpy as np
import math

os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class ImagePreprocessor:

    def __init__(self, clahe_clip_limit=2.0, clahe_tile_size=(8, 8)):

        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_size
        )

    def process(self, frame, landmarks=None):

        result = frame.copy()

        if landmarks is not None:
            result = self._face_alignment(result, landmarks)

        result = self._apply_clahe(result)
        return result

    def _face_alignment(self, frame, landmarks):

        h, w, _ = frame.shape

        LEFT_EYE_IDX  = 263
        RIGHT_EYE_IDX = 33

        lx = int(landmarks[LEFT_EYE_IDX][0] * w)
        ly = int(landmarks[LEFT_EYE_IDX][1] * h)
        rx = int(landmarks[RIGHT_EYE_IDX][0] * w)
        ry = int(landmarks[RIGHT_EYE_IDX][1] * h)

        delta_y = ry - ly
        delta_x = rx - lx

        if delta_x == 0:
            return frame

        angle_deg = math.degrees(math.atan2(delta_y, delta_x))

        if abs(angle_deg) < 1.0 or abs(angle_deg) > 30.0:
            return frame

        center = ((lx + rx) // 2, (ly + ry) // 2)
        m_affine = cv2.getRotationMatrix2D(center, angle_deg, scale=1.0)

        aligned_frame = cv2.warpAffine(
            frame, m_affine, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        return aligned_frame

    def _apply_clahe(self, frame):

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        l_clahe = self.clahe.apply(l_channel)
        lab_clahe = cv2.merge([l_clahe, a_channel, b_channel])

        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from core.face_mesh import FaceMeshExtractor

    preprocessor = ImagePreprocessor()
    extractor    = FaceMeshExtractor()
    cap          = cv2.VideoCapture(0)

    print("Testing Image Preprocessor. Press 'Q' to exit.")
    print("Left Window = Source Stream | Right Window = CLAHE + Alignment Pipeline")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = extractor.process_frame(frame)
        processed = preprocessor.process(frame, landmarks)

        cv2.imshow("Original", frame)
        cv2.imshow("After Preprocessing (CLAHE + Alignment)", processed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    extractor.release()
    cv2.destroyAllWindows()
    print("Testing completed.")
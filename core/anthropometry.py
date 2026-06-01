import numpy as np
import math

class AnthropometryAnalyzer:

    def __init__(self):

        self.LEFT_INNER_EYE  = 362
        self.RIGHT_INNER_EYE = 133

        self.LANDMARK_PAIRS = [
            (362, 263,  'left_eye_width'),
            (33,  133,  'right_eye_width'),
            (159, 145,  'left_eye_height'),
            (386, 374,  'right_eye_height'),
            (362, 133,  'interpupillary_distance'),

            (1,   168,  'nose_bridge_height'),
            (129, 358,  'nose_width'),
            (168, 6,    'nose_ridge_length'),
            (1,   2,    'nose_tip_vertical'),

            (168, 0,    'bridge_to_upper_lip'),
            (362, 1,    'left_eye_to_nose'),
            (133, 1,    'right_eye_to_nose'),
            (1,   17,   'nose_to_chin'),
            (10,  152,  'face_height'),
            (234, 454,  'face_width'),

            (61,  291,  'mouth_width'),
            (0,   17,   'mouth_height'),
            (13,  14,   'inter_lip_distance'),

            (70,  300,  'inter_eyebrow_width'),
            (105, 334,  'eyebrow_to_eye_height'),
        ]

    def _euclidean_distance(self, p1, p2):

        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2 +
            (p1[2] - p2[2]) ** 2
        )

    def extract_embedding(self, landmarks):

        if landmarks is None or len(landmarks) < 468:
            return None

        p_left  = landmarks[self.LEFT_INNER_EYE]
        p_right = landmarks[self.RIGHT_INNER_EYE]
        d_base  = self._euclidean_distance(p_left, p_right)

        if d_base < 1e-6:
            return None

        feature_vector = []
        for (idx_a, idx_b, _) in self.LANDMARK_PAIRS:
            pa = landmarks[idx_a]
            pb = landmarks[idx_b]
            distance = self._euclidean_distance(pa, pb)
            feature_vector.append(distance / d_base)

        return np.array(feature_vector, dtype=np.float32)

    def get_feature_names(self):

        return [name for (_, _, name) in self.LANDMARK_PAIRS]


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ["GLOG_minloglevel"] = "3"

    import cv2
    from core.face_mesh import FaceMeshExtractor
    from core.preprocessor import ImagePreprocessor

    extractor    = FaceMeshExtractor()
    preprocessor = ImagePreprocessor()
    analyzer     = AnthropometryAnalyzer()
    cap          = cv2.VideoCapture(0)

    print("Testing Anthropometry Analyzer. Press 'Q' to exit.")
    print(f"Feature vector dimensionality: {len(analyzer.LANDMARK_PAIRS)}")
    print(f"Feature names: {analyzer.get_feature_names()}\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = extractor.process_frame(frame)
        processed = preprocessor.process(frame, landmarks)
        embedding = analyzer.extract_embedding(landmarks)

        if embedding is not None:
            frame = extractor.draw_landmarks(frame, landmarks)
            print(f"Embedding vector [{len(embedding)} values]:", end=" ")
            print(" | ".join(f"{v:.3f}" for v in embedding), end="\r")
        else:
            cv2.putText(frame, "Face Not Found", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Anthropometry Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    extractor.release()
    cv2.destroyAllWindows()
    print("\nTesting completed.")
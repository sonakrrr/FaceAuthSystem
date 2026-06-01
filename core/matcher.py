import numpy as np

class BiometricMatcher:

    def __init__(self, euclidean_threshold=0.35, cosine_threshold=0.92):

        self.euclidean_threshold = euclidean_threshold
        self.cosine_threshold    = cosine_threshold

    def euclidean_distance(self, vec1, vec2):

        return float(np.linalg.norm(vec1 - vec2))

    def cosine_similarity(self, vec1, vec2):

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def verify(self, current_vector, saved_vector):

        if current_vector is None or saved_vector is None:
            return {
                'authenticated': False,
                'euclidean'    : None,
                'cosine'       : None,
                'eu_passed'    : False,
                'cos_passed'   : False,
            }

        eu_dist  = self.euclidean_distance(current_vector, saved_vector)
        cos_sim  = self.cosine_similarity(current_vector, saved_vector)

        eu_passed  = eu_dist <= self.euclidean_threshold
        cos_passed = cos_sim >= self.cosine_threshold

        authenticated = eu_passed and cos_passed

        return {
            'authenticated': authenticated,
            'euclidean'    : round(eu_dist, 4),
            'cosine'       : round(cos_sim, 4),
            'eu_passed'    : eu_passed,
            'cos_passed'   : cos_passed,
        }

    def update_thresholds(self, euclidean_threshold=None, cosine_threshold=None):

        if euclidean_threshold is not None:
            self.euclidean_threshold = euclidean_threshold
        if cosine_threshold is not None:
            self.cosine_threshold = cosine_threshold


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import cv2
    from core.face_mesh import FaceMeshExtractor
    from core.preprocessor import ImagePreprocessor
    from core.anthropometry import AnthropometryAnalyzer

    extractor    = FaceMeshExtractor()
    preprocessor = ImagePreprocessor()
    analyzer     = AnthropometryAnalyzer()
    matcher      = BiometricMatcher()
    cap          = cv2.VideoCapture(0)

    reference_vector = None
    print("=" * 60)
    print("Biometric Matcher Test Pipeline")
    print("Step 1: Look at the camera and press ENTER to save reference template.")
    print("Step 2: The system will verify your identity in real-time.")
    print("Press 'Q' to exit.")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = extractor.process_frame(frame)
        embedding = analyzer.extract_embedding(landmarks)

        if embedding is not None:
            frame = extractor.draw_landmarks(frame, landmarks)

            if reference_vector is None:
                cv2.putText(frame, "Press ENTER to save reference template",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)
            else:
                result = matcher.verify(embedding, reference_vector)

                eu   = result['euclidean']
                cos  = result['cosine']
                auth = result['authenticated']

                color = (0, 255, 0) if auth else (0, 0, 255)
                label = "ACCESS GRANTED" if auth else "ACCESS DENIED"

                cv2.putText(frame, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"Euclidean: {eu:.4f} (threshold <= {matcher.euclidean_threshold})",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 0) if result['eu_passed'] else (0, 0, 255), 1)
                cv2.putText(frame, f"Cosine: {cos:.4f} (threshold >= {matcher.cosine_threshold})",
                            (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 0) if result['cos_passed'] else (0, 0, 255), 1)

                print(f"Euclidean: {eu:.4f} {'✓' if result['eu_passed'] else '✗'} | "
                      f"Cosine: {cos:.4f} {'✓' if result['cos_passed'] else '✗'} | "
                      f"{'ACCESS GRANTED ✓' if auth else 'DENIED ✗'}",
                      end="\r")
        else:
            cv2.putText(frame, "Face Not Found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Biometric Matcher Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 13:  # Enter Key
            if embedding is not None:
                reference_vector = embedding.copy()
                print("\nReference profile saved! Starting verification pipeline...")

    cap.release()
    extractor.release()
    cv2.destroyAllWindows()
    print("\nTesting completed.")
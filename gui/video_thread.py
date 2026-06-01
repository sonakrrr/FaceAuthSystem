import os
import sys
import cv2
import numpy as np

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.face_mesh import FaceMeshExtractor
from core.preprocessor import ImagePreprocessor
from core.anthropometry import AnthropometryAnalyzer
from core.matcher import BiometricMatcher
from core.liveness_detector import LivenessDetector
from core.texture_liveness import TextureLivenessDetector

class VideoThread(QThread):

    frame_ready     = Signal(QImage)
    auth_result     = Signal(bool, float, float)
    status_message  = Signal(str)
    liveness_status = Signal(str)
    error_occurred  = Signal(str)

    MODE_IDLE     = "idle"
    MODE_REGISTER = "register"
    MODE_AUTH     = "auth"

    def __init__(self, db_manager):
        super().__init__()

        self.db = db_manager

        self.extractor    = FaceMeshExtractor()
        self.preprocessor = ImagePreprocessor()
        self.analyzer     = AnthropometryAnalyzer()
        self.matcher      = BiometricMatcher()

        self.liveness         = LivenessDetector()
        self.texture_liveness = TextureLivenessDetector()

        self.mode         = self.MODE_IDLE
        self.current_user = None
        self.is_running   = False

        self._register_embeddings    = []
        self._register_frames_needed = 30
        self._auth_logged            = False

    def run(self):

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            self.error_occurred.emit("Critical Error: Unable to open camera video graph.")
            return

        self.is_running = True
        self.status_message.emit("Camera video capture stream initialized successfully.")

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                self.error_occurred.emit("Hardware Error: Failed to fetch video frame stream.")
                break

            frame = cv2.flip(frame, 1)

            landmarks = self.extractor.process_frame(frame)
            processed_frame = self.preprocessor.process(frame, landmarks)

            embedding = None
            if landmarks:
                embedding = self.analyzer.extract_embedding(landmarks)

            display_frame = processed_frame.copy()
            if landmarks:
                display_frame = self.extractor.draw_landmarks(display_frame, landmarks)

            if self.mode == self.MODE_REGISTER:
                display_frame = self._handle_register(display_frame, embedding, landmarks)
            elif self.mode == self.MODE_AUTH:
                display_frame = self._handle_auth(display_frame, embedding, landmarks)
            else:
                if landmarks is None:
                    cv2.putText(
                        display_frame,
                        "Please align face within camera bounds",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 255), 2
                    )

            qt_image = self._convert_to_qimage(display_frame)
            self.frame_ready.emit(qt_image)

        cap.release()
        self.extractor.release()

    def _handle_register(self, frame, embedding, landmarks):

        if embedding is None or landmarks is None:
            cv2.putText(frame, "Face Not Found", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return frame

        tex_res = self.texture_liveness.check_texture(frame, landmarks)
        if tex_res['is_spoof']:
            self._draw_hardware_block_ui(frame, tex_res)
            return frame

        liveness_res = self.liveness.check(landmarks)
        if not liveness_res['is_live']:
            self._draw_liveness_instructions(frame, liveness_res, tex_res)
            return frame

        self._register_embeddings.append(embedding)
        collected = len(self._register_embeddings)
        needed    = self._register_frames_needed

        progress = int((collected / needed) * 100)
        cv2.putText(
            frame,
            f"Registration: {progress}% ({collected}/{needed} frames)",
            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2
        )
        cv2.putText(
            frame, "Look straight into the lens",
            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
        )

        bar_x, bar_y, bar_w, bar_h = 10, 80, 300, 15
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * collected / needed), bar_y + bar_h), (0, 255, 0), -1)

        if collected >= needed:
            avg_embedding = np.mean(self._register_embeddings, axis=0)
            success = self.db.save_user(self.current_user, avg_embedding)

            if success:
                self.status_message.emit(f"Profile '{self.current_user}' registered successfully.")
            else:
                self.status_message.emit(f"Error: Profile '{self.current_user}' already exists.")

            self._register_embeddings = []
            self.mode         = self.MODE_IDLE
            self.current_user = None

        return frame

    def _handle_auth(self, frame, embedding, landmarks):

        if embedding is None or landmarks is None:
            cv2.putText(frame, "Face Not Found", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return frame

        tex_res = self.texture_liveness.check_texture(frame, landmarks)
        if tex_res['is_spoof']:
            self._draw_hardware_block_ui(frame, tex_res)
            return frame

        liveness_res = self.liveness.check(landmarks)
        if not liveness_res['is_live']:
            self._draw_liveness_instructions(frame, liveness_res, tex_res)
            return frame

        saved_embedding = self.db.get_user(self.current_user)
        if saved_embedding is None:
            cv2.putText(
                frame, f"Profile '{self.current_user}' Not Found",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
            return frame

        result = self.matcher.verify(embedding, saved_embedding)
        authenticated = result['authenticated']
        eu_dist       = result['euclidean']
        cos_sim       = result['cosine']

        self.auth_result.emit(authenticated, eu_dist, cos_sim)

        if authenticated and not self._auth_logged:
            self.db.log_auth_attempt(
                self.current_user,
                success=True,
                euclidean=eu_dist,
                cosine=cos_sim
            )
            self._auth_logged = True

        color = (0, 255, 0) if authenticated else (0, 0, 255)
        label = "ACCESS GRANTED" if authenticated else "ACCESS DENIED"

        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 4)
        cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(
            frame, f"Euclidean: {eu_dist:.4f} | Cosine: {cos_sim:.4f}",
            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1
        )
        cv2.putText(
            frame, f"User Identity: {self.current_user}",
            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
        )

        return frame

    def _draw_hardware_block_ui(self, frame, tex_res):

        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
        cv2.putText(frame, "HARDWARE SPOOF DETECTED", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        reason = "Color Distribution Anomaly" if tex_res['is_color_anomaly'] else "Low Texture Variance (Blur)"
        cv2.putText(frame, f"Reason: {reason}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)
        cv2.putText(frame, f"TEX Var: {tex_res['laplacian_var']} | Cr: {tex_res['mean_cr']} | Cb: {tex_res['mean_cb']}",
                    (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        self.liveness_status.emit("Hardware Spoof Warning Triggered")

    def _draw_liveness_instructions(self, frame, liveness_res, tex_res):

        color = (0, 165, 255)
        msg = liveness_res['instruction']
        self.liveness_status.emit(msg)

        cv2.putText(frame, f"Anti-Spoofing: {msg}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(
            frame,
            f"EAR: {liveness_res['ear']:.2f} | 3D Yaw: {liveness_res['yaw_ratio']:.2f} | TEX Var: {tex_res['laplacian_var']}",
            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )

    def start_registration(self, username):

        self.liveness.reset()
        self._register_embeddings = []
        self.current_user = username
        self.mode         = self.MODE_REGISTER
        self.status_message.emit(f"Initializing profile registration pipeline for user: {username}")

    def start_auth(self, username):

        self.liveness.reset()
        self.current_user = username
        self._auth_logged = False
        self.mode         = self.MODE_AUTH
        self.status_message.emit(f"Running profile identity authentication verification for user: {username}")

    def set_idle(self):

        if self.mode == self.MODE_AUTH and not self._auth_logged and self.current_user:
            self.db.log_auth_attempt(
                self.current_user,
                success=False,
                euclidean=None,
                cosine=None
            )

        self.mode         = self.MODE_IDLE
        self.current_user = None
        self._register_embeddings = []
        self._auth_logged = False
        self.liveness.reset()

    def update_thresholds(self, euclidean, cosine):

        self.matcher.update_thresholds(euclidean, cosine)

    def stop(self):

        self.is_running = False
        self.wait()

    def _convert_to_qimage(self, frame):

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch  = rgb_frame.shape
        bytes_per_line = ch * w

        return QImage(
            rgb_frame.data, w, h,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )
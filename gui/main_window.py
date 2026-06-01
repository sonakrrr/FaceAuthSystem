import os
import sys

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QMessageBox, QGroupBox, QSlider, QStatusBar,
    QSizePolicy, QFrame
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui  import QPixmap, QFont, QColor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import DatabaseManager
from gui.video_thread    import VideoThread

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.db = DatabaseManager()
        self.video_thread = None

        self.setWindowTitle("Biometric Authentication System")
        self.setMinimumSize(900, 650)

        self._build_ui()
        self._apply_styles()
        self._start_video_thread()

    def _build_ui(self):

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        main_layout.addWidget(self._build_video_panel(), stretch=3)
        main_layout.addWidget(self._build_control_panel(), stretch=2)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("System ready")

    def _build_video_panel(self):

        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(frame)

        title = QLabel("Camera Video Stream")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        layout.addWidget(title)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(540, 400)
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        self.video_label.setStyleSheet(
            "background-color: #1a1a2e; border-radius: 8px;"
        )
        self.video_label.setText("Loading camera graph...")
        layout.addWidget(self.video_label)

        return frame

    def _build_control_panel(self):

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)

        layout.addWidget(self._build_user_block())
        layout.addWidget(self._build_actions_block())
        layout.addWidget(self._build_result_block())
        layout.addWidget(self._build_thresholds_block())
        layout.addStretch()

        return panel

    def _build_user_block(self):

        group = QGroupBox("Identity Profile")
        layout = QVBoxLayout(group)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter username...")
        self.name_input.setMinimumHeight(35)
        layout.addWidget(self.name_input)

        return group

    def _build_actions_block(self):

        group = QGroupBox("System Actions")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        self.btn_register = QPushButton("📷  Register Profile")
        self.btn_register.setMinimumHeight(42)
        self.btn_register.setFont(QFont("Arial", 10))
        self.btn_register.clicked.connect(self._on_register_clicked)
        layout.addWidget(self.btn_register)

        self.btn_auth = QPushButton("🔐  Authenticate Identity")
        self.btn_auth.setMinimumHeight(42)
        self.btn_auth.setFont(QFont("Arial", 10))
        self.btn_auth.clicked.connect(self._on_auth_clicked)
        layout.addWidget(self.btn_auth)

        self.btn_cancel = QPushButton("✕  Cancel Operation")
        self.btn_cancel.setMinimumHeight(35)
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)
        self.btn_cancel.setEnabled(False)
        layout.addWidget(self.btn_cancel)

        return group

    def _build_result_block(self):

        group = QGroupBox("Verification Results")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        self.result_label = QLabel("—")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setMinimumHeight(50)
        self.result_label.setFont(QFont("Arial", 13, QFont.Weight.Bold))
        self.result_label.setStyleSheet(
            "background-color: #2d2d2d; border-radius: 6px; color: #aaaaaa;"
        )
        layout.addWidget(self.result_label)

        self.label_liveness = QLabel("Anti-Spoofing: Idle")
        self.label_liveness.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        self.label_liveness.setStyleSheet("color: #f9e2af;")
        layout.addWidget(self.label_liveness)

        self.label_euclidean = QLabel("Euclidean Distance: —")
        self.label_cosine    = QLabel("Cosine Similarity: —")
        for lbl in (self.label_euclidean, self.label_cosine):
            lbl.setFont(QFont("Consolas", 9))
            layout.addWidget(lbl)

        return group

    def _build_thresholds_block(self):

        group = QGroupBox("Decision Threshold Adjustments")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        self.lbl_eu = QLabel("Euclidean Distance Cap: 0.35")
        layout.addWidget(self.lbl_eu)

        self.slider_eu = QSlider(Qt.Orientation.Horizontal)
        self.slider_eu.setRange(10, 80)
        self.slider_eu.setValue(35)
        self.slider_eu.valueChanged.connect(self._on_eu_changed)
        layout.addWidget(self.slider_eu)

        self.lbl_cos = QLabel("Cosine Similarity Floor: 0.92")
        layout.addWidget(self.lbl_cos)

        self.slider_cos = QSlider(Qt.Orientation.Horizontal)
        self.slider_cos.setRange(70, 99)
        self.slider_cos.setValue(92)
        self.slider_cos.valueChanged.connect(self._on_cos_changed)
        layout.addWidget(self.slider_cos)

        return group

    def _start_video_thread(self):

        self.video_thread = VideoThread(self.db)

        self.video_thread.frame_ready.connect(self._update_frame)
        self.video_thread.auth_result.connect(self._handle_auth_result)
        self.video_thread.status_message.connect(self._handle_status)
        self.video_thread.error_occurred.connect(self._handle_error)

        self.video_thread.liveness_status.connect(self._handle_liveness_status)

        self.video_thread.start()

    @Slot(object)
    def _update_frame(self, qt_image):

        pixmap = QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled)

    @Slot(bool, float, float)
    def _handle_auth_result(self, authenticated, euclidean, cosine):

        self.label_euclidean.setText(f"Euclidean Distance:  {euclidean:.4f}")
        self.label_cosine.setText(f"Cosine Similarity:   {cosine:.4f}")

        if authenticated:
            self.result_label.setText("✓  ACCESS GRANTED")
            self.result_label.setStyleSheet(
                "background-color: #1a472a; border-radius: 6px; "
                "color: #51cf66; padding: 5px;"
            )
            self.label_liveness.setText("Anti-Spoofing: Verified Live Person")
            self.label_liveness.setStyleSheet("color: #a6e3a1;") # Green
        else:
            self.result_label.setText("✗  ACCESS DENIED")
            self.result_label.setStyleSheet(
                "background-color: #4a1a1a; border-radius: 6px; "
                "color: #ff6b6b; padding: 5px;"
            )

    @Slot(str)
    def _handle_status(self, message):

        self.status_bar.showMessage(message)

        if "registered" in message.lower() or "exists" in message.lower():
            self._set_buttons_idle()
            QMessageBox.information(self, "Registration Subsystem", message)

    @Slot(str)
    def _handle_liveness_status(self, message):

        self.label_liveness.setText(f"Anti-Spoofing: {message}")
        self.label_liveness.setStyleSheet("color: #fab387;") # Warning Orange

    @Slot(str)
    def _handle_error(self, message):

        QMessageBox.critical(self, "Critical Error", message)

    @Slot()
    def _on_register_clicked(self):

        username = self.name_input.text().strip()

        if not username:
            QMessageBox.warning(
                self, "Validation Warning", "Username string input field is empty!"
            )
            return

        if self.db.user_exists(username):
            reply = QMessageBox.question(
                self, "User Already Exists",
                f"User identity profile '{username}' is already registered.\n"
                "Overwrite existing reference template?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
            self.db.delete_user(username)

        self._reset_result_panel()
        self._set_buttons_busy()

        self.video_thread.start_registration(username)
        self.status_bar.showMessage(
            f"Registering '{username}': Please look directly into the camera lens..."
        )

    @Slot()
    def _on_auth_clicked(self):

        username = self.name_input.text().strip()

        if not username:
            QMessageBox.warning(
                self, "Validation Warning", "Username string input field is empty!"
            )
            return

        if not self.db.user_exists(username):
            QMessageBox.warning(
                self, "Authentication Error",
                f"Identity record matching '{username}' not found in the local schema.\n"
                "Please perform profile registration first."
            )
            return

        self._reset_result_panel()
        self._set_buttons_busy()

        self.video_thread.start_auth(username)
        self.status_bar.showMessage(f"Authenticating '{username}' profile structure...")

    @Slot()
    def _on_cancel_clicked(self):

        if self.video_thread:
            self.video_thread.set_idle()
        self._set_buttons_idle()
        self._reset_result_panel()
        self.status_bar.showMessage("Operation cancelled by user")

    @Slot(int)
    def _on_eu_changed(self, value):

        threshold = value / 100.0
        self.lbl_eu.setText(f"Euclidean Distance Cap: {threshold:.2f}")
        if self.video_thread:
            self.video_thread.update_thresholds(
                euclidean=threshold,
                cosine=self.slider_cos.value() / 100.0
            )

    @Slot(int)
    def _on_cos_changed(self, value):

        threshold = value / 100.0
        self.lbl_cos.setText(f"Cosine Similarity Floor: {threshold:.2f}")
        if self.video_thread:
            self.video_thread.update_thresholds(
                euclidean=self.slider_eu.value() / 100.0,
                cosine=threshold
            )

    def _set_buttons_busy(self):

        self.btn_register.setEnabled(False)
        self.btn_auth.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.name_input.setEnabled(False)

    def _set_buttons_idle(self):

        self.btn_register.setEnabled(True)
        self.btn_auth.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.name_input.setEnabled(True)

    def _reset_result_panel(self):

        self.result_label.setText("—")
        self.result_label.setStyleSheet(
            "background-color: #2d2d2d; border-radius: 6px; color: #aaaaaa;"
        )
        self.label_liveness.setText("Anti-Spoofing: Idle")
        self.label_liveness.setStyleSheet("color: #f9e2af;")
        self.label_euclidean.setText("Euclidean Distance: —")
        self.label_cosine.setText("Cosine Similarity: —")

    def _apply_styles(self):

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: Arial;
                font-size: 10pt;
            }
            QGroupBox {
                border: 1px solid #45475a;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
                font-weight: bold;
                color: #89b4fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLineEdit {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 5px 8px;
                color: #cdd6f4;
            }
            QLineEdit:focus {
                border: 1px solid #89b4fa;
            }
            QPushButton {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 5px;
                padding: 6px 12px;
                color: #cdd6f4;
            }
            QPushButton:hover {
                background-color: #45475a;
                border: 1px solid #89b4fa;
            }
            QPushButton:pressed {
                background-color: #89b4fa;
                color: #1e1e2e;
            }
            QPushButton:disabled {
                background-color: #1e1e2e;
                color: #585b70;
                border: 1px solid #313244;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #45475a;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #89b4fa;
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: #89b4fa;
                border-radius: 3px;
            }
            QStatusBar {
                background-color: #181825;
                color: #6c7086;
                border-top: 1px solid #313244;
            }
        """)

    def closeEvent(self, event):

        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
        self.db.close()
        event.accept()
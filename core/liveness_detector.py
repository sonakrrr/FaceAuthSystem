import math
from collections import deque

class LivenessDetector:

    LEFT_EYE_TOP_1    = 386
    LEFT_EYE_BOTTOM_1 = 374
    LEFT_EYE_LEFT     = 362
    LEFT_EYE_RIGHT    = 263

    RIGHT_EYE_TOP_1    = 159
    RIGHT_EYE_BOTTOM_1 = 145
    RIGHT_EYE_LEFT     = 33
    RIGHT_EYE_RIGHT    = 133

    NOSE_TIP = 1
    LEFT_CHEEK = 234
    RIGHT_CHEEK = 454

    CHALLENGE_SEQUENCE = ['blink', 'turn_left', 'turn_right', 'blink']

    def __init__(self, ear_threshold=0.23, yaw_left_threshold=0.92, yaw_right_threshold=1.10):
        self.ear_threshold = ear_threshold
        self.yaw_left_threshold = yaw_left_threshold
        self.yaw_right_threshold = yaw_right_threshold

        self._current_step_index = 0
        self._eye_was_closed = False

        self.is_live = False

    def check(self, landmarks):

        if landmarks is None:
            return self._empty_result()

        if self.is_live:
            return self._success_result()

        ear = self._calculate_ear(landmarks)
        yaw_ratio = self._calculate_head_yaw(landmarks)

        current_challenge = self.CHALLENGE_SEQUENCE[self._current_step_index]
        instruction = "Processing..."
        step_passed = False

        if current_challenge == 'blink':
            instruction = f"Step {self._current_step_index + 1}/{len(self.CHALLENGE_SEQUENCE)}: Please BLINK"
            if ear < self.ear_threshold:
                self._eye_was_closed = True
            elif self._eye_was_closed and ear > self.ear_threshold:
                step_passed = True
                self._eye_was_closed = False

        elif current_challenge == 'turn_left':
            instruction = f"Step {self._current_step_index + 1}/{len(self.CHALLENGE_SEQUENCE)}: Turn head SLIGHTLY LEFT"
            if yaw_ratio < self.yaw_left_threshold:
                step_passed = True

        elif current_challenge == 'turn_right':
            instruction = f"Step {self._current_step_index + 1}/{len(self.CHALLENGE_SEQUENCE)}: Turn head SLIGHTLY RIGHT"
            if yaw_ratio > self.yaw_right_threshold:
                step_passed = True

        if step_passed:
            self._current_step_index += 1

            if self._current_step_index >= len(self.CHALLENGE_SEQUENCE):
                self.is_live = True
                instruction = "Live Person Verified!"

        return {
            'is_live': self.is_live,
            'instruction': instruction,
            'progress': f"{self._current_step_index}/{len(self.CHALLENGE_SEQUENCE)}",
            'ear': round(ear, 3),
            'yaw_ratio': round(yaw_ratio, 3)
        }

    def reset(self):

        self._current_step_index = 0
        self._eye_was_closed = False
        self.is_live = False

    def _calculate_ear(self, landmarks):

        def eye_ear(top, bot, left, right):
            p_top, p_bot = landmarks[top], landmarks[bot]
            p_left, p_right = landmarks[left], landmarks[right]
            vert = self._dist(p_top, p_bot)
            horiz = self._dist(p_left, p_right)
            return vert / horiz if horiz > 1e-6 else 0.3

        ear_left = eye_ear(self.LEFT_EYE_TOP_1, self.LEFT_EYE_BOTTOM_1, self.LEFT_EYE_LEFT, self.LEFT_EYE_RIGHT)
        ear_right = eye_ear(self.RIGHT_EYE_TOP_1, self.RIGHT_EYE_BOTTOM_1, self.RIGHT_EYE_LEFT, self.RIGHT_EYE_RIGHT)
        return (ear_left + ear_right) / 2.0

    def _calculate_head_yaw(self, landmarks):

        nose = landmarks[self.NOSE_TIP]
        left_cheek = landmarks[self.LEFT_CHEEK]
        right_cheek = landmarks[self.RIGHT_CHEEK]

        dist_left = self._dist(nose, left_cheek)
        dist_right = self._dist(nose, right_cheek)

        if dist_right < 1e-6: return 1.0
        return dist_left / dist_right

    def _dist(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

    def _empty_result(self):
        return {
            'is_live': False,
            'instruction': "Face not detected",
            'progress': f"{self._current_step_index}/{len(self.CHALLENGE_SEQUENCE)}",
            'ear': 0.0,
            'yaw_ratio': 1.0
        }

    def _success_result(self):
        return {
            'is_live': True,
            'instruction': "Live Person Verified!",
            'progress': f"{len(self.CHALLENGE_SEQUENCE)}/{len(self.CHALLENGE_SEQUENCE)}",
            'ear': 0.0,
            'yaw_ratio': 1.0
        }
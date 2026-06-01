import cv2
import numpy as np

class TextureLivenessDetector:

    def __init__(self, laplacian_threshold=35.0):
        self.laplacian_threshold = laplacian_threshold

    def check_texture(self, frame, landmarks):

        if frame is None or landmarks is None:
            return {
                'is_spoof': False, 'laplacian_var': 0.0,
                'mean_cr': 150.0, 'mean_cb': 100.0,
                'is_color_anomaly': False, 'is_blur_anomaly': False
            }

        h, w, _ = frame.shape
        xs = [int(pt[0] * w) for pt in landmarks]
        ys = [int(pt[1] * h) for pt in landmarks]

        xmin, xmax = max(0, min(xs)), min(w, max(xs))
        ymin, ymax = max(0, min(ys)), min(h, max(ys))

        if (xmax - xmin) < 50 or (ymax - ymin) < 50:
            return {
                'is_spoof': False, 'laplacian_var': 0.0,
                'mean_cr': 150.0, 'mean_cb': 100.0,
                'is_color_anomaly': False, 'is_blur_anomaly': False
            }

        face_crop = frame[ymin:ymax, xmin:xmax]

        gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_crop, cv2.CV_64F).var()

        ycrcb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YCrCb)
        _, cr, cb = cv2.split(ycrcb_crop)
        mean_cr = np.mean(cr)
        mean_cb = np.mean(cb)

        is_color_anomaly = not (125 <= mean_cr <= 185 and 70 <= mean_cb <= 135)
        is_blur_anomaly = laplacian_var < self.self_calibrated_threshold(gray_crop)

        is_spoof = is_blur_anomaly or is_color_anomaly

        return {
            'is_spoof': is_spoof,
            'laplacian_var': round(laplacian_var, 1),
            'mean_cr': round(mean_cr, 1),
            'mean_cb': round(mean_cb, 1),
            'is_color_anomaly': is_color_anomaly,
            'is_blur_anomaly': is_blur_anomaly
        }

    def self_calibrated_threshold(self, gray_img):

        brightness = np.mean(gray_img)
        if brightness < 60:
            return max(15.0, self.laplacian_threshold - 15)
        return self.laplacian_threshold
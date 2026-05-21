import cv2
import numpy as np

RECT_X_MM = 130.0
RECT_Y_MM = 167.5

ROI_LEFT = 40
ROI_RIGHT = 540
ROI_TOP = 35
ROI_BOTTOM = 400

HSV_RANGES = {
    "red1": ([0, 60, 50], [12, 255, 255]),
    "red2": ([168, 60, 50], [180, 255, 255]),
    "green": ([35, 40, 40], [90, 255, 255]),
    "blue": ([85, 40, 40], [140, 255, 255]),
    "yellow": ([15, 60, 60], [40, 255, 255])
}

MIN_AREA = 400
MAX_AREA = 20000


def get_color_mask(hsv, color):
    if color == "red":
        lower1 = np.array(HSV_RANGES["red1"][0], dtype=np.uint8)
        upper1 = np.array(HSV_RANGES["red1"][1], dtype=np.uint8)
        lower2 = np.array(HSV_RANGES["red2"][0], dtype=np.uint8)
        upper2 = np.array(HSV_RANGES["red2"][1], dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)

        return cv2.bitwise_or(mask1, mask2)

    lower = np.array(HSV_RANGES[color][0], dtype=np.uint8)
    upper = np.array(HSV_RANGES[color][1], dtype=np.uint8)

    return cv2.inRange(hsv, lower, upper)


def pixel_to_local_mm(cx, cy):
    x_mm = ((cx - ROI_LEFT) / (ROI_RIGHT - ROI_LEFT)) * RECT_X_MM - RECT_X_MM / 2
    y_mm = RECT_Y_MM / 2 - ((cy - ROI_TOP) / (ROI_BOTTOM - ROI_TOP)) * RECT_Y_MM

    return x_mm, y_mm


def detect_cube(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    roi_mask[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT] = 255

    best_result = None
    best_area = 0

    for color in ["red", "green", "blue", "yellow"]:
        mask = get_color_mask(hsv, color)
        mask = cv2.bitwise_and(mask, roi_mask)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < MIN_AREA or area > MAX_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            if w < 15 or h < 15:
                continue

            ratio = w / h

            if ratio < 0.4 or ratio > 2.5:
                continue

            M = cv2.moments(contour)

            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            if area > best_area:
                x_mm, y_mm = pixel_to_local_mm(cx, cy)

                best_result = {
                    "color": color,
                    "x_mm": x_mm,
                    "y_mm": y_mm,
                    "cx": cx,
                    "cy": cy,
                    "area": area,
                    "bbox": [x, y, w, h]
                }

                best_area = area

    return best_result


def detect_object(frame):
    result = detect_cube(frame)

    if result is None:
        return None

    return result["x_mm"], result["y_mm"]


def detect_color(frame):
    result = detect_cube(frame)

    if result is None:
        return None

    return result["color"]
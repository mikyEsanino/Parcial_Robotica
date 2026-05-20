import cv2
import numpy as np

RECT_X_MM = 130.0
RECT_Y_MM = 167.5

MANUAL_BOARD_POINTS = None

HSV_RANGES = {
    "red1": ([0, 80, 80], [10, 255, 255]),
    "red2": ([170, 80, 80], [180, 255, 255]),
    "green": ([35, 50, 50], [85, 255, 255]),
    "blue": ([90, 50, 50], [130, 255, 255]),
    "yellow": ([20, 80, 80], [35, 255, 255])
}


def order_points(points):
    points = np.array(points, dtype=np.float32)

    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)

    top_left = points[np.argmin(s)]
    bottom_right = points[np.argmax(s)]
    top_right = points[np.argmin(diff)]
    bottom_left = points[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def find_board_points(frame):
    if MANUAL_BOARD_POINTS is not None:
        return order_points(MANUAL_BOARD_POINTS)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 1000:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * peri, True)

        if len(approx) == 4:
            points = approx.reshape(4, 2)
            return order_points(points)

    largest = contours[0]
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)

    return order_points(box)


def get_homography(board_points):
    src = np.array(board_points, dtype=np.float32)

    dst = np.array([
        [-RECT_X_MM / 2, RECT_Y_MM / 2],
        [RECT_X_MM / 2, RECT_Y_MM / 2],
        [RECT_X_MM / 2, -RECT_Y_MM / 2],
        [-RECT_X_MM / 2, -RECT_Y_MM / 2]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src, dst)

    return H


def pixel_to_mm(cx, cy, H):
    point = np.array([[[cx, cy]]], dtype=np.float32)
    result = cv2.perspectiveTransform(point, H)

    x_mm = float(result[0][0][0])
    y_mm = float(result[0][0][1])

    return x_mm, y_mm


def create_board_mask(frame, board_points):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    points = np.array(board_points, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)

    return mask


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


def detect_cube(frame):
    board_points = find_board_points(frame)

    if board_points is None:
        return None

    H = get_homography(board_points)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    board_mask = create_board_mask(frame, board_points)

    best_result = None
    best_area = 0

    for color in ["red", "green", "blue", "yellow"]:
        mask = get_color_mask(hsv, color)
        mask = cv2.bitwise_and(mask, board_mask)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < 300:
                continue

            if area > best_area:
                M = cv2.moments(contour)

                if M["m00"] == 0:
                    continue

                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                x_mm, y_mm = pixel_to_mm(cx, cy, H)

                best_result = {
                    "color": color,
                    "x_mm": x_mm,
                    "y_mm": y_mm,
                    "cx": cx,
                    "cy": cy,
                    "area": area,
                    "board_points": board_points
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


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        exit()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("No se pudo leer la cámara")
            break

        result = detect_cube(frame)

        if result is not None:
            cx = result["cx"]
            cy = result["cy"]
            color = result["color"]
            x_mm = result["x_mm"]
            y_mm = result["y_mm"]
            board_points = result["board_points"].astype(int)

            cv2.polylines(frame, [board_points], True, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)

            text = f"{color} x={x_mm:.1f} y={y_mm:.1f}"
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            print(text)

        cv2.imshow("vision", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
vision.py
Mostrando vision.py.
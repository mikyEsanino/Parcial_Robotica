from pymycobot.mycobot import MyCobot
from vision import detect_cube
import cv2
import time
import numpy as np

PORT = "/dev/ttyUSB0"
BAUD = 1000000

SPEED = 20
GRIPPER_SPEED = 80

INIT_ANGLES = [1.58, -0.35, -0.96, -1.66, 0.43, -44.12]
WATCH_ANGLES = [9.05, 3.42, -1.4, -73.21, 2.1, -30.23]

PLACE_BY_COLOR = {
    "yellow": [149.4, 216.4],
    "red": [78.0, 214.2],
    "green": [15.9, 217.9],
    "blue": [-47.5, 217.6]
}

Z_UP = 170
Z_PICK = 108
Z_PLACE = 120

RX = -177
RY = -5
RZ = 43

PICK_OFFSET_X_ROBOT = 5.0
PICK_OFFSET_Y_ROBOT = 20.0

DOUBLE_NEGATIVE_OFFSET_X = 8.0
DOUBLE_NEGATIVE_OFFSET_Y = 5.0

calib_pixels = np.array([
    [109, 120],
    [464, 54],
    [85, 366],
    [369, 248]
], dtype=float)

calib_robots = np.array([
    [244.1, 34.6],
    [256.4, -56.1],
    [169.1, 55.0],
    [201.4, -27.5]
], dtype=float)

A = np.column_stack([
    calib_pixels[:, 0],
    calib_pixels[:, 1],
    np.ones(len(calib_pixels))
])

coef_x, _, _, _ = np.linalg.lstsq(A, calib_robots[:, 0], rcond=None)
coef_y, _, _, _ = np.linalg.lstsq(A, calib_robots[:, 1], rcond=None)

mc = MyCobot(PORT, BAUD)
mc.power_on()
time.sleep(1)


def pixel_to_robot(cx, cy):
    x_robot = coef_x[0] * cx + coef_x[1] * cy + coef_x[2]
    y_robot = coef_y[0] * cx + coef_y[1] * cy + coef_y[2]

    return x_robot, y_robot


def make_coords(x, y, z):
    return [x, y, z, RX, RY, RZ]


def goto_pose(pose, mode="coords", seconds=3):
    if mode == "angles":
        mc.send_angles(pose, SPEED)
    else:
        mc.send_coords(pose, SPEED, 1)

    time.sleep(seconds)


def open_gripper():
    mc.set_gripper_value(100, GRIPPER_SPEED)
    time.sleep(1)


def close_gripper():
    mc.set_gripper_value(0, GRIPPER_SPEED)
    time.sleep(1)


def init_pose():
    goto_pose(INIT_ANGLES, mode="angles")


def watch_pose():
    goto_pose(WATCH_ANGLES, mode="angles")


def pick(x_pick, y_pick):
    goto_pose(make_coords(x_pick, y_pick, Z_UP))
    goto_pose(make_coords(x_pick, y_pick, Z_PICK))
    close_gripper()
    time.sleep(0.8)
    goto_pose(make_coords(x_pick, y_pick, Z_UP))


def place(x_place, y_place):
    goto_pose(make_coords(x_place, y_place, Z_UP))
    goto_pose(make_coords(x_place, y_place, Z_PLACE))
    open_gripper()
    time.sleep(0.8)
    goto_pose(make_coords(x_place, y_place, Z_UP))


def get_place_by_color(color):
    color = color.strip().lower()

    if color == "yellow":
        return 149.4, 216.4

    elif color == "red":
        return 78.0, 214.2

    elif color == "green":
        return 15.9, 217.9

    elif color == "blue":
        return -47.5, 217.6

    else:
        return None


def vision_result_to_pick_robot(result):
    cx = result["cx"]
    cy = result["cy"]
    x_mm = result["x_mm"]
    y_mm = result["y_mm"]

    x_robot, y_robot = pixel_to_robot(cx, cy)

    offset_x = 0.0
    offset_y = 0.0

    if x_mm < 0 and y_mm < 0:
        offset_x = DOUBLE_NEGATIVE_OFFSET_X
        offset_y = DOUBLE_NEGATIVE_OFFSET_Y
        print("Zona doble negativa: compensación especial")

    elif y_mm < 0:
        offset_x = PICK_OFFSET_X_ROBOT
        offset_y = PICK_OFFSET_Y_ROBOT
        print("Zona Y negativa")

    elif x_mm > 0 and y_mm > 0:
        offset_x = PICK_OFFSET_X_ROBOT
        offset_y = PICK_OFFSET_Y_ROBOT
        print("Zona X positiva e Y positiva")

    x_robot -= offset_x
    y_robot -= offset_y

    print("Pixel:", cx, cy)
    print("Vision mm:", x_mm, y_mm)
    print("Offset aplicado:", offset_x, offset_y)
    print("Pick robot:", x_robot, y_robot)

    return x_robot, y_robot


def capturar_resultado_fresco():
    watch_pose()
    time.sleep(2)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No se pudo abrir cámara")
        return None

    best = None
    best_area = 0

    for i in range(30):
        ret, frame = cap.read()
        time.sleep(0.05)

        if not ret:
            continue

        result = detect_cube(frame)

        if result is not None and result["area"] > best_area:
            best = result
            best_area = result["area"]

    cap.release()

    print("MEJOR DETECCIÓN:", best)
    return best


def run_cycle(cycle=1):
    result = capturar_resultado_fresco()

    if result is None:
        print("No se detectó cubo")
        return False

    color = result["color"].strip().lower()

    destino = get_place_by_color(color)

    if destino is None:
        print("Color no reconocido:", color)
        return False

    x_place, y_place = destino
    x_pick, y_pick = vision_result_to_pick_robot(result)

    print("========== DEBUG CICLO ==========")
    print("Ciclo:", cycle)
    print("Color detectado:", color)
    print("Pick calculado:", x_pick, y_pick)
    print("Destino elegido:", x_place, y_place)
    print("=================================")

    open_gripper()
    pick(x_pick, y_pick)

    print("DEPOSITANDO EN COLOR:", color)
    print("COORDENADAS DESTINO:", x_place, y_place)

    place(x_place, y_place)
    init_pose()

    return True


def run_cycle_vision(cycle=1):
    return run_cycle(cycle)

CONTROL_LOG = []
POSE_TABLE = []

try:
    from cinematica import ik_solve
except Exception:
    ik_solve = None


def log_event(cycle, phase, status, cause="-", elapsed=0.0):
    row = {
        "cycle": cycle,
        "phase": phase,
        "status": status,
        "cause": cause,
        "elapsed_seconds": round(elapsed, 2)
    }

    CONTROL_LOG.append(row)

    print("[LOG]", row)


def record_pose(cycle, pose_name):
    angles = mc.get_angles()
    coords = mc.get_coords()

    row = {
        "cycle": cycle,
        "pose": pose_name,
        "angles": angles,
        "coords": coords
    }

    POSE_TABLE.append(row)

    print("[POSE]", row)


def execute_phase(cycle, phase, action, attempts=2):
    last_error = "-"

    for attempt in range(1, attempts + 1):
        start = time.time()

        try:
            action()
            elapsed = time.time() - start
            log_event(cycle, phase, "OK", f"intento {attempt}", elapsed)
            return True

        except Exception as e:
            elapsed = time.time() - start
            last_error = str(e)
            log_event(cycle, phase, "FALLO", f"intento {attempt}: {last_error}", elapsed)
            time.sleep(1)

    return False


def calculate_ik_if_available(x, y, z):
    if ik_solve is None:
        return None

    try:
        return ik_solve(x, y, z)
    except Exception as e:
        print("No se pudo calcular IK:", e)
        return None


def pick_logged(cycle, x_pick, y_pick):
    pick_above = make_coords(x_pick, y_pick, Z_UP)
    pick_down = make_coords(x_pick, y_pick, Z_PICK)

    ik_angles = calculate_ik_if_available(x_pick, y_pick, Z_PICK)
    print("IK pick:", ik_angles)

    if not execute_phase(cycle, "pick_above", lambda: goto_pose(pick_above)):
        return False
    record_pose(cycle, "pick_above")

    if not execute_phase(cycle, "pick_down", lambda: goto_pose(pick_down)):
        return False
    record_pose(cycle, "pick_down")

    if not execute_phase(cycle, "close_gripper", close_gripper):
        return False

    if not execute_phase(cycle, "lift_after_pick", lambda: goto_pose(pick_above)):
        return False
    record_pose(cycle, "lift_after_pick")

    return True


def place_logged(cycle, x_place, y_place, color):
    place_above = make_coords(x_place, y_place, Z_UP)
    place_down = make_coords(x_place, y_place, Z_PLACE)

    if not execute_phase(cycle, f"place_above_{color}", lambda: goto_pose(place_above)):
        return False
    record_pose(cycle, f"place_above_{color}")

    if not execute_phase(cycle, f"place_down_{color}", lambda: goto_pose(place_down)):
        return False
    record_pose(cycle, f"place_down_{color}")

    if not execute_phase(cycle, "open_gripper_place", open_gripper):
        return False

    if not execute_phase(cycle, f"lift_after_place_{color}", lambda: goto_pose(place_above)):
        return False
    record_pose(cycle, f"lift_after_place_{color}")

    return True


def run_cycle(cycle=1):
    cycle_start = time.time()

    result = None

    if not execute_phase(cycle, "detect_object", lambda: globals().update(result_detection=capturar_resultado_fresco())):
        return False

    result = globals().get("result_detection")

    if result is None:
        log_event(cycle, "detect_object", "FALLO", "No se detectó cubo", 0)
        return False

    color = result["color"].strip().lower()
    destino = get_place_by_color(color)

    if destino is None:
        log_event(cycle, "validate_color", "FALLO", f"Color no reconocido: {color}", 0)
        return False

    x_place, y_place = destino
    x_pick, y_pick = vision_result_to_pick_robot(result)

    print("========== CICLO CONTROL ==========")
    print("Ciclo:", cycle)
    print("Color detectado:", color)
    print("Pick calculado:", x_pick, y_pick)
    print("Destino elegido:", x_place, y_place)
    print("===================================")

    if not execute_phase(cycle, "init_pose", init_pose):
        return False
    record_pose(cycle, "init_pose")

    if not execute_phase(cycle, "open_gripper_start", open_gripper):
        return False

    if not execute_phase(cycle, "watch_pose", watch_pose):
        return False
    record_pose(cycle, "watch_pose")

    if not pick_logged(cycle, x_pick, y_pick):
        return False

    if not place_logged(cycle, x_place, y_place, color):
        return False

    if not execute_phase(cycle, "return_init", init_pose):
        return False
    record_pose(cycle, "return_init")

    total_elapsed = time.time() - cycle_start
    log_event(cycle, "cycle_end", "OK", "ciclo completado", total_elapsed)

    return True


def run_5_cycles():
    success = 0
    total = 5

    for cycle in range(1, total + 1):
        print("\n==============================")
        print("EJECUTANDO CICLO", cycle)
        print("==============================")

        result = run_cycle(cycle)

        if result:
            success += 1
            print("Ciclo", cycle, "OK")
        else:
            print("Ciclo", cycle, "FALLÓ")

        time.sleep(1)

    tasa = (success / total) * 100

    print("\n========== RESUMEN ==========")
    print("Éxitos:", success, "/", total)
    print("Tasa de éxito:", round(tasa, 2), "%")
    print("=============================")

    return tasa


def show_control_log():
    try:
        import pandas as pd
        return pd.DataFrame(CONTROL_LOG)
    except Exception:
        return CONTROL_LOG


def show_pose_table():
    try:
        import pandas as pd
        return pd.DataFrame(POSE_TABLE)
    except Exception:
        return POSE_TABLE


if __name__ == "__main__":
    print("1. Init pose")
    print("2. Watch pose")
    print("3. Run cycle")

    option = input("Opción: ")

    if option == "1":
        init_pose()

    elif option == "2":
        watch_pose()

    elif option == "3":
        run_cycle(1)
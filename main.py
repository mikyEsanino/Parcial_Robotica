import time
import csv
import traceback
from datetime import datetime

import cv2

import control
from vision import detect_cube

try:
    from cinematica import ik_solve
except Exception as e:
    ik_solve = None
    IK_IMPORT_ERROR = str(e)


LOG_FILE = "pipeline_log.csv"
TOTAL_CYCLES = 5


def log_event(cycle, state, status, message, data=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = [
        timestamp,
        cycle,
        state,
        status,
        message,
        str(data)
    ]

    print(row)

    try:
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        print("No se pudo escribir log:", e)


def capture_detection(cycle):
    control.watch_pose()
    time.sleep(2)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        log_event(cycle, "DETECTANDO", "FALLO", "No se pudo abrir la cámara")
        return None

    best = None
    best_area = 0

    for _ in range(30):
        ret, frame = cap.read()
        time.sleep(0.05)

        if not ret:
            continue

        result = detect_cube(frame)

        if result is not None and result["area"] > best_area:
            best = result
            best_area = result["area"]

    cap.release()

    if best is None:
        log_event(cycle, "DETECTANDO", "FALLO", "No se detectó cubo")
        return None

    log_event(cycle, "DETECTANDO", "OK", "Cubo detectado", best)
    return best


def calc_ik(cycle, x_pick, y_pick):
    if ik_solve is None:
        log_event(cycle, "CALC_IK", "WARN", "No se importó ik_solve, se usará movimiento por coordenadas")
        return None

    try:
        ik_angles = ik_solve(x_pick, y_pick, control.Z_PICK)
        log_event(cycle, "CALC_IK", "OK", "IK calculada", ik_angles)
        return ik_angles

    except Exception as e:
        log_event(cycle, "CALC_IK", "FALLO", str(e))
        return None


def run_one_cycle(cycle):
    try:
        log_event(cycle, "IDLE", "OK", "Inicio de ciclo")

        result = capture_detection(cycle)

        if result is None:
            return False

        color = result["color"].strip().lower()
        destino = control.get_place_by_color(color)

        if destino is None:
            log_event(cycle, "DETECTANDO", "FALLO", "Color no reconocido", color)
            return False

        x_place, y_place = destino
        x_pick, y_pick = control.vision_result_to_pick_robot(result)

        log_event(cycle, "CALC_IK", "INFO", "Calculando IK para pick", {
            "x_pick": x_pick,
            "y_pick": y_pick,
            "z_pick": control.Z_PICK
        })

        ik_angles = calc_ik(cycle, x_pick, y_pick)

        log_event(cycle, "AGARRANDO", "INFO", "Ejecutando pick", {
            "x_pick": x_pick,
            "y_pick": y_pick,
            "ik": ik_angles
        })

        control.open_gripper()
        control.pick(x_pick, y_pick)

        log_event(cycle, "DEPOSITAR", "INFO", "Ejecutando place", {
            "color": color,
            "x_place": x_place,
            "y_place": y_place
        })

        control.place(x_place, y_place)

        log_event(cycle, "RETURN_INIT", "INFO", "Volviendo a init_pose")
        control.init_pose()

        angles = control.mc.get_angles()
        coords = control.mc.get_coords()

        log_event(cycle, "FIN", "OK", "Ciclo completado", {
            "angles": angles,
            "coords": coords
        })

        return True

    except Exception as e:
        log_event(cycle, "ERROR", "FALLO", str(e), traceback.format_exc())

        try:
            control.init_pose()
        except Exception:
            pass

        return False


def run_pipeline(total_cycles=TOTAL_CYCLES):
    success = 0

    log_event(0, "STARTUP", "INFO", "Iniciando pipeline end-to-end")

    try:
        control.init_pose()
        log_event(0, "STARTUP", "OK", "Robot en init_pose")
    except Exception as e:
        log_event(0, "STARTUP", "FALLO", str(e))
        return False

    for cycle in range(1, total_cycles + 1):
        print("\n==============================")
        print("CICLO", cycle)
        print("==============================")

        ok = run_one_cycle(cycle)

        if ok:
            success += 1

        time.sleep(1)

    success_rate = (success / total_cycles) * 100

    log_event(0, "SUMMARY", "OK", "Resumen de sesión", {
        "success": success,
        "total": total_cycles,
        "success_rate": success_rate
    })

    print("\n========== RESUMEN ==========")
    print("Ciclos exitosos:", success, "/", total_cycles)
    print("Tasa de éxito:", round(success_rate, 2), "%")
    print("=============================")

    return success_rate >= 80


if __name__ == "__main__":
    run_pipeline(5)
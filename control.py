from pymycobot.mycobot import MyCobot
import time
import csv
from datetime import datetime

PORT = "/dev/ttyUSB0"
BAUD = 1000000

SPEED = 20
GRIPPER_SPEED = 80

INIT_ANGLES = [1.58, -0.35, -0.96, -1.66, 0.43, -44.12]
WATCH_ANGLES = [9.05, 3.42, -1.4, -73.21, 2.1, -30.23]

CENTER_ROBOT = [188.8, -17.0]
POINT_X_ROBOT = [228.4, -26.7]
POINT_Y_ROBOT = [206.1, 50.2]

CALIBRATION_DISTANCE = 50.0

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

LOG_FILE = "log_control.csv"

mc = None


def connect_robot():
    global mc

    if mc is None:
        mc = MyCobot(PORT, BAUD)
        mc.power_on()
        time.sleep(1)

    return mc


def save_log(cycle, phase, elapsed, status, cause):
    exists = False

    try:
        with open(LOG_FILE, "r"):
            exists = True
    except FileNotFoundError:
        exists = False

    with open(LOG_FILE, "a", newline="") as file:
        writer = csv.writer(file)

        if not exists:
            writer.writerow(["fecha_hora", "ciclo", "fase", "tiempo_segundos", "estado", "causa"])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            cycle,
            phase,
            round(elapsed, 2),
            status,
            cause
        ])


def execute_phase(cycle, phase, action):
    start = time.time()

    try:
        action()
        end = time.time()
        save_log(cycle, phase, end - start, "OK", "-")
        return True
    except Exception as e:
        end = time.time()
        save_log(cycle, phase, end - start, "FALLO", str(e))
        return False


def local_to_robot(x_local, y_local):
    vx_x = (POINT_X_ROBOT[0] - CENTER_ROBOT[0]) / CALIBRATION_DISTANCE
    vx_y = (POINT_X_ROBOT[1] - CENTER_ROBOT[1]) / CALIBRATION_DISTANCE

    vy_x = (POINT_Y_ROBOT[0] - CENTER_ROBOT[0]) / CALIBRATION_DISTANCE
    vy_y = (POINT_Y_ROBOT[1] - CENTER_ROBOT[1]) / CALIBRATION_DISTANCE

    x_robot = CENTER_ROBOT[0] + x_local * vx_x + y_local * vy_x
    y_robot = CENTER_ROBOT[1] + x_local * vx_y + y_local * vy_y

    return x_robot, y_robot


def make_coords(x, y, z):
    return [x, y, z, RX, RY, RZ]


def goto_pose(pose, mode="coords", seconds=3):
    robot = connect_robot()

    if mode == "angles":
        robot.send_angles(pose, SPEED)
    else:
        robot.send_coords(pose, SPEED, 1)

    time.sleep(seconds)


def open_gripper():
    robot = connect_robot()
    robot.set_gripper_value(100, GRIPPER_SPEED)
    time.sleep(1)


def close_gripper():
    robot = connect_robot()
    robot.set_gripper_value(0, GRIPPER_SPEED)
    time.sleep(1)


def init_pose():
    goto_pose(INIT_ANGLES, mode="angles")


def watch_pose():
    goto_pose(WATCH_ANGLES, mode="angles")


def pick(x_pick, y_pick):
    goto_pose(make_coords(x_pick, y_pick, Z_UP))
    goto_pose(make_coords(x_pick, y_pick, Z_PICK))
    close_gripper()
    goto_pose(make_coords(x_pick, y_pick, Z_UP))


def place(x_place, y_place):
    goto_pose(make_coords(x_place, y_place, Z_UP))
    goto_pose(make_coords(x_place, y_place, Z_PLACE))
    open_gripper()
    goto_pose(make_coords(x_place, y_place, Z_UP))


def run_cycle(cycle, x_local, y_local, color):
    if color not in PLACE_BY_COLOR:
        save_log(cycle, "validar_color", 0, "FALLO", "color no reconocido")
        return False

    x_pick, y_pick = local_to_robot(x_local, y_local)
    x_place, y_place = PLACE_BY_COLOR[color]

    if not execute_phase(cycle, "init_pose", init_pose):
        return False

    if not execute_phase(cycle, "open_gripper", open_gripper):
        return False

    if not execute_phase(cycle, "watch_pose", watch_pose):
        return False

    if not execute_phase(cycle, "pick", lambda: pick(x_pick, y_pick)):
        return False

    if not execute_phase(cycle, "place", lambda: place(x_place, y_place)):
        return False

    if not execute_phase(cycle, "return_init", init_pose):
        return False

    return True


if __name__ == "__main__":
    connect_robot()

    print("1. Probar init_pose")
    print("2. Probar watch_pose")
    print("3. Probar ciclo manual")

    option = input("Opción: ")

    if option == "1":
        init_pose()

    elif option == "2":
        watch_pose()

    elif option == "3":
        x_local = float(input("x_local mm: "))
        y_local = float(input("y_local mm: "))
        color = input("color red/green/blue/yellow: ").strip().lower()

        result = run_cycle(1, x_local, y_local, color)

        if result:
            print("Ciclo OK")
        else:
            print("Ciclo falló")
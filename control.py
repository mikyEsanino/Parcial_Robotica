from pymycobot.mycobot import MyCobot
import time

mc = MyCobot("/dev/ttyUSB0", 1000000)
mc.power_on()
time.sleep(1)

while True:
    opcion = input("Presiona ENTER para leer coordenadas o escribe salir: ")

    if opcion.lower() == "salir":
        break

    print("Ángulos:", mc.get_angles())
    print("Coordenadas:", mc.get_coords())
    print()

from pymycobot.mycobot import MyCobot
import time

mc = MyCobot("/dev/ttyUSB0", 1000000)
mc.power_on()
time.sleep(1)

SPEED = 30
GRIPPER_SPEED = 80

INIT_ANGLES = [0.26, 0.87, -0.79, -1.49, 0.61, -44.12]

PICK_UP = [33.0, 215.5, 131.6, -173.81, 1.58, 48.76]
PICK_DOWN = [31.1, 219.6, 108.3, -177.72, 5.13, 48.8]

LIFT = [36.2, 193.1, 184.5, -172.23, 0.19, 48.61]
CENTER = [199.8, -8.8, 171.4, -174.94, 2.7, -33.71]

PLACE_UP = [81.9, -200.6, 160.4, -172.29, -4.55, -91.14]
PLACE_DOWN = [101.5, -204.9, 108.8, -178.43, -3.48, -85.94]

def move_angles(angles, seconds=3):
    mc.send_angles(angles, SPEED)
    time.sleep(seconds)

def move_coords(coords, seconds=3):
    mc.send_coords(coords, SPEED, 1)
    time.sleep(seconds)

def open_gripper():
    mc.set_gripper_value(100, GRIPPER_SPEED)
    time.sleep(1)

def close_gripper():
    mc.set_gripper_value(0, GRIPPER_SPEED)
    time.sleep(1)

def run_cycle():
    move_angles(INIT_ANGLES)
    open_gripper()

    move_coords(PICK_UP)
    move_coords(PICK_DOWN)
    close_gripper()

    move_coords(LIFT)
    move_coords(CENTER)

    move_coords(PLACE_UP)
    move_coords(PLACE_DOWN)
    open_gripper()

    move_coords(PLACE_UP)
    move_coords(CENTER)
    move_angles(INIT_ANGLES)

for i in range(5):
    print("Ciclo", i + 1)
    run_cycle()

print("Terminado")
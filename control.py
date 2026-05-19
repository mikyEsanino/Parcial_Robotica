from pymycobot.mycobot import MyCobot
import time

mc = MyCobot("/dev/ttyUSB0", 1000000)
mc.power_on()
time.sleep(1)

while True:
    input("Presiona ENTER para leer coordenadas...")
    print("Ángulos:", mc.get_angles())
    print("Coordenadas:", mc.get_coords())
    print()
from pymycobot.mycobot import MyCobot
import time

mc = MyCobot("/dev/ttyUSB0", 1000000)
mc.power_on()
time.sleep(1)

while True:
    nombre = input("Nombre de la pose: ")

    if nombre == "salir":
        break

    print(nombre)
    print("Ángulos:", mc.get_angles())
    print("Coordenadas:", mc.get_coords())
    print()
import time

# Pose home
mc.send_angles([0, 0, 0, 0, 0, 0], 30)
time.sleep(3)
coords_home = mc.get_coords()
print("HOME [0,0,0,0,0,0] coords:", coords_home)

# Solo rotar base 90°
mc.send_angles([90, 0, 0, 0, 0, 0], 30)
time.sleep(3)
coords_base = mc.get_coords()
print("BASE 90° coords:", coords_base)

# Solo bajar hombro
mc.send_angles([0, 45, 0, 0, 0, 0], 30)
time.sleep(3)
coords_hombro = mc.get_coords()
print("HOMBRO 45° coords:", coords_hombro)
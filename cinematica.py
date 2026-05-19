# Test definitivo de offsets
import time

tests = [
    [0,   0,   0, 0, 0, 0],
    [0,  90,   0, 0, 0, 0],  # J2 puro
    [0, -90,   0, 0, 0, 0],  # J2 puro negativo
    [0,   0,  90, 0, 0, 0],  # J3 puro
    [0,   0, -90, 0, 0, 0],  # J3 puro negativo
    [0,   0,   0, 90, 0, 0], # J4 puro
    [0,   0,   0, 0, 90, 0], # J5 puro
]

for angles in tests:
    mc.send_angles(angles, 30)
    time.sleep(3)
    c = mc.get_coords()
    print(f"angles={angles} → coords={c[:3] if c else 'ERROR'}")
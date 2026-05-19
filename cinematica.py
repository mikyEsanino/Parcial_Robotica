# Diagnóstico definitivo de offsets
fk = ForwardKinematics()

# Test: home real vs FK calculada
q_home = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
T = fk.compute_fk(q_home)
print("FK calcula:", round(T[0,3],2), round(T[1,3],2), round(T[2,3],2))

mc.send_angles(q_home, 30)
time.sleep(3)
real = mc.get_coords()
print("Robot real:", real[:3] if real else "ERROR")
print("Diferencia Z:", round(T[2,3] - real[2], 2) if real else "?")

# Ver qué tan lejos están los targets de IK
import numpy as np
for pos in [[140.0, 0.0, 220.0], [120.0, 60.0, 180.0], [150.0, -40.0, 200.0]]:
    x, y, z = pos
    r_total = np.sqrt(x**2 + y**2)
    wrist_offset = 66.39 + 73.18 + 48.6
    r_w = r_total - wrist_offset
    print(f"Target {pos}: r_total={r_total:.1f}, r_w={r_w:.1f} {'❌ NEGATIVO' if r_w < 0 else '✅'}")
import time
import numpy as np

def DH_matrix(theta_deg, d, a, alpha_deg):
    theta = np.radians(theta_deg)
    alpha = np.radians(alpha_deg)
    ct = np.cos(theta); st = np.sin(theta)
    ca = np.cos(alpha); sa = np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,     d],
        [0,   0,      0,      1]
    ])

class ForwardKinematics:
    def __init__(self):
        self.d1 = 131.56
        self.a2 = 110.4
        self.a3 = 96.0
        self.d4 = 66.39
        self.d5 = 73.18
        self.d6 = 48.6

    def compute_fk(self, joints):
        q1, q2, q3, q4, q5, q6 = joints
        A1 = DH_matrix(q1, self.d1, 0, 90)
        A2 = DH_matrix(q2, 0, self.a2, 0)
        A3 = DH_matrix(q3, 0, self.a3, 0)
        A4 = DH_matrix(q4, self.d4, 0, -90)
        A5 = DH_matrix(q5, self.d5, 0, 90)
        A6 = DH_matrix(q6, self.d6, 0, 0)
        return A1 @ A2 @ A3 @ A4 @ A5 @ A6

class InverseKinematics:
    def __init__(self, fk_model: ForwardKinematics):
        self.fk = fk_model

    def ik_solve(self, x, y, z):
        d1, a2, a3 = self.fk.d1, self.fk.a2, self.fk.a3
        q1_rad = np.arctan2(y, x)
        r = np.sqrt(x**2 + y**2)
        zc = z - d1
        cos_q3 = np.clip((r**2 + zc**2 - a2**2 - a3**2)/(2*a2*a3), -1.0, 1.0)
        q3_rad_up = np.arccos(cos_q3)
        q3_rad_down = -np.arccos(cos_q3)

        def q2_for_q3(q3_rad):
            alpha = np.arctan2(zc, r)
            beta = np.arctan2(a3 * np.sin(q3_rad), a2 + a3 * np.cos(q3_rad))
            return alpha - beta

        q2_rad_up = q2_for_q3(q3_rad_up)
        q2_rad_down = q2_for_q3(q3_rad_down)

        solutions = [
            [np.degrees(q1_rad), np.degrees(q2_rad_up), np.degrees(q3_rad_up)],
            [np.degrees(q1_rad), np.degrees(q2_rad_down), np.degrees(q3_rad_down)]
        ]
        chosen = solutions[1]  # preferir codo abajo
        chosen += [0.0, 0.0, 0.0]
        return chosen

class CollisionChecker:
    def check_collision(self, joints):
        return False

# ---------- TEST CON ROBOT ----------
fk = ForwardKinematics()
ik = InverseKinematics(fk)

cinco_configuraciones = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [30.0, 15.0, -20.0, 0.0, 45.0, 0.0],
    [-45.0, -10.0, 30.0, 15.0, -30.0, 10.0],
    [0.0, 45.0, -45.0, 0.0, 90.0, 0.0],
    [15.0, -30.0, 45.0, -15.0, 0.0, 25.0]
]

print("CONFIGURACIÓN | ERROR X (mm) | ERROR Y (mm) | ERROR Z (mm)")
print("-"*65)

for idx, angulos in enumerate(cinco_configuraciones, 1):
    mc.send_angles(angulos, 30)  # enviar al robot
    time.sleep(3.5)  # espera a que llegue a la posición
    
    coords_robot = mc.get_coords()  # leer posición real
    T_sim = fk.compute_fk(angulos)
    x_sim, y_sim, z_sim = T_sim[0,3], T_sim[1,3], T_sim[2,3]
    
    if coords_robot:
        err_x = abs(x_sim - coords_robot[0])
        err_y = abs(y_sim - coords_robot[1])
        err_z = abs(z_sim - coords_robot[2])
        print(f"Config {idx}     | {err_x:12.3f} | {err_y:12.3f} | {err_z:12.3f}")
    else:
        print(f"Config {idx}     | Error al leer posición del robot")

# TEST IK con robot
tres_posiciones = [
    [140.0, 0.0, 220.0],
    [120.0, 60.0, 180.0],
    [150.0, -40.0, 200.0]
]

print("\nPOSICIÓN OBJETIVO   | ERROR q1 (°) | ERROR q2 (°) | ERROR q3 (°)")
print("-"*65)

for idx, pos in enumerate(tres_posiciones, 1):
    angulos_calc = ik.ik_solve(*pos)
    mc.send_angles(angulos_calc, 30)
    time.sleep(3.5)
    
    angulos_robot = mc.get_angles()
    
    if angulos_robot:
        err_q1 = abs(angulos_calc[0] - angulos_robot[0])
        err_q2 = abs(angulos_calc[1] - angulos_robot[1])
        err_q3 = abs(angulos_calc[2] - angulos_robot[2])
        print(f"Pos {idx} {pos} | {err_q1:12.3f} | {err_q2:12.3f} | {err_q3:12.3f}")
    else:
        print(f"Pos {idx} {pos} | Error al leer ángulos del robot")
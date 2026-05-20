import time
import numpy as np

# =====================================================================
# 1. MATRIZ DH UNIFICADA
# =====================================================================
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

# =====================================================================
# 2. CLASE FK CON OFFSET HOME INCORPORADO
# =====================================================================
class ForwardKinematics:
    def __init__(self):
        # Dimensiones reales calibradas para el robot
        self.d2 = 134.65  
        self.a3 = -110.0   
        self.a4 = -96.0    
        self.d5 = 63.4    
        self.d6 = 75.05   
        self.a6 = 51.8    

        # Offsets home (medidos en robot real)
        self.offset_angles = [0.7, 0.35, -0.17, -1.4, 0.79, -44.29]  # grados
        self.offset_coords = [49.9, -63.3, 409.7]  # mm (X,Y,Z)

    def compute_fk(self, joints):
        # Aplicar offsets de home reales a los ángulos
        q1, q2, q3, q4, q5, q6 = [j + off for j, off in zip(joints, self.offset_angles)]

        A1 = DH_matrix(q1, 0, 0, 0)
        A2 = DH_matrix(q2, self.d2, 0, 90)
        A3 = DH_matrix(q3, 0, self.a3, 0)
        A4 = DH_matrix(q4, 0, self.a4, 0)
        A5 = DH_matrix(q5, self.d5, 0, -90)
        A6 = DH_matrix(q6, self.d6, self.a6, 90)

        T = A1 @ A2 @ A3 @ A4 @ A5 @ A6

        # Ajuste posicional según home real
        T[0,3] += self.offset_coords[0]
        T[1,3] += self.offset_coords[1]
        T[2,3] += self.offset_coords[2]

        return T

# =====================================================================
# 3. CLASE IK CON OFFSET HOME
# =====================================================================
class InverseKinematics:
    def __init__(self, fk_model: ForwardKinematics):
        self.fk = fk_model

    def ik_solve(self, x, y, z):
        # Compensar coordenadas por home real
        x_corr = x - self.fk.offset_coords[0]
        y_corr = y - self.fk.offset_coords[1]
        z_corr = z - self.fk.offset_coords[2]

        d2 = self.fk.d2
        a3 = self.fk.a3
        a4 = self.fk.a4

        # 1. Base
        q1_rad = np.arctan2(y_corr, x_corr)
        r = np.sqrt(x_corr**2 + y_corr**2)
        zc = z_corr - d2

        # 2. Ley de cosenos para codo abajo
        cos_q3 = np.clip((r**2 + zc**2 - a3**2 - a4**2)/(2*a3*a4), -1.0, 1.0)
        q3_rad = -np.arccos(cos_q3)

        # 3. Hombro
        alpha = np.arctan2(zc, r)
        beta = np.arctan2(a4 * np.sin(q3_rad), a3 + a4 * np.cos(q3_rad))
        q2_rad = alpha - beta

        # 4. Conversión a grados
        q1 = np.degrees(q1_rad)
        q2 = np.degrees(q2_rad)
        q3 = np.degrees(q3_rad)

        # Aplicar offsets home (para IK inversa)
        q1 -= self.fk.offset_angles[0]
        q2 -= self.fk.offset_angles[1]
        q3 -= self.fk.offset_angles[2]

        return [q1, q2, q3, 0.0, 0.0, 0.0]

# =====================================================================
# 4. CLASE CHECK COLLISION
# =====================================================================
class CollisionChecker:
    def check_collision(self, joints):
        return False

# =====================================================================
# 5. TEST DIRECTO CON ROBOT
# =====================================================================
fk = ForwardKinematics()
ik = InverseKinematics(fk)

cinco_configuraciones = [
    [0.0,   0.0,   0.0,  0.0,  0.0,  0.0],
    [30.0,  15.0, -20.0, 0.0,  45.0, 0.0],
    [-45.0,-10.0,  30.0, 15.0,-30.0, 10.0],
    [0.0,   45.0, -45.0, 0.0,  90.0, 0.0],
    [15.0, -30.0,  45.0,-15.0,  0.0, 25.0]
]

print("\n--- EJECUTANDO TEST CINEMÁTICA DIRECTA (FK) ---")
print("CONFIGURACIÓN | ERROR X (mm) | ERROR Y (mm) | ERROR Z (mm)")
print("-"*65)

for idx, angulos in enumerate(cinco_configuraciones, 1):
    mc.send_angles(angulos, 30)
    time.sleep(3.5)
    coords_robot = mc.get_coords()

    T_tuya = fk.compute_fk(angulos)
    mi_x, mi_y, mi_z = T_tuya[0,3], T_tuya[1,3], T_tuya[2,3]

    if coords_robot:
        err_x = abs(mi_x - coords_robot[0])
        err_y = abs(mi_y - coords_robot[1])
        err_z = abs(mi_z - coords_robot[2])
        print(f"Config {idx}     | {err_x:12.3f} | {err_y:12.3f} | {err_z:12.3f}")
    else:
        print(f"Config {idx}     | Error al leer datos del robot físico.")

# Test IK
tres_posiciones = [
    [140.0, 0.0, 220.0],
    [120.0, 60.0, 180.0],
    [150.0, -40.0, 200.0]
]

print("\n--- EJECUTANDO TEST CINEMÁTICA INVERSA (IK) ---")
print("POSICIÓN OBJETIVO   | ERROR q1 (°) | ERROR q2 (°) | ERROR q3 (°)")
print("-"*65)

for idx, pos in enumerate(tres_posiciones, 1):
    tus_angulos = ik.ik_solve(*pos)
    mc.send_angles(tus_angulos, 30)
    time.sleep(3.5)
    angulos_robot = mc.get_angles()

    if angulos_robot:
        err_q1 = abs(tus_angulos[0] - angulos_robot[0])
        err_q2 = abs(tus_angulos[1] - angulos_robot[1])
        err_q3 = abs(tus_angulos[2] - angulos_robot[2])
        print(f"Pos {idx} {pos} | {err_q1:12.3f} | {err_q2:12.3f} | {err_q3:12.3f}")
    else:
        print(f"Pos {idx} {pos} | Error al leer los ángulos de los motores.")
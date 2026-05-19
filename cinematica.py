import time
import sympy as sp
import numpy as np

sp.init_printing(use_unicode=True)

def DH_matrix(theta_deg, d, a, alpha_deg):
    """Matriz de transformación DH en grados"""
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
        # Parámetros DH del MyCobot 280
        self.d1 = 131.56
        self.a2 = 110.4
        self.a3 = 96.0
        self.d4 = 66.39
        self.d5 = 73.18
        self.d6 = 48.6

    def compute_fk(self, joints):
        q1, q2, q3, q4, q5, q6 = joints
        
        # Matrices DH con offsets consistentes (sin restas arbitrarias)
        A1 = DH_matrix(q1, self.d1, 0, 90)
        A2 = DH_matrix(q2, 0, self.a2, 0)
        A3 = DH_matrix(q3, 0, self.a3, 0)
        A4 = DH_matrix(q4, self.d4, 0, -90)
        A5 = DH_matrix(q5, self.d5, 0, 90)
        A6 = DH_matrix(q6, self.d6, 0, 0)
        
        T = A1 @ A2 @ A3 @ A4 @ A5 @ A6
        return T

class InverseKinematics:
    def __init__(self, fk_model: ForwardKinematics):
        self.fk = fk_model

    def ik_solve(self, x, y, z):
        d1 = self.fk.d1
        a2 = self.fk.a2
        a3 = self.fk.a3

        # 1. Base
        q1_rad = np.arctan2(y, x)

        # 2. Proyección en plano vertical
        r = np.sqrt(x**2 + y**2)
        zc = z - d1

        # 3. Ley de cosenos para codo
        cos_q3 = np.clip((r**2 + zc**2 - a2**2 - a3**2)/(2*a2*a3), -1.0, 1.0)
        
        # Dos soluciones de codo
        q3_rad_up = np.arccos(cos_q3)      # codo arriba
        q3_rad_down = -np.arccos(cos_q3)   # codo abajo

        # 4. Ángulo del hombro para ambas soluciones
        def q2_for_q3(q3_rad):
            alpha = np.arctan2(zc, r)
            beta = np.arctan2(a3 * np.sin(q3_rad), a2 + a3 * np.cos(q3_rad))
            return alpha - beta

        q2_rad_up = q2_for_q3(q3_rad_up)
        q2_rad_down = q2_for_q3(q3_rad_down)

        # 5. Convertir a grados
        solutions = [
            [np.degrees(q1_rad), np.degrees(q2_rad_up), np.degrees(q3_rad_up)],
            [np.degrees(q1_rad), np.degrees(q2_rad_down), np.degrees(q3_rad_down)]
        ]

        # Seleccionar solución más “natural” (codo abajo preferido)
        chosen = solutions[1]  # puedes cambiar a 0 si quieres codo arriba

        # Completar ángulos restantes como 0 para simplificación
        chosen += [0.0, 0.0, 0.0]

        return chosen

class CollisionChecker:
    def __init__(self):
        pass

    def check_collision(self, joints):
        return False
    
###TEST FK###
fk = ForwardKinematics()

cinco_configuraciones = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],         
    [30.0, 15.0, -20.0, 0.0, 45.0, 0.0],    
    [-45.0, -10.0, 30.0, 15.0, -30.0, 10.0],
    [0.0, 45.0, -45.0, 0.0, 90.0, 0.0],     
    [15.0, -30.0, 45.0, -15.0, 0.0, 25.0]   
]

print("CONFIGURACIÓN | ERROR X (mm) | ERROR Y (mm) | ERROR Z (mm)")
print("-" * 65)

for idx, angulos in enumerate(cinco_configuraciones, 1):
    mc.send_angles(angulos, 30)
    time.sleep(3.0) 
    
    coords_robot = mc.get_coords()
    
    T_tuya = fk.compute_fk(angulos)
    mi_x = T_tuya[0, 3]
    mi_y = T_tuya[1, 3]
    mi_z = T_tuya[2, 3]
    
    if coords_robot:
        err_x = abs(mi_x - coords_robot[0])
        err_y = abs(mi_y - coords_robot[1])
        err_z = abs(mi_z - coords_robot[2])
        print(f"Config {idx}     | {err_x:12.3f} | {err_y:12.3f} | {err_z:12.3f}")
    else:
        print(f"Config {idx}     | Error al leer datos del robot físico.")

###TEST IK##
ik = InverseKinematics(fk)

tres_posiciones = [
    [140.0, 0.0, 220.0],    
    [120.0, 60.0, 180.0],   
    [150.0, -40.0, 200.0]   
]

print("\nPOSICIÓN OBJETIVO   | ERROR q1 (°) | ERROR q2 (°) | ERROR q3 (°)")
print("-" * 65)

for idx, pos in enumerate(tres_posiciones, 1):
    x_obj, y_obj, z_obj = pos
    
    tus_angulos = ik.ik_solve(x_obj, y_obj, z_obj)
    
    # ENVIAMOS TUS ÁNGULOS CALCULADOS AL ROBOT REAL
    mc.send_angles(tus_angulos, 30)
    time.sleep(3.5) 
    
    angulos_robot = mc.get_angles()
    
    if angulos_robot:
        err_q1 = abs(tus_angulos[0] - angulos_robot[0])
        err_q2 = abs(tus_angulos[1] - angulos_robot[1])
        err_q3 = abs(tus_angulos[2] - angulos_robot[2])
        print(f"Pos {idx} {str(pos):10} | {err_q1:12.3f} | {err_q2:12.3f} | {err_q3:12.3f}")
    else:
        print(f"Pos {idx} {str(pos):10} | Error al leer los ángulos de los motores.")
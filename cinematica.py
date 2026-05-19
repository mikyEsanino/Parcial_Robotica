import time
import sympy as sp
import numpy as np

sp.init_printing(use_unicode=True)

def DH_matrix_calculo(theta, d, a, alpha, es_simbolico=False):
    if es_simbolico:
        ct = sp.cos(theta); st = sp.sin(theta)
        ca = sp.cos(alpha); sa = sp.sin(alpha)
        return sp.Matrix([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0,   sa,       ca,      d     ],
            [0,   0,        0,       1     ]
        ])
    else:
        rad_theta = np.radians(theta)
        rad_alpha = np.radians(alpha)
        ct = np.cos(rad_theta); st = np.sin(rad_theta)
        ca = np.cos(rad_alpha); sa = np.sin(rad_alpha)
        return np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0,   sa,       ca,      d     ],
            [0,   0,        0,       1     ]
        ])

# Variables simbólicas
θ1, θ2, θ3, θ4, θ5, θ6 = sp.symbols('θ1 θ2 θ3 θ4 θ5 θ6')

# Expresión simbólica exacta usando sp.pi (Para tu reporte P1)
A1 = DH_matrix_calculo(θ1, 131.56, 0, 90 * sp.pi / 180, es_simbolico=True)
A2 = DH_matrix_calculo(θ2, 0, 110.4, 0, es_simbolico=True)
A3 = DH_matrix_calculo(θ3, 0, 96, 0, es_simbolico=True)
A4 = DH_matrix_calculo(θ4, 66.39, 0, -90 * sp.pi / 180, es_simbolico=True)
A5 = DH_matrix_calculo(θ5, 73.18, 0, 90 * sp.pi / 180, es_simbolico=True)
A6 = DH_matrix_calculo(θ6, 48.6, 0, 0, es_simbolico=True)


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
        
        math_q1 = q1
        math_q2 = q2 - 90.0
        math_q3 = q3
        math_q4 = q4 - 90.0
        math_q5 = q5
        
        A1_n = DH_matrix_calculo(math_q1, self.d1, 0, 90)
        A2_n = DH_matrix_calculo(math_q2, 0, self.a2, 0)
        A3_n = DH_matrix_calculo(math_q3, 0, self.a3, 0)
        A4_n = DH_matrix_calculo(math_q4, self.d4, 0, -90)
        A5_n = DH_matrix_calculo(math_q5, self.d5, 0, 90)
        A6_n = DH_matrix_calculo(q6, self.d6, 0, 0)
        
        return A1_n @ A2_n @ A3_n @ A4_n @ A5_n @ A6_n


class InverseKinematics:
    def __init__(self, fk_model: ForwardKinematics):
        self.fk = fk_model

    def ik_solve(self, x, y, z):
        d1 = self.fk.d1
        a2 = self.fk.a2
        a3 = self.fk.a3

        # 1. Ángulo de la Base
        q1_rad = np.arctan2(y, x)

        # 2. Proyección en el plano vertical
        r = np.sqrt(x**2 + y**2)
        zc = z - d1 

        # 3. Ley de Cosenos para el Codo
        num = r**2 + zc**2 - a2**2 - a3**2
        den = 2 * a2 * a3
        cos_q3 = np.clip(num / den, -1.0, 1.0)
        
        q3_rad = -np.arccos(cos_q3) 

        # 4. Ángulo del Hombro
        alpha = np.arctan2(zc, r)
        beta = np.arctan2(a3 * np.sin(q3_rad), a2 + a3 * np.cos(q3_rad))
        q2_rad = alpha - beta

        # 5. Conversión a grados aplicando la inversa de los offsets de hardware
        q1 = np.degrees(q1_rad)
        q2 = np.degrees(q2_rad) + 90.0 
        q3 = np.degrees(q3_rad)

        return [q1, q2, q3, 0.0, 0.0, 0.0]

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
import time
import sympy as sp
import numpy as np

sp.init_printing(use_unicode=True)

def DH_matrix_calculo(theta, d, a, alpha, es_simbolico=False):
    """Calcula la matriz unificada DH para evitar errores de acumulación"""
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

θ1, θ2, θ3, θ4, θ5, θ6 = sp.symbols('θ1 θ2 θ3 θ4 θ5 θ6')

A1 = DH_matrix_calculo(θ1, 0, 0, 0, es_simbolico=True)
A2 = DH_matrix_calculo(θ2, 134.65, 0, sp.pi/2, es_simbolico=True)
A3 = DH_matrix_calculo(θ3 - np.radians(90), 0, -110, 0, es_simbolico=True)
A4 = DH_matrix_calculo(θ4, 0, -96, 0, es_simbolico=True)
A5 = DH_matrix_calculo(θ5 + np.radians(90), 63.4, 0, -sp.pi/2, es_simbolico=True)
A6 = DH_matrix_calculo(θ6 + np.radians(90), 75.05, 51.8, sp.pi/2, es_simbolico=True)

class ForwardKinematics:
    def __init__(self):
        # Constantes de longitud derivadas de tus nuevas matrices fijas
        self.d2 = 134.65
        self.a3 = -110.0
        self.a4 = -96.0
        self.d5 = 63.4
        self.d6 = 75.05
        self.a6 = 51.8

    def compute_fk(self, joints):
        q1, q2, q3, q4, q5, q6 = joints

        # Mapeo idéntico a los argumentos que definiste para tus matrices simbólicas
        A1_num = DH_matrix_calculo(q1, 0, 0, 0)
        A2_num = DH_matrix_calculo(q2, self.d2, 0, 90)
        A3_num = DH_matrix_calculo(q3 - 90.0, 0, self.a3, 0)
        A4_num = DH_matrix_calculo(q4, 0, self.a4, 0)
        A5_num = DH_matrix_calculo(q5 + 90.0, self.d5, 0, -90)
        A6_num = DH_matrix_calculo(q6 + 90.0, self.d6, self.a6, 90)

        return A1_num @ A2_num @ A3_num @ A4_num @ A5_num @ A6_num


class InverseKinematics:
    def __init__(self, fk_model: ForwardKinematics):
        self.fk = fk_model

    def ik_solve(self, x, y, z):
        # Usamos las dimensiones de los eslabones principales coplanares
        d2 = self.fk.d2
        a3 = abs(self.fk.a3)
        a4 = abs(self.fk.a4)

        # 1. Ángulo de la Base
        q1_rad = np.arctan2(y, x)

        # 2. Desplazamiento planar vertical
        r = np.sqrt(x**2 + y**2)
        zc = z - d2

        # 3. Teorema del Coseno (Codo)
        num = r**2 + zc**2 - a3**2 - a4**2
        den = 2 * a3 * a4
        cos_q3 = np.clip(num / den, -1.0, 1.0)
        q3_rad = np.arccos(cos_q3)

        # 4. Orientación del Hombro
        alpha = np.arctan2(zc, r)
        beta = np.arctan2(a4 * np.sin(q3_rad), a3 + a4 * np.cos(q3_rad))
        q2_rad = alpha - beta

        # 5. Conversión a grados e inversión de tus desfases
        q1 = np.degrees(q1_rad)
        q2 = np.degrees(q2_rad)
        q3 = np.degrees(q3_rad) + 90.0 # Operación matemática inversa al offset articular

        return [q1, q2, q3, 0.0, 0.0, 0.0]

###TESTSSSSS###
fk = ForwardKinematics()

cinco_configuraciones = [
    [0.0,   0.0,   0.0,  0.0,  0.0,  0.0],
    [30.0,  15.0, -20.0, 0.0,  45.0, 0.0],
    [-45.0,-10.0,  30.0, 15.0,-30.0, 10.0],
    [0.0,   45.0, -45.0, 0.0,  90.0, 0.0],
    [15.0, -30.0,  45.0,-15.0,  0.0, 25.0]
]

print("CONFIGURACIÓN | ERROR X (mm) | ERROR Y (mm) | ERROR Z (mm)")
print("-" * 65)

for idx, angulos in enumerate(cinco_configuraciones, 1):
    mc.send_angles(angulos, 30)
    time.sleep(3.0)
    coords_robot = mc.get_coords()

    T_tuya = fk.compute_fk(angulos)
    mi_x, mi_y, mi_z = T_tuya[0, 3], T_tuya[1, 3], T_tuya[2, 3]

    if coords_robot:
        err_x = abs(mi_x - coords_robot[0])
        err_y = abs(mi_y - coords_robot[1])
        err_z = abs(mi_z - coords_robot[2])
        print(f"Config {idx}     | {err_x:12.3f} | {err_y:12.3f} | {err_z:12.3f}")
    else:
        print(f"Config {idx}     | Error al leer datos del robot físico.")


ik = InverseKinematics(fk)

tres_posiciones = [
    [140.0,  0.0, 220.0],
    [120.0, 60.0, 180.0],
    [150.0,-40.0, 200.0]
]

print("\nPOSICIÓN OBJETIVO   | ERROR q1 (°) | ERROR q2 (°) | ERROR q3 (°)")
print("-" * 65)

for idx, pos in enumerate(tres_posiciones, 1):
    x_obj, y_obj, z_obj = pos
    tus_angulos = ik.ik_solve(x_obj, y_obj, z_obj)

    # CORRECCIÓN DE LA API: Enviamos tus ángulos calculados para probar tu solucionador
    mc.send_angles(tus_angulos, 30)
    time.sleep(3.5)
    angulos_robot = mc.get_angles()

    if angulos_robot:
        err_q1 = abs(tus_angulos[0] - angulos_robot[0])
        err_q2 = abs(tus_angulos[1] - angulos_robot[1])
        err_q3 = abs(tus_angulos[2] - angulos_robot[2])
        print(f"Pos {idx} {str(pos):18} | {err_q1:12.3f} | {err_q2:12.3f} | {err_q3:12.3f}")
    else:
        print(f"Pos {idx} {str(pos):18} | Error al leer los ángulos de los motores.")

# COMPROBACIÓN FINAL DE CALIBRACIÓN EN HOME
T1 = fk.compute_fk([0, 0, 0, 0, 0, 0])
print("\n--- VERIFICACIÓN EN HOME ---")
print(f"Tu FK calculada: {T1[0,3]:.1f}, {T1[1,3]:.1f}, {T1[2,3]:.1f}")
print("Robot real objetivo:  50.3, -63.3, 409.8")
import time
import sympy as sp
import numpy as np
from sympy import symbols, diff, sin, cos
from sympy.physics.mechanics import dynamicsymbols

sp.init_printing(use_unicode=True)

# definimos rotacion y traslacion
def trans(x,y,z):
  M = sp.Matrix([[1, 0, 0, x],[0, 1, 0, y],[0, 0, 1, z],[0, 0, 0, 1]])
  return M

def rotx(ang):
  M = sp.Matrix([[1,0,0,0],[0, sp.cos(ang), -sp.sin(ang) , 0],[0, sp.sin(ang), sp.cos(ang),0 ],[0,0,0,1]])
  return M

def roty(ang):
  M = sp.Matrix([[ sp.cos(ang), 0, sp.sin(ang), 0],[0,1,0,0],[-sp.sin(ang), 0, sp.cos(ang),0],[0,0,0,1]])
  return M

def rotz(ang):
  M = sp.Matrix([[ sp.cos(ang), -sp.sin(ang), 0, 0],[sp.sin(ang), sp.cos(ang),0, 0],[0,0,1,0], [0,0,0,1]])
  return M

# definimos DH
def DH(theta,d,a,alpha):
  tr = rotz(theta)*trans(0,0,d)*trans(a,0,0)*rotx(alpha)
  return tr

# definimos las variables simbolicas
θ1, θ2, θ3, θ4, θ5, θ6 = sp.symbols('θ1 θ2 θ3 θ4 θ5 θ6')

# armamos las 6 matrices de transformación con los parametros DH
A1 = DH(θ1, 131.56, 0, 90) #base
sp.pprint(A1)

A2 = DH(θ2, 0, 110.4, 0) #hombro
sp.pprint(A2)

A3 = DH(θ3, 0, 96, 0) #codo
sp.pprint(A3)

A4 = DH(θ4, 66.39, 0, -90) #muñeca 1
sp.pprint(A4)

A5 = DH(θ5, 73.18, 0, 90) #muñeca 2
sp.pprint(A5)

A6 = DH(θ6, 48.6, 0, 0) #gripper
sp.pprint(A6)

class ForwardKinematics:
    def __init__(self):
        self.d1 = 131.56
        self.a2 = 110.4
        self.a3 = 96.0
        self.d4 = 66.39
        self.d5 = 73.18
        self.d6 = 48.6

    def _dh_matrix(self, theta, d, a, alpha):
        rad_theta = np.radians(theta)
        rad_alpha = np.radians(alpha)
        
        ct = np.cos(rad_theta)
        st = np.sin(rad_theta)
        ca = np.cos(rad_alpha)
        sa = np.sin(rad_alpha)
        
        return np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0,   sa,       ca,      d     ],
            [0,   0,        0,       1     ]
        ])

    def compute_fk(self, joints):
        q1, q2, q3, q4, q5, q6 = joints
        
        A1_num = self._dh_matrix(q1, self.d1, 0, 90)
        A2_num = self._dh_matrix(q2, 0, self.a2, 0)
        A3_num = self._dh_matrix(q3, 0, self.a3, 0)
        A4_num = self._dh_matrix(q4, self.d4, 0, -90)
        A5_num = self._dh_matrix(q5, self.d5, 0, 90)
        A6_num = self._dh_matrix(q6, self.d6, 0, 0)
        
        T_0_6 = A1_num @ A2_num @ A3_num @ A4_num @ A5_num @ A6_num
        return T_0_6
    
class InverseKinematics:
    def __init__(self, fk_model: ForwardKinematics):
        self.fk = fk_model

    def ik_solve(self, x, y, z):
        #Plano 2R + Base
        d1 = self.fk.d1
        a2 = self.fk.a2
        a3 = self.fk.a3

        #Base
        q1_rad = np.arctan2(y, x)

        #brazo
        r = np.sqrt(x**2 + y**2)
        zc = z - d1 

        #Codo
        num = r**2 + zc**2 - a2**2 - a3**2
        den = 2 * a2 * a3
        cos_q3 = np.clip(num / den, -1.0, 1.0)
        
        #codo arriba
        q3_rad = np.arccos(cos_q3) 

        #Hombro
        alpha = np.arctan2(zc, r)
        beta = np.arctan2(a3 * np.sin(q3_rad), a2 + a3 * np.cos(q3_rad))
        q2_rad = alpha - beta

        #Pasar a grados para el MyCobot
        q1 = np.degrees(q1_rad)
        q2 = np.degrees(q2_rad)
        q3 = np.degrees(q3_rad)

        # Dejamos las muñecas en 0 por la simplificación planar solicitada
        return [q1, q2, q3, 0.0, 0.0, 0.0]

class CollisionChecker:
    def __init__(self):
        pass

    def check_collision(self, joints):
        return False
    
###TEST FK###
fk = ForwardKinematics()

# Definimos 5 configuraciones distintas de ángulos en grados [q1, q2, q3, q4, q5, q6]
cinco_configuraciones = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],         # Config 1: Home / Todo recto
    [30.0, 15.0, -20.0, 0.0, 45.0, 0.0],    # Config 2: Movimiento general 
    [-45.0, -10.0, 30.0, 15.0, -30.0, 10.0],# Config 3: Movimiento opuesto
    [0.0, 45.0, -45.0, 0.0, 90.0, 0.0],     # Config 4: Ángulos rectos
    [15.0, -30.0, 45.0, -15.0, 0.0, 25.0]   # Config 5: Postura arbitraria
]

print("CONFIGURACIÓN | ERROR X (mm) | ERROR Y (mm) | ERROR Z (mm)")
print("-" * 65)

for idx, angulos in enumerate(cinco_configuraciones, 1):
    # 1. Mover el robot real mediante la API del fabricante
    mc.send_angles(angulos, 30)
    time.sleep(3.0) # Esperamos 3 segundos a que se detenga físicamente
    
    # 2. Leer la posición cartesiana real reportada por el fabricante
    coords_robot = mc.get_coords()
    
    # 3. Calcular la matriz homogénea teórica con TU código
    T_tuya = fk.compute_fk(angulos)
    mi_x = T_tuya[0, 3]
    mi_y = T_tuya[1, 3]
    mi_z = T_tuya[2, 3]
    
    if coords_robot:
        # 4. Calcular el error absoluto matemático en mm
        err_x = abs(mi_x - coords_robot[0])
        err_y = abs(mi_y - coords_robot[1])
        err_z = abs(mi_z - coords_robot[2])
        
        print(f"Config {idx}     | {err_x:12.3f} | {err_y:12.3f} | {err_z:12.3f}")
    else:
        print(f"Config {idx}     | Error al leer datos del robot físico.")

###TEST IK##
ik = InverseKinematics(fk)

# Definimos 3 posiciones [X, Y, Z] en mm que sean seguras y alcanzables para el brazo
tres_posiciones = [
    [140.0, 0.0, 220.0],    # Posición 1: Centrado y elevado
    [120.0, 60.0, 180.0],   # Posición 2: Lateral medio
    [150.0, -40.0, 200.0]   # Posición 3: Lateral opuesto bajo
]

print("POSICIÓN OBJETIVO   | ERROR q1 (°) | ERROR q2 (°) | ERROR q3 (°)")
print("-" * 65)

for idx, pos in enumerate(tres_posiciones, 1):
    x_obj, y_obj, z_obj = pos
    
    # 1. Tu algoritmo calcula analíticamente qué ángulos se necesitan
    tus_angulos = ik.ik_solve(x_obj, y_obj, z_obj)
    
    # 2. Le ordenamos al robot real ir a esa coordenada usando su API cartesiana nativa
    # Nota: Pasamos orientaciones [rx, ry, rz] en 0 para la trayectoria
    mc.send_coords([x_obj, y_obj, z_obj, 0.0, 0.0, 0.0], 30, 1)
    time.sleep(3.5) # Esperamos a que los motores completen la trayectoria
    
    # 3. Leemos qué ángulos reales adoptaron físicamente los servomotores
    angulos_robot = mc.get_angles()
    
    if angulos_robot:
        # 4. Comparamos los primeros 3 joints del modelo planar contra la realidad física
        err_q1 = abs(tus_angulos[0] - angulos_robot[0])
        err_q2 = abs(tus_angulos[1] - angulos_robot[1])
        err_q3 = abs(tus_angulos[2] - angulos_robot[2])
        
        print(f"Pos {idx} {str(pos):10} | {err_q1:12.3f} | {err_q2:12.3f} | {err_q3:12.3f}")
    else:
        print(f"Pos {idx} {str(pos):10} | Error al leer los ángulos de los motores.")
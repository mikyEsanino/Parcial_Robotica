import time
import sympy as sp
import numpy as np
 
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
A1 = DH(θ1, 173.5, 0, sp.pi/2) # d1 deducido: cuando J2=90, Z=173.5
sp.pprint(A1)

A2 = DH(θ2, 0, 113.4, 0) # a2 deducido: cuando J3=90, Z=286.9 -> a2=286.9-173.5
sp.pprint(A2)

A3 = DH(θ3, 0, 96, 0) # a3 se mantiene del manual
sp.pprint(A3)

A4 = DH(θ4, 66.39, 0, -sp.pi/2) # d4 del manual
sp.pprint(A4)

A5 = DH(θ5, 73.18, 0, sp.pi/2) # d5 del manual
sp.pprint(A5)

A6 = DH(θ6, 48.6, 0, 0) # d6 del manual
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
        ct = np.cos(rad_theta); st = np.sin(rad_theta)
        ca = np.cos(rad_alpha); sa = np.sin(rad_alpha)
        return np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,   sa,     ca,    d   ],
            [0,   0,      0,     1   ]
        ])
 
    def compute_fk(self, joints):
        q1, q2, q3, q4, q5, q6 = joints
 
        # FIX: offsets de hardware del MyCobot 280
        # q_dh = q_robot + offset
        # Validado empiricamente: home [0,0,0,0,0,0] -> Z=409mm
        A1_num = self._dh_matrix(q1,        self.d1, 0,       90)
        A2_num = self._dh_matrix(q2 - 90.0, 0,       self.a2, 0)
        A3_num = self._dh_matrix(q3,        0,       self.a3, 0)
        A4_num = self._dh_matrix(q4 - 90.0, self.d4, 0,      -90)
        A5_num = self._dh_matrix(q5,        self.d5, 0,       90)
        A6_num = self._dh_matrix(q6,        self.d6, 0,       0)
 
        return A1_num @ A2_num @ A3_num @ A4_num @ A5_num @ A6_num
 
 
class InverseKinematics:
    def __init__(self, fk_model: ForwardKinematics):
        self.fk = fk_model
 
    def ik_solve(self, x, y, z):
        d1 = self.fk.d1  # 131.56
        a2 = self.fk.a2  # 110.4
        a3 = self.fk.a3  # 96.0
 
        # Base
        q1_rad = np.arctan2(y, x)
 
        # Proyección planar — usamos el punto target directamente
        # (modelo 2R simplificado, primeros 3 joints)
        r  = np.sqrt(x**2 + y**2)
        zc = z - d1
 
        # Codo (elbow-up: arccos positivo)
        cos_q3 = np.clip((r**2 + zc**2 - a2**2 - a3**2) / (2*a2*a3), -1.0, 1.0)
        q3_rad = np.arccos(cos_q3)
 
        # Hombro
        alpha  = np.arctan2(zc, r)
        beta   = np.arctan2(a3*np.sin(q3_rad), a2 + a3*np.cos(q3_rad))
        q2_rad = alpha - beta
 
        # Pasar a grados
        q1 = np.degrees(q1_rad)
        q3 = np.degrees(q3_rad)
 
        # FIX: q2 necesita +90° para compensar el offset de hardware
        q2 = np.degrees(q2_rad) + 90.0
 
        return [q1, q2, q3, 0.0, 0.0, 0.0]
 
 
class CollisionChecker:
    def __init__(self):
        pass
 
    def check_collision(self, joints):
        return False
 
 
###TEST FK###
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
 
 
###TEST IK###
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
 
    mc.send_coords([x_obj, y_obj, z_obj, 0.0, 0.0, 0.0], 30, 1)
    time.sleep(3.5)
    angulos_robot = mc.get_angles()
 
    if angulos_robot:
        err_q1 = abs(tus_angulos[0] - angulos_robot[0])
        err_q2 = abs(tus_angulos[1] - angulos_robot[1])
        err_q3 = abs(tus_angulos[2] - angulos_robot[2])
        print(f"Pos {idx} {str(pos):18} | {err_q1:12.3f} | {err_q2:12.3f} | {err_q3:12.3f}")
    else:
        print(f"Pos {idx} {str(pos):18} | Error al leer los ángulos de los motores.")

fk = ForwardKinematics()

# Sin offset
T1 = fk.compute_fk([0, 0, 0, 0, 0, 0])
print("Tu FK actual:", round(T1[0,3],1), round(T1[1,3],1), round(T1[2,3],1))
print("Robot real:   50.3, -63.3, 409.8")
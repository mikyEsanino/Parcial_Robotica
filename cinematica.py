import time
import numpy as np
import sympy as sp

sp.init_printing(use_unicode=True)

# =====================================================================
# FUNCIONES SIMBÓLICAS (Para reporte P1)
# =====================================================================
def trans(x,y,z):
    return sp.Matrix([[1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]])

def rotx(ang):
    return sp.Matrix([[1,0,0,0],[0,sp.cos(ang),-sp.sin(ang),0],[0,sp.sin(ang),sp.cos(ang),0],[0,0,0,1]])

def rotz(ang):
    return sp.Matrix([[sp.cos(ang),-sp.sin(ang),0,0],[sp.sin(ang),sp.cos(ang),0,0],[0,0,1,0],[0,0,0,1]])

def DH_sym(theta,d,a,alpha):
    return rotz(theta)*trans(0,0,d)*trans(a,0,0)*rotx(alpha)

θ1,θ2,θ3,θ4,θ5,θ6 = sp.symbols('θ1 θ2 θ3 θ4 θ5 θ6')

A1 = DH_sym(θ1, 134.65,   0,      sp.pi/2)
A2 = DH_sym(θ2,   0,    -110.0,   0      )
A3 = DH_sym(θ3,   0,     -96.0,   0      )
A4 = DH_sym(θ4,  63.4,    0,    -sp.pi/2 )
A5 = DH_sym(θ5,  75.05,   0,     sp.pi/2 )
A6 = DH_sym(θ6,  51.8,    0,      0      )

# =====================================================================
# MATRIZ DH NUMÉRICA
# =====================================================================
def DH_matrix(theta_deg, d, a, alpha_deg):
    theta = np.radians(theta_deg)
    alpha = np.radians(alpha_deg)
    ct = np.cos(theta); st = np.sin(theta)
    ca = np.cos(alpha); sa = np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d   ],
        [0,   0,      0,     1   ]
    ])

# =====================================================================
# CLASE FK  — sin offsets de posición, solo parámetros DH reales
# =====================================================================
class ForwardKinematics:
    def __init__(self):
        self.d1  = 134.65
        self.a2  = -110.0
        self.a3  = -96.0
        self.d4  = 63.4
        self.d5  = 75.05
        self.a6  = 51.8

    def compute_fk(self, joints):
        """
        joints: [q1..q6] en grados, convención del robot.
        Retorna T_0_6 (4x4 numpy).
        """
        q1, q2, q3, q4, q5, q6 = joints

        A1 = DH_matrix(q1, self.d1,  0,       90)
        A2 = DH_matrix(q2, 0,        self.a2,  0)
        A3 = DH_matrix(q3, 0,        self.a3,  0)
        A4 = DH_matrix(q4, self.d4,  0,       -90)
        A5 = DH_matrix(q5, self.d5,  0,        90)
        A6 = DH_matrix(q6, self.a6,  0,         0)

        return A1 @ A2 @ A3 @ A4 @ A5 @ A6

    def get_all_frames(self, joints):
        """Retorna todos los sistemas de referencia intermedios."""
        q1, q2, q3, q4, q5, q6 = joints
        matrices = [
            DH_matrix(q1, self.d1,  0,       90),
            DH_matrix(q2, 0,        self.a2,  0),
            DH_matrix(q3, 0,        self.a3,  0),
            DH_matrix(q4, self.d4,  0,       -90),
            DH_matrix(q5, self.d5,  0,        90),
            DH_matrix(q6, self.a6,  0,         0),
        ]
        frames = [np.eye(4)]
        T = np.eye(4)
        for A in matrices:
            T = T @ A
            frames.append(T.copy())
        return frames


# =====================================================================
# CLASE IK — q1,q2,q3 por geometría + q4,q5,q6 por Euler ZYZ
# =====================================================================
class InverseKinematics:
    def __init__(self, fk_model: ForwardKinematics):
        self.fk = fk_model

    def ik_solve(self, x, y, z, roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0):
        """
        Resuelve IK completa para posición (x,y,z) en mm
        y orientación (roll, pitch, yaw) en grados (convención ZYZ).

        Para el modelo planar simplificado (P3) llamar con
        roll=pitch=yaw=0 y los primeros 3 joints son suficientes.
        """
        d1 = self.fk.d1    # 134.65
        a2 = self.fk.a2    # -110.0
        a3 = self.fk.a3    # -96.0
        d4 = self.fk.d4    #  63.4
        d5 = self.fk.d5    #  75.05

        # ── POSICIÓN: primeros 3 joints ──────────────────────────
        # q1: ángulo de base
        q1_rad = np.arctan2(y, x)

        # proyección planar
        r  = np.sqrt(x**2 + y**2)
        zc = z - d1

        # ley de cosenos (codo abajo)
        cos_q3 = np.clip(
            (r**2 + zc**2 - a2**2 - a3**2) / (2*a2*a3),
            -1.0, 1.0
        )
        q3_rad = -np.arccos(cos_q3)

        alpha  = np.arctan2(zc, r)
        beta   = np.arctan2(a3*np.sin(q3_rad), a2 + a3*np.cos(q3_rad))
        q2_rad = alpha - beta

        q1 = np.degrees(q1_rad)
        q2 = np.degrees(q2_rad)
        q3 = np.degrees(q3_rad)

        # ── ORIENTACIÓN: joints 4,5,6 por Euler ZYZ ─────────────
        # Construimos la rotación deseada del efector final
        roll  = np.radians(roll_deg)
        pitch = np.radians(pitch_deg)
        yaw   = np.radians(yaw_deg)

        # Matriz de rotación deseada (ZYZ: yaw → pitch → roll)
        def Rz(a):
            return np.array([[np.cos(a),-np.sin(a),0],
                             [np.sin(a), np.cos(a),0],
                             [0,         0,        1]])
        def Ry(a):
            return np.array([[ np.cos(a),0,np.sin(a)],
                             [0,         1,0        ],
                             [-np.sin(a),0,np.cos(a)]])

        R_des = Rz(yaw) @ Ry(pitch) @ Rz(roll)

        # Rotación acumulada de los primeros 3 joints (R_0_3)
        T03 = (DH_matrix(q1, d1,  0,  90) @
               DH_matrix(q2,  0, a2,   0) @
               DH_matrix(q3,  0, a3,   0))
        R03 = T03[:3, :3]

        # Rotación que deben aportar joints 4,5,6
        # R_0_3 @ R_3_6 = R_des  →  R_3_6 = R_0_3.T @ R_des
        R36 = R03.T @ R_des

        # Extraer ángulos de Euler ZYZ de R_3_6
        # R_ZYZ = Rz(q4) @ Ry(q5) @ Rz(q6)
        # Elemento R[1,2] = sin(q5)*sin(q4), etc.
        sy = np.sqrt(R36[0,2]**2 + R36[1,2]**2)

        if sy > 1e-6:   # caso general
            q5 = np.arctan2(sy, R36[2,2])
            q4 = np.arctan2(R36[1,2]/sy,  R36[0,2]/sy)
            q6 = np.arctan2(R36[2,1]/sy, -R36[2,0]/sy)
        else:           # singularidad (q5 ≈ 0 o π)
            q5 = 0.0
            q4 = 0.0
            q6 = np.arctan2(-R36[1,0], R36[0,0])

        q4 = np.degrees(q4)
        q5 = np.degrees(q5)
        q6 = np.degrees(q6)

        return [round(q1,3), round(q2,3), round(q3,3),
                round(q4,3), round(q5,3), round(q6,3)]


# =====================================================================
# CLASE COLLISION CHECKER
# =====================================================================
class CollisionChecker:
    JOINT_LIMITS = [
        (-160, 160),
        (-160, 160),
        (-150, 150),
        (-145, 145),
        (-165, 165),
        (-180, 180),
    ]
    Z_MIN            = 20.0
    WORKSPACE_RADIUS = 280.0

    def __init__(self, fk=None):
        self._fk = fk or ForwardKinematics()

    def check_collision(self, joints):
        # Escenario 1: límites articulares
        for i, (q, (lo, hi)) in enumerate(zip(joints, self.JOINT_LIMITS)):
            if not (lo <= q <= hi):
                print(f"  ⚠ COLISIÓN [J{i+1}]: {q:.1f}° fuera de [{lo}°,{hi}°]")
                return True

        T  = self._fk.compute_fk(joints)
        px, py, pz = T[0,3], T[1,3], T[2,3]

        # Escenario 2: fuera del workspace
        if np.sqrt(px**2 + py**2 + pz**2) > self.WORKSPACE_RADIUS:
            print(f"  ⚠ COLISIÓN [workspace]: ({px:.1f},{py:.1f},{pz:.1f})")
            return True

        # Escenario 3: colisión con mesa
        if pz < self.Z_MIN:
            print(f"  ⚠ COLISIÓN [mesa]: Z={pz:.1f} mm")
            return True

        # Escenario 4: singularidad de muñeca
        if abs(joints[4]) < 5.0 and abs(joints[3]) > 150.0:
            print(f"  ⚠ COLISIÓN [muñeca singular]: J4={joints[3]:.1f}°, J5={joints[4]:.1f}°")
            return True

        return False


# =====================================================================
# TEST FK  (P2)
# =====================================================================
fk = ForwardKinematics()
ik = InverseKinematics(fk)
cc = CollisionChecker(fk)

cinco_configuraciones = [
    [ 0.0,   0.0,   0.0,  0.0,  0.0,  0.0],
    [30.0,  15.0, -20.0,  0.0, 45.0,  0.0],
    [-45.0,-10.0,  30.0, 15.0,-30.0, 10.0],
    [ 0.0,  45.0, -45.0,  0.0, 90.0,  0.0],
    [15.0, -30.0,  45.0,-15.0,  0.0, 25.0]
]

print("CONFIGURACIÓN | ERROR X (mm) | ERROR Y (mm) | ERROR Z (mm)")
print("-"*65)

for idx, angulos in enumerate(cinco_configuraciones, 1):
    mc.send_angles(angulos, 30)
    time.sleep(3.5)
    coords_robot = mc.get_coords()

    T = fk.compute_fk(angulos)
    mi_x, mi_y, mi_z = T[0,3], T[1,3], T[2,3]

    if coords_robot:
        err_x = abs(mi_x - coords_robot[0])
        err_y = abs(mi_y - coords_robot[1])
        err_z = abs(mi_z - coords_robot[2])
        print(f"Config {idx}     | {err_x:12.3f} | {err_y:12.3f} | {err_z:12.3f}")
    else:
        print(f"Config {idx}     | Error al leer datos del robot físico.")


# =====================================================================
# TEST IK  (P3)
# =====================================================================
tres_posiciones = [
    [140.0,  0.0, 220.0],
    [120.0, 60.0, 180.0],
    [150.0,-40.0, 200.0]
]

print("\nPOSICIÓN OBJETIVO   | ERROR q1 (°) | ERROR q2 (°) | ERROR q3 (°)")
print("-"*65)

for idx, (x, y, z) in enumerate(tres_posiciones, 1):
    q_calc = ik.ik_solve(x, y, z)  # orientación 0,0,0 por simplificación planar

    if cc.check_collision(q_calc):
        print(f"Pos {idx} {[x,y,z]} | COLISIÓN – omitido")
        continue

    mc.send_coords([x, y, z, 0.0, 0.0, 0.0], 30, 1)
    time.sleep(3.5)
    angulos_robot = mc.get_angles()

    if angulos_robot:
        err_q1 = abs(q_calc[0] - angulos_robot[0])
        err_q2 = abs(q_calc[1] - angulos_robot[1])
        err_q3 = abs(q_calc[2] - angulos_robot[2])
        print(f"Pos {idx} {[x,y,z]} | {err_q1:12.3f} | {err_q2:12.3f} | {err_q3:12.3f}")
    else:
        print(f"Pos {idx} {[x,y,z]} | Error al leer los ángulos de los motores.")

# Test definitivo de parámetros reales
import numpy as np

# Con estos datos que ya tenemos:
# [0,90,0,0,0,0] -> Z=173.5  => d1=173.5
# [0,0,0,0,0,0]  -> Z=409.8  => d1+a2+a3+muñeca=409.8
# [0,0,90,0,0,0] -> Z=286.9  => d1+a2=286.9 => a2=113.4

# Verificamos cuánto da la FK con estos valores
def dh(t,d,a,al):
    t,al=np.radians(t),np.radians(al)
    ct,st,ca,sa=np.cos(t),np.sin(t),np.cos(al),np.sin(al)
    return np.array([[ct,-st*ca,st*sa,a*ct],[st,ct*ca,-ct*sa,a*st],[0,sa,ca,d],[0,0,0,1]])

d1=173.5; a2=113.4; a3=96.0; d4=63.4; d5=75.05; d6=51.8

T = dh(0,d1,0,90) @ dh(0-90,0,a2,0) @ dh(0,0,a3,0) @ \
    dh(0-90,d4,0,-90) @ dh(0,d5,0,90) @ dh(0,d6,0,0)

print(f"FK home: ({T[0,3]:.1f}, {T[1,3]:.1f}, {T[2,3]:.1f})")
print(f"Real:    (50.3, -63.3, 409.8)")
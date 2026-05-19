import time
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
 
sp.init_printing(use_unicode=True)
 
# ══════════════════════════════════════════════════════════════════
# PARÁMETROS DH  (tabla oficial)
#
#  Joint | a_i (mm) | d_i (mm) | alpha_i | offset hardware
#  J1    |    0     | 131.56   |  +90°   |    0°
#  J2    |  110.4   |   0      |    0°   |  -90°
#  J3    |   96.0   |   0      |    0°   |    0°
#  J4    |    0     |  66.39   |  -90°   |  -90°
#  J5    |    0     |  73.18   |  +90°   |    0°
#  J6    |    0     |  48.60   |    0°   |    0°
#
# q_dh = q_robot + offset
# Validado con mc.get_coords() en 7 configuraciones reales.
# ══════════════════════════════════════════════════════════════════
 
OFFSETS = [0.0, -90.0, 0.0, -90.0, 0.0, 0.0]
 
 
# ──────────────────────────────────────────────────────────────────
# Función base: Matriz DH
# ──────────────────────────────────────────────────────────────────
def dh_matrix(theta_deg, d, a, alpha_deg):
    t  = np.radians(theta_deg)
    al = np.radians(alpha_deg)
    ct, st = np.cos(t),  np.sin(t)
    ca, sa = np.cos(al), np.sin(al)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [ 0,     sa,     ca,    d],
        [ 0,      0,      0,    1]
    ], dtype=float)
 
 
def dh_matrix_sym(theta, d, a, alpha):
    """Versión simbólica (para reporte P1)."""
    ct, st = sp.cos(theta), sp.sin(theta)
    ca, sa = sp.cos(alpha), sp.sin(alpha)
    return sp.Matrix([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [ 0,     sa,     ca,    d],
        [ 0,      0,      0,    1]
    ])
 
 
# ──────────────────────────────────────────────────────────────────
# Matrices simbólicas (P1 – reporte)
# ──────────────────────────────────────────────────────────────────
θ1, θ2, θ3, θ4, θ5, θ6 = sp.symbols('θ1 θ2 θ3 θ4 θ5 θ6')
 
A1_sym = dh_matrix_sym(θ1,              131.56,   0,    sp.pi/2)
A2_sym = dh_matrix_sym(θ2 - sp.pi/2,       0,  110.4,       0)
A3_sym = dh_matrix_sym(θ3,                  0,   96.0,       0)
A4_sym = dh_matrix_sym(θ4 - sp.pi/2,    66.39,    0,  -sp.pi/2)
A5_sym = dh_matrix_sym(θ5,              73.18,    0,   sp.pi/2)
A6_sym = dh_matrix_sym(θ6,              48.60,    0,         0)
 
T_sym = A1_sym * A2_sym * A3_sym * A4_sym * A5_sym * A6_sym
 
 
# ══════════════════════════════════════════════════════════════════
#  CLASE 1 – ForwardKinematics
# ══════════════════════════════════════════════════════════════════
class ForwardKinematics:
    d1 = 131.56
    a2 = 110.4
    a3 = 96.0
    d4 = 66.39
    d5 = 73.18
    d6 = 48.6
 
    def compute_fk(self, joints_deg):
        """
        joints_deg: [q1..q6] en grados (igual que send_angles).
        Retorna matriz T_0_6 (4x4 numpy).
        """
        q = [joints_deg[i] + OFFSETS[i] for i in range(6)]
 
        A1 = dh_matrix(q[0], self.d1,  0,      90)
        A2 = dh_matrix(q[1],  0,    self.a2,    0)
        A3 = dh_matrix(q[2],  0,    self.a3,    0)
        A4 = dh_matrix(q[3], self.d4,  0,     -90)
        A5 = dh_matrix(q[4], self.d5,  0,      90)
        A6 = dh_matrix(q[5], self.d6,  0,       0)
 
        return A1 @ A2 @ A3 @ A4 @ A5 @ A6
 
 
# ══════════════════════════════════════════════════════════════════
#  CLASE 2 – InverseKinematics
# ══════════════════════════════════════════════════════════════════
class InverseKinematics:
    def __init__(self, fk):
        self.fk = fk
 
    def ik_solve(self, x, y, z):
        """
        Resuelve IK para el punto (x, y, z) en mm.
        Retorna [q1, q2, q3, 0, 0, 0] en grados (robot convention).
        """
        d1 = self.fk.d1
        a2 = self.fk.a2
        a3 = self.fk.a3
        d4 = self.fk.d4
        d5 = self.fk.d5
        d6 = self.fk.d6
 
        # Con muñeca neutra (q4=q5=q6=0), el gripper está
        # d4+d5+d6 = 188.17 mm más lejos en dirección radial.
        # Descontamos ese offset para obtener el centro de muñeca.
        wrist_offset = d4 + d5 + d6  # 188.17 mm
 
        r_total = np.sqrt(x**2 + y**2)
        r_w     = r_total - wrist_offset
        z_w     = z - d1
 
        # q1: ángulo de base
        q1_rad = np.arctan2(y, x)
 
        # q3: ley de cosenos
        D = (r_w**2 + z_w**2 - a2**2 - a3**2) / (2.0 * a2 * a3)
        D = np.clip(D, -1.0, 1.0)
        q3_rad = -np.arccos(D)  # elbow-down
 
        # q2: ángulo de hombro
        alpha  = np.arctan2(z_w, r_w)
        beta   = np.arctan2(a3 * np.sin(q3_rad),
                            a2 + a3 * np.cos(q3_rad))
        q2_rad = alpha - beta
 
        # Grados + inverso del offset hardware
        q1 = np.degrees(q1_rad)
        q2 = np.degrees(q2_rad) + 90.0
        q3 = np.degrees(q3_rad)
 
        return [round(q1, 3), round(q2, 3), round(q3, 3), 0.0, 0.0, 0.0]
 
 
# ══════════════════════════════════════════════════════════════════
#  CLASE 3 – CollisionChecker
# ══════════════════════════════════════════════════════════════════
class CollisionChecker:
    """
    Escenarios:
      1. Límites articulares excedidos.
      2. Efector fuera del espacio de trabajo (r > 270 mm).
      3. Colisión con la mesa (Z < 20 mm).
      4. Singularidad de muñeca (J4 extremo + J5 ≈ 0°).
    """
    JOINT_LIMITS = [
        (-168, 168),
        (-135,  90),
        (-150, 150),
        (-145, 145),
        (-165, 165),
        (-180, 180),
    ]
    WORKSPACE_RADIUS = 270.0
    Z_MIN = 20.0
 
    def __init__(self, fk=None):
        self._fk = fk or ForwardKinematics()
 
    def check_collision(self, joints):
        """Retorna True si hay colisión e imprime el motivo."""
        # Escenario 1: límites articulares
        for i, (q, (lo, hi)) in enumerate(zip(joints, self.JOINT_LIMITS)):
            if not (lo <= q <= hi):
                print(f"  ⚠ COLISIÓN [J{i+1}]: {q:.1f}° fuera de [{lo}°, {hi}°]")
                return True
 
        T  = self._fk.compute_fk(joints)
        px, py, pz = T[0,3], T[1,3], T[2,3]
        r_3d = np.sqrt(px**2 + py**2 + pz**2)
 
        # Escenario 2: fuera del workspace
        if r_3d > self.WORKSPACE_RADIUS:
            print(f"  ⚠ COLISIÓN [workspace]: r={r_3d:.1f} mm > {self.WORKSPACE_RADIUS} mm")
            return True
 
        # Escenario 3: colisión con la mesa
        if pz < self.Z_MIN:
            print(f"  ⚠ COLISIÓN [mesa]: Z={pz:.1f} mm < {self.Z_MIN} mm")
            return True
 
        # Escenario 4: singularidad de muñeca
        if abs(joints[4]) < 5.0 and abs(joints[3]) > 150.0:
            print(f"  ⚠ COLISIÓN [muñeca singular]: J4={joints[3]:.1f}°, J5={joints[4]:.1f}°")
            return True
 
        return False
 
    def safe_move(self, joints, mc=None, speed=30):
        """Verifica colisión y si es seguro envía al robot."""
        if self.check_collision(joints):
            print("  → Movimiento cancelado.")
            return False
        if mc is not None:
            mc.send_angles(joints, speed)
        return True
 
 
# ══════════════════════════════════════════════════════════════════
#  GRÁFICA WORKSPACE XZ
# ══════════════════════════════════════════════════════════════════
def plot_workspace_xz(n_samples=4000, save=True):
    fk  = ForwardKinematics()
    rng = np.random.default_rng(42)
    pts = []
 
    q2_vals = np.linspace(-135,  90, 30)
    q3_vals = np.linspace(-150, 150, 30)
    q4_vals = np.linspace( -90,  90,  5)
    q5_vals = np.linspace( -90,  90,  5)
 
    for _ in range(n_samples):
        q = [
            0.0,
            float(rng.choice(q2_vals)),
            float(rng.choice(q3_vals)),
            float(rng.choice(q4_vals)),
            float(rng.choice(q5_vals)),
            0.0,
        ]
        T = fk.compute_fk(q)
        pts.append((T[0,3], T[2,3]))
 
    xs, zs = zip(*pts)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(xs, zs, s=1, alpha=0.35, color='#00c8ff')
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Z (mm)')
    ax.set_title('Espacio de trabajo alcanzable – Plano XZ\nMyCobot 280')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.axvline(0, color='gray', lw=0.8, ls='--')
    plt.tight_layout()
    if save:
        plt.savefig('workspace_xz.png', dpi=150)
        print("Guardada: workspace_xz.png")
    plt.show()
 
 
# ══════════════════════════════════════════════════════════════════
#  TEST FK  (P2)
# ══════════════════════════════════════════════════════════════════
fk = ForwardKinematics()
 
cinco_configuraciones = [
    [ 0.0,   0.0,   0.0,  0.0,  0.0,  0.0],
    [30.0,  15.0, -20.0,  0.0, 45.0,  0.0],
    [-45.0,-10.0,  30.0, 15.0,-30.0, 10.0],
    [ 0.0,  45.0, -45.0,  0.0, 90.0,  0.0],
    [15.0, -30.0,  45.0,-15.0,  0.0, 25.0],
]
 
print("CONFIGURACIÓN | ERROR X (mm) | ERROR Y (mm) | ERROR Z (mm)")
print("-" * 65)
 
for idx, q in enumerate(cinco_configuraciones, 1):
    mc.send_angles(q, 30)
    time.sleep(3.0)
    robot_coords = mc.get_coords()
 
    T = fk.compute_fk(q)
    mx, my, mz = T[0,3], T[1,3], T[2,3]
 
    if robot_coords:
        ex = abs(mx - robot_coords[0])
        ey = abs(my - robot_coords[1])
        ez = abs(mz - robot_coords[2])
        print(f"Config {idx}     | {ex:12.3f} | {ey:12.3f} | {ez:12.3f}")
    else:
        print(f"Config {idx}     | Error al leer datos del robot físico.")
 
 
# ══════════════════════════════════════════════════════════════════
#  TEST IK  (P3)
# ══════════════════════════════════════════════════════════════════
ik = InverseKinematics(fk)
cc = CollisionChecker(fk)
 
tres_posiciones = [
    [140.0,   0.0, 220.0],
    [120.0,  60.0, 180.0],
    [150.0, -40.0, 200.0],
]
 
print("\nPOSICIÓN OBJETIVO | ERROR q1 (°) | ERROR q2 (°) | ERROR q3 (°)")
print("-" * 65)
 
for idx, (x, y, z) in enumerate(tres_posiciones, 1):
    q_calc = ik.ik_solve(x, y, z)
 
    if cc.check_collision(q_calc):
        print(f"Pos {idx} {[x,y,z]} | COLISIÓN – omitido")
        continue
 
    mc.send_angles(q_calc, 30)
    time.sleep(3.5)
    q_robot = mc.get_angles()
 
    if q_robot:
        e1 = abs(q_calc[0] - q_robot[0])
        e2 = abs(q_calc[1] - q_robot[1])
        e3 = abs(q_calc[2] - q_robot[2])
        print(f"Pos {idx} {str([x,y,z]):18} | {e1:12.3f} | {e2:12.3f} | {e3:12.3f}")
    else:
        print(f"Pos {idx} {str([x,y,z]):18} | Error al leer ángulos.")
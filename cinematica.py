# =====================================================================
# CINEMÁTICA MyCobot 280 - FK, IK y Verificación con API
# =====================================================================
import numpy as np
import time

# =====================================================================
# PARÁMETROS DH (numéricos)
# =====================================================================
# Tabla DH: [theta_offset, d, a, alpha]
DH_PARAMS = [
    [0,  0,    0,          0],   # Joint 1
    [0,    134.65,    0,  np.pi/2],   # Joint 2
    [-np.pi/2,    0,     -110.0,  0      ],   # Joint 3
    [0,   0,    -96.0,    0       ],   # Joint 4
    [np.pi/2,   63.4,   0,    -np.pi/2],   # Joint 5
    [np.pi/2,   75.05,    51.8,     np.pi/2],   # Joint 6
]

# =====================================================================
# CLASE: FORWARD KINEMATICS
# =====================================================================
class ForwardKinematics:
    def __init__(self, dh_params):
        self.dh = dh_params

    def _dh_matrix(self, theta, d, a, alpha):
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [ 0,     sa,    ca,     d],
            [ 0,      0,     0,     1]
        ])

    def fk(self, joint_angles_deg):
        """
        Calcula FK completa.
        joint_angles_deg: lista de 6 ángulos en grados.
        Retorna: T (matriz 4x4), posición xyz en mm, RPY en grados.
        """
        angles_rad = np.radians(joint_angles_deg)
        T = np.eye(4)
        for i, (dh, q) in enumerate(zip(self.dh, angles_rad)):
            theta_off, d, a, alpha = dh
            T = T @ self._dh_matrix(theta_off + q, d, a, alpha)

        x, y, z = T[0,3], T[1,3], T[2,3]

        # Extraer RPY (roll-pitch-yaw) de la matriz de rotación R = T[:3,:3]
        R = T[:3, :3]
        pitch = np.arctan2(-R[2,0], np.sqrt(R[0,0]**2 + R[1,0]**2))
        if abs(np.cos(pitch)) > 1e-6:
            roll  = np.arctan2(R[2,1], R[2,2])
            yaw   = np.arctan2(R[1,0], R[0,0])
        else:  # singularidad gimbal lock
            roll  = 0.0
            yaw   = np.arctan2(-R[0,1], R[1,1])

        rpy_deg = np.degrees([roll, pitch, yaw])
        return T, np.array([x, y, z]), rpy_deg


# =====================================================================
# CLASE: INVERSE KINEMATICS (analítica, 3 joints + Euler para 4-5-6)
# =====================================================================
class InverseKinematics:
    def __init__(self, dh_params):
        self.dh = dh_params
        self.fk_solver = ForwardKinematics(dh_params)

        # Longitudes de eslabón para modelo planar 2R + base (joints 1-3)
        self.d1  = dh_params[1][1]   # 134.65  (altura base)
        self.a2  = abs(dh_params[2][2])  # 110.0
        self.a3  = abs(dh_params[3][2])  #  96.0
        # Offset vertical del muñeca (d4)
        self.d4  = dh_params[4][1]   # 63.4

    def ik_solve(self, x, y, z, rx=0.0, ry=0.0, rz=0.0):
        """
        IK analítica simplificada.
        - Joints 1-3: modelo planar 2R proyectado.
        - Joints 4-6: orientación con ángulos de Euler ZYX (rx,ry,rz en grados).
        Retorna: [j1,j2,j3,j4,j5,j6] en grados, o None si fuera de rango.
        """
        # --- Joint 1: rotación en plano XY ---
        j1 = np.degrees(np.arctan2(y, x))

        # --- Proyección al plano de trabajo (distancia radial - offset muñeca) ---
        r_xy = np.sqrt(x**2 + y**2)
        # Altura efectiva descontando base y offset muñeca
        z_eff = z - self.d1
        r_eff = r_xy  # simplificado: muñeca en línea con end-effector

        # --- Joints 2 y 3: modelo planar 2R ---
        L1, L2 = self.a2, self.a3
        D = (r_eff**2 + z_eff**2 - L1**2 - L2**2) / (2 * L1 * L2)

        if abs(D) > 1.0:
            print(f"  [IK] Posición ({x:.1f},{y:.1f},{z:.1f}) fuera de alcance.")
            return None

        # Codo abajo (elbow-down)
        j3 = np.degrees(np.arctan2(-np.sqrt(1 - D**2), D))

        alpha = np.arctan2(z_eff, r_eff)
        beta  = np.arctan2(L2 * np.sin(np.radians(j3)),
                           L1 + L2 * np.cos(np.radians(j3)))
        j2 = np.degrees(alpha - beta)

        # --- Joints 4-6: orientación deseada (Euler ZYX) ---
        # El robot MyCobot 280 acepta rx,ry,rz como orientación del EE.
        # Aquí los pasamos directamente como muñeca (simplificado).
        j4 = rx
        j5 = ry
        j6 = rz

        return [round(j1,2), round(j2,2), round(j3,2),
                round(j4,2), round(j5,2), round(j6,2)]


# =====================================================================
# CLASE: COLLISION CHECKER (básico)
# =====================================================================
class CollisionChecker:
    """
    Escenario 1: límites de articulación excedidos.
    Escenario 2: posición del EE dentro de zona prohibida (cubo alrededor de base).
    """
    JOINT_LIMITS = [(-165,165),(-165,165),(-165,165),
                    (-165,165),(-165,165),(-175,175)]
    FORBIDDEN_ZONE = {'x': (-60,60), 'y': (-60,60), 'z': (0, 80)}  # mm

    def check_joints(self, angles_deg):
        for i, (ang, (lo, hi)) in enumerate(zip(angles_deg, self.JOINT_LIMITS)):
            if not (lo <= ang <= hi):
                return True, f"Joint {i+1} ({ang}°) fuera de límites [{lo},{hi}]"
        return False, "OK"

    def check_position(self, xyz):
        x, y, z = xyz
        fz = self.FORBIDDEN_ZONE
        if (fz['x'][0] < x < fz['x'][1] and
            fz['y'][0] < y < fz['y'][1] and
            fz['z'][0] < z < fz['z'][1]):
            return True, f"EE en zona prohibida: ({x:.1f},{y:.1f},{z:.1f})"
        return False, "OK"

    def evasion(self, angles_deg, xyz):
        """Retorna True si hay colisión (detiene movimiento)."""
        col_j, msg_j = self.check_joints(angles_deg)
        col_p, msg_p = self.check_position(xyz)
        if col_j:
            print(f"  [COLISIÓN-Joints] {msg_j}")
            return True
        if col_p:
            print(f"  [COLISIÓN-Zona]   {msg_p}")
            return True
        return False


# =====================================================================
# INSTANCIAR
# =====================================================================
fk_solver  = ForwardKinematics(DH_PARAMS)
ik_solver  = InverseKinematics(DH_PARAMS)
collision  = CollisionChecker()

print("Clases instanciadas: ForwardKinematics, InverseKinematics, CollisionChecker ✓")


# =====================================================================
# P2 - VERIFICACIÓN FK: 5 configuraciones calculada vs. real
# =====================================================================
configs_deg = [
    [ 0,   0,   0,  0,  0,  0],
    [30, -30,  45,  0,  0,  0],
    [60,  20, -30, 10,  5, 15],
    [-45, 10,  60, -5, 10,  0],
    [90, -45,  30, 20,-10, 30],
]

print("="*70)
print(f"{'Config':>6} | {'Calc X':>8} {'Calc Y':>8} {'Calc Z':>8} | {'Real X':>8} {'Real Y':>8} {'Real Z':>8} | {'Err mm':>8}")
print("="*70)

tabla_fk = []

for i, cfg in enumerate(configs_deg):
    # 1. Enviar ángulos al robot
    mc.send_angles(cfg, 30)
    time.sleep(3)

    # 2. FK calculada
    T, pos_calc, rpy_calc = fk_solver.fk(cfg)

    # 3. Posición real del robot
    coords_real = mc.get_coords()
    if coords_real is None or len(coords_real) < 3:
        print(f"  Config {i+1}: no se pudo leer coords. Reintentando...")
        time.sleep(1)
        coords_real = mc.get_coords()

    pos_real = np.array(coords_real[:3]) if coords_real else np.array([0,0,0])

    # 4. Error euclidiano
    error_mm = np.linalg.norm(pos_calc - pos_real)

    tabla_fk.append({
        'config': cfg,
        'calc':   pos_calc.tolist(),
        'real':   pos_real.tolist(),
        'error':  error_mm
    })

    print(f"  C{i+1}   | {pos_calc[0]:8.2f} {pos_calc[1]:8.2f} {pos_calc[2]:8.2f} "
          f"| {pos_real[0]:8.2f} {pos_real[1]:8.2f} {pos_real[2]:8.2f} "
          f"| {error_mm:8.2f}")

print("="*70)



# =====================================================================
# P3 - VERIFICACIÓN IK: 3 posiciones cartesianas calculada vs. real
# =====================================================================
# Posiciones objetivo [x, y, z, rx, ry, rz] en mm y grados
targets = [
    [150,  50, 200,  0,  0,  0],
    [100, -80, 150,  0, 30,  0],
    [200,   0, 100,  0, 45, 45],
]

print("="*80)
print(f"{'Target':>5} | {'J1':>6}{'J2':>6}{'J3':>6}{'J4':>6}{'J5':>6}{'J6':>6} | "
      f"{'Err J1':>7}{'Err J2':>7}{'Err J3':>7}{'Err J4':>7}{'Err J5':>7}{'Err J6':>7}")
print("="*80)

tabla_ik = []

for i, tgt in enumerate(targets):
    x, y, z, rx, ry, rz = tgt

    # 1. Resolver IK
    angles_calc = ik_solver.ik_solve(x, y, z, rx, ry, rz)
    if angles_calc is None:
        print(f"  T{i+1}: IK sin solución.")
        continue

    # 2. Verificar colisiones antes de mover
    _, pos_check, _ = fk_solver.fk(angles_calc)
    if collision.evasion(angles_calc, pos_check):
        print(f"  T{i+1}: movimiento bloqueado por colisión.")
        continue

    # 3. Enviar posición al robot (IK del robot)
    mc.send_coords([x, y, z, rx, ry, rz], 30, 1)
    time.sleep(3)

    # 4. Leer ángulos reales que adoptó el robot
    angles_real = mc.get_angles()
    if angles_real is None or len(angles_real) < 6:
        time.sleep(1)
        angles_real = mc.get_angles()

    angles_real = list(angles_real) if angles_real else [0]*6

    # 5. Error por joint (grados)
    errors = [abs(c - r) for c, r in zip(angles_calc, angles_real)]

    tabla_ik.append({
        'target': tgt,
        'calc':   angles_calc,
        'real':   angles_real,
        'errors': errors
    })

    print(f"  T{i+1}  | "
          + "".join(f"{a:6.1f}" for a in angles_calc)
          + " | "
          + "".join(f"{e:7.2f}" for e in errors))

print("="*80)



# =====================================================================
# ESPACIO DE TRABAJO - Nube de puntos 2D en plano XZ
# =====================================================================
import matplotlib.pyplot as plt

print("Calculando espacio de trabajo (plano XZ)...")

puntos_x, puntos_z = [], []

# Barrido de joints 1-3 (los más relevantes para alcance)
for j1 in np.linspace(-150, 150, 15):
    for j2 in np.linspace(-120, 120, 15):
        for j3 in np.linspace(-120, 120, 15):
            cfg = [j1, j2, j3, 0, 0, 0]
            try:
                _, pos, _ = fk_solver.fk(cfg)
                puntos_x.append(pos[0])
                puntos_z.append(pos[2])
            except:
                pass

plt.figure(figsize=(8,8))
plt.scatter(puntos_x, puntos_z, s=1, alpha=0.3, color='steelblue')
plt.xlabel('X (mm)')
plt.ylabel('Z (mm)')
plt.title('Espacio de trabajo alcanzable - Plano XZ\nMyCobot 280')
plt.axis('equal')
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('espacio_trabajo_XZ.png', dpi=150)
plt.show()
print("Gráfica guardada: espacio_trabajo_XZ.png")
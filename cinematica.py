import numpy as np
import pandas as pd
import math
import time

DH_PARAMS = [
    [0, 0, 0],          # θ1 -> se reemplaza con ángulo real
    [0, 134.65, 90],    # θ2
    [110, 0, 0],        # θ3-90
    [-96, 0, 0],        # θ4
    [0, 63.4, -90],     # θ5+90
    [51.8, 75.05, 90]   # θ6+90
]

kinematic_fk_rows = []
kinematic_ik_rows = []

def dh_matrix(a, d, alpha_deg, theta_deg):
    alpha = np.radians(alpha_deg)
    theta = np.radians(theta_deg)

    return np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
        [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])

def fk_mycobot(angles):
    T = np.eye(4)

    for i in range(6):
        a, d, alpha = DH_PARAMS[i]
        theta = angles[i]
        T = T @ dh_matrix(a, d, alpha, theta)

    return T, np.array([T[0, 3], T[1, 3], T[2, 3]])

def position_error_mm(calc_pos, real_coords):
    real_pos = np.array(real_coords[:3])
    diff = real_pos - calc_pos
    error = np.linalg.norm(diff)
    return diff, error

def ik_analytic_3dof(x, y, z, elbow=1):
    L2 = 110.4
    L3 = 96.0
    d1 = 131.56

    theta1 = math.degrees(math.atan2(y, x))

    r = math.sqrt(x**2 + y**2)
    zp = z - d1

    cos_theta3 = (r**2 + zp**2 - L2**2 - L3**2) / (2 * L2 * L3)

    if cos_theta3 < -1 or cos_theta3 > 1:
        return None

    sin_theta3 = elbow * math.sqrt(1 - cos_theta3**2)
    theta3 = math.degrees(math.atan2(sin_theta3, cos_theta3))

    theta2 = math.degrees(
        math.atan2(zp, r) - math.atan2(L3 * sin_theta3, L2 + L3 * cos_theta3)
    )

    return [theta1, theta2, theta3]

def best_ik_analytic(x, y, z, api_angles):
    sol1 = ik_analytic_3dof(x, y, z, elbow=1)
    sol2 = ik_analytic_3dof(x, y, z, elbow=-1)

    candidates = []

    for sol in [sol1, sol2]:
        if sol is not None:
            diff = np.array(api_angles[:3]) - np.array(sol)
            error = np.linalg.norm(diff)
            candidates.append((sol, diff, error))

    if len(candidates) == 0:
        return None, None, None

    return min(candidates, key=lambda item: item[2])

def record_fk(label):
    angles_real = mc.get_angles()
    coords_real = mc.get_coords()

    T, fk_pos = fk_mycobot(angles_real)
    diff, error = position_error_mm(fk_pos, coords_real)

    row = {
        "pose": label,
        "angulos_get_angles": angles_real,
        "coords_get_coords": coords_real,
        "fk_x": round(fk_pos[0], 2),
        "fk_y": round(fk_pos[1], 2),
        "fk_z": round(fk_pos[2], 2),
        "real_x": round(coords_real[0], 2),
        "real_y": round(coords_real[1], 2),
        "real_z": round(coords_real[2], 2),
        "error_x_mm": round(diff[0], 2),
        "error_y_mm": round(diff[1], 2),
        "error_z_mm": round(diff[2], 2),
        "error_total_mm": round(error, 2)
    }

    kinematic_fk_rows.append(row)
    print("FK registrado:", label, "error:", round(error, 2), "mm")

def record_ik(label, target_coords):
    x, y, z = target_coords[:3]
    api_angles = mc.get_angles()
    coords_real = mc.get_coords()

    ik_calc, diff, error = best_ik_analytic(x, y, z, api_angles)

    if ik_calc is None:
        row = {
            "pose": label,
            "target_coords": target_coords,
            "ik_analitica": "fuera de alcance",
            "api_get_angles": api_angles,
            "error_total_grados": "no calculado"
        }
    else:
        row = {
            "pose": label,
            "target_x": round(x, 2),
            "target_y": round(y, 2),
            "target_z": round(z, 2),
            "theta1_ik": round(ik_calc[0], 2),
            "theta2_ik": round(ik_calc[1], 2),
            "theta3_ik": round(ik_calc[2], 2),
            "theta1_api": round(api_angles[0], 2),
            "theta2_api": round(api_angles[1], 2),
            "theta3_api": round(api_angles[2], 2),
            "error_theta1": round(diff[0], 2),
            "error_theta2": round(diff[1], 2),
            "error_theta3": round(diff[2], 2),
            "error_total_grados": round(error, 2),
            "coords_reales_get_coords": coords_real
        }

    kinematic_ik_rows.append(row)
    print("IK registrado:", label)


#segunda parte

def goto_pose_kinematic(label, pose, mode="coords", record_ik_flag=False):
    goto_pose(pose, mode=mode)
    time.sleep(0.5)

    record_fk(label)

    if record_ik_flag and mode == "coords":
        record_ik(label, pose)

def run_cycle_vision_kinematics(cycle):
    result = capturar_resultado_fresco()

    if result is None:
        print("No se detectó cubo")
        return False

    color = result["color"].strip().lower()

    destino = get_place_by_color(color)

    if destino is None:
        print("Color no reconocido:", color)
        return False

    x_place, y_place = destino
    x_pick, y_pick = vision_result_to_pick_robot(result)

    pick_above = make_coords(x_pick, y_pick, Z_UP)
    pick_down = make_coords(x_pick, y_pick, Z_PICK)

    place_above = make_coords(x_place, y_place, Z_UP)
    place_down = make_coords(x_place, y_place, Z_PLACE)

    print("========== CICLO CINEMÁTICA ==========")
    print("Color:", color)
    print("Pick:", x_pick, y_pick)
    print("Place:", x_place, y_place)
    print("======================================")

    goto_pose_kinematic("init_pose", INIT_ANGLES, mode="angles")
    open_gripper()

    goto_pose_kinematic("watch_pose", WATCH_ANGLES, mode="angles")

    goto_pose_kinematic("pick_above", pick_above, mode="coords", record_ik_flag=True)
    goto_pose_kinematic("pick_down", pick_down, mode="coords", record_ik_flag=True)

    close_gripper()
    time.sleep(0.8)

    goto_pose_kinematic("lift_after_pick", pick_above, mode="coords", record_ik_flag=True)

    goto_pose_kinematic("place_above_" + color, place_above, mode="coords", record_ik_flag=True)
    goto_pose_kinematic("place_down_" + color, place_down, mode="coords", record_ik_flag=True)

    open_gripper()
    time.sleep(0.8)

    goto_pose_kinematic("lift_after_place_" + color, place_above, mode="coords", record_ik_flag=True)
    goto_pose_kinematic("return_init", INIT_ANGLES, mode="angles")

    return True

#ejecutar pruebas

kinematic_fk_rows = []
kinematic_ik_rows = []

run_cycle_vision_kinematics(1)

df_fk_pipeline = pd.DataFrame(kinematic_fk_rows)
df_ik_pipeline = pd.DataFrame(kinematic_ik_rows)

df_fk_pipeline

#en otra celda
df_ik_pipeline



class CollisionChecker:
    def __init__(self, obstacles=[]):
        """
        obstacles: lista de obstáculos, cada uno definido como [x, y, z, radio]
        """
        self.obstacles = obstacles

    def add_obstacle(self, x, y, z, radius):
        self.obstacles.append([x, y, z, radius])

    def check_collision(self, x, y, z):
        """
        Retorna True si el punto (x,y,z) está dentro de algún obstáculo
        """
        for obs in self.obstacles:
            ox, oy, oz, r = obs
            if np.linalg.norm([x - ox, y - oy, z - oz]) <= r:
                return True
        return False


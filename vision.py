#!/usr/bin/env python3
# encoding: utf-8

import cv2 as cv
import numpy as np
import math
import logging
from datetime import datetime

# ════════════════════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s"
)
log = logging.getLogger("vision")

# ════════════════════════════════════════════════════════════════════════════
# RANGOS HSV CALIBRADOS
# Tomados directamente de HSV_config.txt del repositorio
# El Rol 4 debe verificar y ajustar estos valores en el laboratorio
# ════════════════════════════════════════════════════════════════════════════

COLOR_HSV = {
    "red":    ((0,   181,  46),  (10,  255, 234)),
    "green":  ((34,   99,  45),  (76,  255, 255)),
    "blue":   ((99,  129, 102),  (120, 255, 231)),
    "yellow": ((21,  172, 185),  (255, 255, 242)),
}


TARGET_COLOR = "red"

# ════════════════════════════════════════════════════════════════════════════
# PARÁMETROS DE CONVERSIÓN  píxel → metros → mm
# Tomados de grasp_controller.py del repositorio
# ════════════════════════════════════════════════════════════════════════════

# Offsets de calibración (del repositorio original)
OFFSET_X = -0.012    # metros
OFFSET_Y =  0.0005   # metros

# Área mínima del contorno para considerar detección válida
AREA_MINIMA = 1000   # píxeles²


# ════════════════════════════════════════════════════════════════════════════
# CLASE PRINCIPAL — basada en identify_GetTarget del repositorio
# ════════════════════════════════════════════════════════════════════════════

class identify_GetTarget:
    """
    Detecta objetos de color en un frame de cámara.
    Código basado en jetcobot_color_identify/identify_target.py
    """

    def __init__(self):
        self.image      = None
        self.color_name = None

    def get_Sqaure(self, color_hsv: tuple):
        """
        Detecta el objeto por color HSV y devuelve su posición.

        Parámetros
        ----------
        color_hsv : ((H_low, S_low, V_low), (H_high, S_high, V_high))

        Devuelve
        --------
        (a, b, yaw) en metros  |  None si no detecta nada
            a   = posición horizontal  (eje Y del robot)
            b   = posición frontal     (eje X del robot)
            yaw = ángulo de rotación del objeto en grados
        """
        try:
            (lowerb, upperb) = color_hsv

            # Máscara de color
            mask    = self.image.copy()
            hsv_img = cv.cvtColor(self.image, cv.COLOR_BGR2HSV)
            img     = cv.inRange(hsv_img, lowerb, upperb)
            mask[img == 0] = [0, 0, 0]

            # Morfología para limpiar ruido
            kernel  = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
            dst_img = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

            # Binarización
            dst_img       = cv.cvtColor(dst_img, cv.COLOR_RGB2GRAY)
            _, binary     = cv.threshold(dst_img, 10, 255, cv.THRESH_BINARY)

            # Encontrar contornos
            find_contours = cv.findContours(
                binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )
            contours = find_contours[1] if len(find_contours) == 3 \
                       else find_contours[0]

            if not contours:
                return None

            # Contorno más grande → ángulo de rotación (yaw)
            c    = max(contours, key=cv.contourArea)
            rect = cv.minAreaRect(c)
            yaw  = rect[2]

            # Dibujar rectángulo mínimo en el frame (visible en el video)
            corners = np.int64(cv.boxPoints(rect))
            cv.drawContours(self.image, [corners], 0, (255, 0, 0), 3)

            # Calcular centroide de cada contorno válido
            for cnt in contours:
                x, y, w, h = cv.boundingRect(cnt)
                area        = cv.contourArea(cnt)

                if area > AREA_MINIMA:
                    # Centroide en píxeles
                    cx = float(x + w / 2)
                    cy = float(y + h / 2)

                    # Dibujar centroide y etiqueta en el frame
                    cv.circle(self.image, (int(cx), int(cy)),
                              5, (0, 0, 255), -1)
                    cv.putText(self.image, self.color_name,
                               (int(x - 15), int(y - 15)),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

                    # ── Conversión píxel → metros ────────────────────────
                    # Fórmula exacta del repositorio identify_target.py
                    # Imagen normalizada a 640×480
                    a = round((cx - 320) / 4000, 5)
                    b = round((480 - cy) / 3000 * 0.7 + 0.15, 5)

                    log.debug("Centroide: (%.1f, %.1f) px → a=%.5f m, b=%.5f m",
                              cx, cy, a, b)
                    return (a, b, yaw)

        except Exception as e:
            log.error("get_Sqaure error: %s", e)
            return None

    def select_color(self, image: np.ndarray,
                     color_hsv: dict, color_list: dict):
        """
        Detecta todos los colores de color_list en el frame.

        Parámetros
        ----------
        image      : frame BGR de OpenCV (cualquier resolución)
        color_hsv  : dict {nombre: (lower, upper)} con rangos HSV
        color_list : dict {'1': 'red', '2': 'green', ...}

        Devuelve
        --------
        (frame_anotado, msg)
        msg = {nombre_color: (a, b, yaw)} — solo los detectados
        """
        # Normalizar a 640×480 (igual que el repositorio)
        self.image = cv.resize(image, (640, 480))
        msg = {}

        if not color_list:
            return self.image, msg

        # Orden de prioridad igual al repositorio original
        for key in ['4', '3', '2', '1']:
            if key in color_list:
                self.color_name = color_list[key]
                pos = self.get_Sqaure(color_hsv[self.color_name])
                if pos is not None:
                    msg[self.color_name] = pos

        return self.image, msg


# ════════════════════════════════════════════════════════════════════════════
# CONVERSIÓN metros → mm  (sistema de referencia del robot)
# ════════════════════════════════════════════════════════════════════════════

def metros_a_mm(a: float, b: float) -> tuple[float, float]:
    """
    Convierte la posición del objeto de metros a mm en el sistema del robot.

    El sistema de coordenadas del robot es:
        x_robot = dirección frontal (b en la imagen)
        y_robot = dirección lateral (-a en la imagen, eje invertido)

    Parámetros
    ----------
    a : posición horizontal de la imagen en metros
    b : posición frontal de la imagen en metros

    Devuelve
    --------
    (x_mm, y_mm) en el sistema de referencia del robot
    """
    x_mm = (b + OFFSET_X) * 1000
    y_mm = (-a + OFFSET_Y) * 1000
    return round(x_mm, 2), round(y_mm, 2)


# ════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL EXPORTADA — usada por main.py
# ════════════════════════════════════════════════════════════════════════════

_detector = identify_GetTarget()

def detect_object(frame: np.ndarray,
                  color: str = TARGET_COLOR) -> tuple[float, float] | None:
    """
    Detecta el objeto de color en el frame y devuelve su posición en mm.

    Parámetros
    ----------
    frame : imagen BGR capturada con cv2.VideoCapture
    color : nombre del color a detectar ("red", "green", "blue", "yellow")

    Devuelve
    --------
    (x_mm, y_mm) en el sistema del robot  |  None si no hay objeto
    """
    if color not in COLOR_HSV:
        log.error("Color '%s' no definido en COLOR_HSV.", color)
        return None

    color_list = {'1': color}
    _, msg     = _detector.select_color(frame, COLOR_HSV, color_list)

    if color not in msg:
        log.debug("Objeto '%s' no detectado.", color)
        return None

    a, b, yaw = msg[color]
    x_mm, y_mm = metros_a_mm(a, b)

    log.info("Objeto '%s': a=%.5f m, b=%.5f m → x=%.1f mm, y=%.1f mm, yaw=%.1f°",
             color, a, b, x_mm, y_mm, yaw)
    return x_mm, y_mm


# ════════════════════════════════════════════════════════════════════════════
# VALIDACIÓN — P6 paso 4: probar con 3 posiciones distintas del objeto
# Ejecutar directamente: python3 vision.py
# ════════════════════════════════════════════════════════════════════════════

def validar_deteccion():
    """
    Abre la cámara y muestra la detección en tiempo real.
    Registra las posiciones detectadas para la tabla de validación del examen.
    Presiona 'g' para guardar una medición, 'q' para salir.
    """
    cap = cv.VideoCapture(0)
    cap.set(6, cv.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    cap.set(cv.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        log.error("No se pudo abrir la cámara.")
        return

    log.info("Cámara abierta. Presiona 'g' para guardar posición, 'q' para salir.")
    mediciones = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detectar objeto
        resultado = detect_object(frame, TARGET_COLOR)

        # Mostrar frame anotado
        label = f"No detectado"
        if resultado:
            x_mm, y_mm = resultado
            label = f"x={x_mm:.1f} mm  y={y_mm:.1f} mm"

        cv.putText(frame, label, (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv.imshow("P6 - Validacion vision", frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g') and resultado:
            mediciones.append(resultado)
            log.info("Medicion %d guardada: x=%.1f mm, y=%.1f mm",
                     len(mediciones), resultado[0], resultado[1])

    cap.release()
    cv.destroyAllWindows()

    # Mostrar tabla de validación (P6 entregable)
    print("\n══ TABLA DE VALIDACIÓN P6 ══")
    print(f"{'#':>3} | {'x_mm':>8} | {'y_mm':>8}")
    print("-" * 28)
    for i, (x, y) in enumerate(mediciones):
        print(f"{i+1:>3} | {x:>8.1f} | {y:>8.1f}")
    print(f"\nTotal mediciones guardadas: {len(mediciones)}")


if __name__ == "__main__":
    validar_deteccion()
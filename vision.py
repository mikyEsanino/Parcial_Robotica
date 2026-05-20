#!/usr/bin/env python3
# encoding: utf-8

import cv2 as cv
import numpy as np
import logging

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
# Actualizar con valores del HSV_calibration.ipynb en laboratorio
# ════════════════════════════════════════════════════════════════════════════

COLOR_HSV = {
    "red":    ((0,   181,  46),  (10,  255, 234)),
    "green":  ((34,   99,  45),  (76,  255, 255)),
    "blue":   ((99,  129, 102),  (120, 255, 231)),
    "yellow": ((21,  172, 185),  (255, 255, 242)),
}

TARGET_COLOR = "red"

# ════════════════════════════════════════════════════════════════════════════
# PARÁMETROS DEL RECTÁNGULO VISIBLE
# Medidas reales del área que ve la cámara en watch_pose
# ════════════════════════════════════════════════════════════════════════════

# Resolución
IMG_W = 640
IMG_H = 480

# Área real visible en mm
AREA_X_MM  = 130.0    # ancho real
AREA_Y_MM  = 167.5    # alto real

# Bordes del rectángulo en píxeles
# Calibrar en laboratorio midiendo dónde caen los bordes en la imagen
X_LEFT   = 0           # píxel borde izquierdo
X_RIGHT  = IMG_W       # píxel borde derecho
Y_TOP    = 0           # píxel borde superior
Y_BOTTOM = IMG_H       # píxel borde inferior

# Watch pose
WATCH_ANGLES = [9.05, 3.42, -1.4, -73.21, 2.1, -30.23]

# Área mínima del contorno
AREA_MINIMA = 1000


# ════════════════════════════════════════════════════════════════════════════
# CONVERSIÓN  píxel → mm
# Fórmula proporcionada por el Rol 3 (Control)
# Origen (0,0) = centro del rectángulo
# x_mm va de -65 a +65
# y_mm va de -83.75 a +83.75
# ════════════════════════════════════════════════════════════════════════════

def pixel_a_mm(cx: float, cy: float) -> tuple:
    """
    Convierte centroide (cx, cy) en píxeles a (x_mm, y_mm) en mm.

    Fórmula:
        x_mm = ((cx - x_left) / (x_right - x_left)) * 130.0 - 65.0
        y_mm = 83.75 - ((cy - y_top) / (y_bottom - y_top)) * 167.5

    Parámetros
    ----------
    cx : centroide X en píxeles
    cy : centroide Y en píxeles

    Devuelve
    --------
    (x_mm, y_mm) con origen en el centro del rectángulo
    """
    x_mm = ((cx - X_LEFT) / (X_RIGHT - X_LEFT)) * AREA_X_MM - (AREA_X_MM / 2)
    y_mm = (AREA_Y_MM / 2) - ((cy - Y_TOP) / (Y_BOTTOM - Y_TOP)) * AREA_Y_MM
    return round(x_mm, 2), round(y_mm, 2)


# ════════════════════════════════════════════════════════════════════════════
# DETECCIÓN — Pipeline P6
# 1. Recibir frame
# 2. Convertir a HSV
# 3. Detectar cubo por color
# 4. Encontrar contorno más grande
# 5. Calcular centroide (cx, cy)
# 6. Convertir a mm
# 7. Devolver (x_mm, y_mm)
# ════════════════════════════════════════════════════════════════════════════

def detect_object(frame: np.ndarray,
                  color: str = TARGET_COLOR) -> tuple:
    """
    Detecta el cubo de color y devuelve su posición en mm.

    Parámetros
    ----------
    frame : imagen BGR de cv2.VideoCapture
    color : color a detectar ("red", "green", "blue", "yellow")

    Devuelve
    --------
    (x_mm, y_mm) | None
    """
    if color not in COLOR_HSV:
        log.error("Color '%s' no definido.", color)
        return None

    # Paso 1 — Normalizar frame
    frame = cv.resize(frame, (IMG_W, IMG_H))

    lower_bound = np.array(COLOR_HSV[color][0])
    upper_bound = np.array(COLOR_HSV[color][1])

    # Paso 2 — Convertir a HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Paso 3 — Detectar cubo por color
    mask = cv.inRange(hsv, lower_bound, upper_bound)

    # Limpiar ruido con morfología
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    mask   = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # Paso 4 — Encontrar contorno más grande
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL,
                                   cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    cnt  = max(contours, key=cv.contourArea)
    area = cv.contourArea(cnt)

    if area < AREA_MINIMA:
        return None

    # Paso 5 — Calcular centroide (cx, cy)
    x, y, w, h = cv.boundingRect(cnt)
    cx = float(x + w / 2)
    cy = float(y + h / 2)

    # Paso 6 — Convertir a mm
    x_mm, y_mm = pixel_a_mm(cx, cy)

    log.info("Cubo '%s': (%.1f,%.1f)px → (%.1f,%.1f)mm",
             color, cx, cy, x_mm, y_mm)

    # Paso 7 — Devolver posición
    return (x_mm, y_mm)


# ════════════════════════════════════════════════════════════════════════════
# VALIDACIÓN P6 — tabla de 3 posiciones
# Ejecutar: python3 vision.py
# 'g' = guardar medición | 'q' = salir
# ════════════════════════════════════════════════════════════════════════════

def validar_deteccion():
    """Valida la detección en 3 posiciones distintas del objeto."""
    cap = cv.VideoCapture(0)
    cap.set(6, cv.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    cap.set(cv.CAP_PROP_FRAME_WIDTH,  IMG_W)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, IMG_H)

    if not cap.isOpened():
        log.error("No se pudo abrir la cámara.")
        return

    log.info("Cámara abierta. 'g'=guardar  'q'=salir")
    mediciones = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resultado = detect_object(frame, TARGET_COLOR)
        frame_show = frame.copy()

        if resultado:
            x_mm, y_mm = resultado
            cv.putText(frame_show,
                       f"x={x_mm:.1f}mm  y={y_mm:.1f}mm",
                       (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 255, 0), 2)
        else:
            cv.putText(frame_show, "No detectado",
                       (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 0, 255), 2)

        cv.imshow("P6 - Validacion vision", frame_show)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g') and resultado:
            mediciones.append(resultado)
            log.info("Medicion %d: (%.1f, %.1f) mm",
                     len(mediciones), resultado[0], resultado[1])

    cap.release()
    cv.destroyAllWindows()

    # Tabla de validación para el informe
    print("\n══ TABLA DE VALIDACIÓN P6 ══")
    print(f"{'#':>3} | {'x_real(mm)':>10} | {'y_real(mm)':>10} | "
          f"{'x_det(mm)':>10} | {'y_det(mm)':>10} | {'error(mm)':>10}")
    print("-" * 65)
    for i, (x, y) in enumerate(mediciones):
        print(f"{i+1:>3} | {'medir':>10} | {'medir':>10} | "
              f"{x:>10.1f} | {y:>10.1f} | {'calcular':>10}")
    print("\nCompleta x_real e y_real midiendo con regla en el laboratorio.")


if __name__ == "__main__":
    validar_deteccion()
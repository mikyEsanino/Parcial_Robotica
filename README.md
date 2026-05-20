CONFIGURACIÓN | ERROR X (mm) | ERROR Y (mm) | ERROR Z (mm)
-----------------------------------------------------------------
Config 1     |      156.700 |       51.490 |      205.360
Config 2     |      225.523 |       49.586 |      181.027
Config 3     |      100.005 |      126.418 |      222.669
Config 4     |      303.565 |       51.490 |       95.095
Config 5     |      116.782 |       20.426 |      217.213

POSICIÓN OBJETIVO   | ERROR q1 (°) | ERROR q2 (°) | ERROR q3 (°)
-----------------------------------------------------------------
Pos 1 [140.0, 0.0, 220.0] |        0.790 |        0.038 |        0.298
Pos 2 [120.0, 60.0, 180.0] |        0.645 |        0.752 |        0.050
Pos 3 [150.0, -40.0, 200.0] |        0.871 |        1.034 |        1.125

HOME [0,0,0,0,0,0] coords: [53.2, -64.4, 409.1, -91.93, 0.61, -89.84]
BASE 90° coords: [65.9, 52.0, 409.1, -91.93, 0.67, -1.51]
HOMBRO 45° coords: [-166.0, -68.0, 361.1, -43.58, 0.99, -89.0]
import cv2
import numpy as np

cap = cv2.VideoCapture(0) # Abre la cámara de la Jetson Nano o PC

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Pasar de BGR (normal) a HSV (ideal para segmentar colores)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # TODO: Reemplazar estos valores fijos por los que obtengas con tus Trackbars
    lower_bound = np.array([H_min, S_min, V_min])
    upper_bound = np.array([H_max, S_max, V_max])
    
    # Crear la máscara binaria (blanco lo que coincide, negro lo que no)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    cv2.imshow("Camara Real", frame)
    cv2.imshow("Mascara de Color", mask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

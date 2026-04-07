"""
=============================================================================
 RECONOCIMIENTO FACIAL EN TIEMPO REAL
 Tecnologías: OpenCV + MediaPipe + NumPy
=============================================================================
 Requisitos de instalación:
   pip install opencv-python mediapipe numpy

 Ejecución:
   python face_recognition_realtime.py

 Controles:
   Q  → Salir
   S  → Guardar snapshot + JSON con medidas actuales
   D  → Alternar modo de depuración (muestra todos los landmarks)
=============================================================================
"""

# ──────────────────────────────────────────────────────────────────────────────
# 1. IMPORTACIONES
# ──────────────────────────────────────────────────────────────────────────────
import cv2
import mediapipe as mp
import numpy as np
import json
import socket
import time
import os
from datetime import datetime
from typing import Optional, Tuple, Dict, List

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ──────────────────────────────────────────────────────────────────────────────
# 2. CONFIGURACIÓN GLOBAL
# ──────────────────────────────────────────────────────────────────────────────

# Colores BGR
COLOR_BOX        = (0,   220, 255)   # Amarillo cian – rectángulo del rostro
COLOR_LANDMARK   = (255, 100,   0)   # Azul     – puntos faciales
COLOR_LINE       = (0,   255, 120)   # Verde    – líneas de medición
COLOR_TEXT_BG    = (15,   15,  15)   # Negro    – fondo de texto
COLOR_TEXT       = (240, 240, 240)   # Blanco   – texto principal
COLOR_ACCENT     = (0,   140, 255)   # Naranja  – valores numéricos
COLOR_WARN       = (0,    60, 255)   # Rojo     – avisos

# Índices de landmarks MediaPipe Face Mesh (478 puntos)
# Referencia: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
LANDMARKS_IDX = {
    # Contorno de la mandíbula / cara
    "jaw_left"      : 234,   # extremo izquierdo de la mandíbula
    "jaw_right"     : 454,   # extremo derecho de la mandíbula
    "chin"          : 152,   # mentón (centro inferior)
    "forehead"      : 10,    # frente (centro superior)

    # Ojos
    "eye_left_outer"  : 33,
    "eye_left_inner"  : 133,
    "eye_right_inner" : 362,
    "eye_right_outer" : 263,
    "eye_left_top"    : 159,
    "eye_left_bot"    : 145,
    "eye_right_top"   : 386,
    "eye_right_bot"   : 374,

    # Nariz
    "nose_tip"        : 1,
    "nose_base_left"  : 129,
    "nose_base_right" : 358,

    # Boca
    "mouth_left"      : 61,
    "mouth_right"     : 291,
    "mouth_top"       : 13,
    "mouth_bot"       : 14,

    # Cejas
    "brow_left"       : 70,
    "brow_right"      : 300,
}

# ──────────────────────────────────────────────────────────────────────────────
# 3. INICIALIZACIÓN DE MEDIAPIPE
# ──────────────────────────────────────────────────────────────────────────────
# Ya no se usan mp.solutions, sino Tasks API.
# Rutas de modelos para Tasks API (deben estar descargados en la ruta)
FACE_LANDMARKER_MODEL_PATH = r"C:\Users\Marquito\face_landmarker.task"



# ──────────────────────────────────────────────────────────────────────────────
# 4. FUNCIÓN: detect_face()
# ──────────────────────────────────────────────────────────────────────────────
def detect_face(
    frame_rgb: np.ndarray,
    face_detector: mp.solutions.face_detection.FaceDetection
) -> Optional[Tuple[int, int, int, int]]:
    """
    Detecta el rostro principal en el frame y devuelve su bounding box.

    Parámetros
    ----------
    frame_rgb   : imagen en formato RGB (H×W×3)
    face_detector: instancia de mp.solutions.face_detection.FaceDetection

    Retorna
    -------
    (x, y, w, h) en píxeles del rostro más prominente, o None si no hay.
    """
    results = face_detector.process(frame_rgb)

    if not results.detections:
        return None

    # Tomamos la primera detección (la de mayor confianza)
    detection   = results.detections[0]
    bboxC       = detection.location_data.relative_bounding_box
    ih, iw      = frame_rgb.shape[:2]

    x = max(0, int(bboxC.xmin * iw))
    y = max(0, int(bboxC.ymin * ih))
    w = min(int(bboxC.width  * iw), iw - x)
    h = min(int(bboxC.height * ih), ih - y)

    return (x, y, w, h)


# ──────────────────────────────────────────────────────────────────────────────
# 5. FUNCIÓN: get_landmarks()
# ──────────────────────────────────────────────────────────────────────────────
def get_landmarks(
    frame_rgb : np.ndarray,
    face_landmarker: vision.FaceLandmarker
) -> Optional[Dict[str, Tuple[int, int]]]:
    """
    Extrae los puntos faciales clave mediante MediaPipe Face Landmarker.

    Parámetros
    ----------
    frame_rgb : imagen en formato RGB
    face_landmarker : instancia de vision.FaceLandmarker

    Retorna
    -------
    Diccionario { nombre_landmark: (x_pixel, y_pixel) }
    o None si no se detecta ningún rostro.
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    # Calculate timestamp manually for video mode
    timestamp_ms = int(time.time() * 1000)
    
    # Run the detection
    # If using IMAGE mode, it would just be face_landmarker.detect(mp_image)
    # Using video mode for continuous stream
    try:
        results = face_landmarker.detect_for_video(mp_image, timestamp_ms)
    except Exception as e:
        print(f"[WARN] Error in landmarker: {e}")
        return None

    if not results.face_landmarks:
        return None

    ih, iw = frame_rgb.shape[:2]
    # First detected face
    face_lm = results.face_landmarks[0]

    # Convertimos coordenadas normalizadas → píxeles
    landmarks_px: Dict[str, Tuple[int, int]] = {}
    for name, idx in LANDMARKS_IDX.items():
        if idx < len(face_lm):
            lm = face_lm[idx]
            landmarks_px[name] = (int(lm.x * iw), int(lm.y * ih))

    # También devolvemos la lista raw completa para modo depuración
    landmarks_px["_raw"] = face_lm   # type: ignore

    return landmarks_px


# ──────────────────────────────────────────────────────────────────────────────
# 6. FUNCIÓN: calculate_distances()
# ──────────────────────────────────────────────────────────────────────────────
def calculate_distances(
    lm: Dict[str, Tuple[int, int]],
    frame_shape: Tuple[int, int]
) -> Dict[str, float]:
    """
    Calcula distancias y proporciones entre puntos externos del rostro.

    Medidas calculadas
    ------------------
    face_width       : distancia horizontal jaw_left → jaw_right
    face_height      : distancia vertical forehead → chin
    face_ratio       : face_width / face_height  (>1 = cara ancha, <1 = cara larga)
    eye_distance     : eye_left_outer → eye_right_outer
    left_eye_open    : apertura vertical ojo izquierdo
    right_eye_open   : apertura vertical ojo derecho
    ear_ratio        : (left_eye_open + right_eye_open) / 2  (detecta somnolencia)
    nose_width       : nose_base_left → nose_base_right
    mouth_width      : mouth_left → mouth_right
    mouth_open       : apertura vertical de la boca
    brow_height      : promedio de altura de cejas respecto a ojos

    Retorna
    -------
    Diccionario con los valores calculados en píxeles / adimensional.
    """
    def dist(p1: str, p2: str) -> float:
        """Distancia euclídea entre dos landmarks."""
        a = np.array(lm[p1])
        b = np.array(lm[p2])
        return float(np.linalg.norm(a - b))

    ih, iw = frame_shape

    face_w  = dist("jaw_left",        "jaw_right")
    face_h  = dist("forehead",        "chin")
    eye_d   = dist("eye_left_outer",  "eye_right_outer")
    l_eye   = dist("eye_left_top",    "eye_left_bot")
    r_eye   = dist("eye_right_top",   "eye_right_bot")
    nose_w  = dist("nose_base_left",  "nose_base_right")
    mouth_w = dist("mouth_left",      "mouth_right")
    mouth_o = dist("mouth_top",       "mouth_bot")

    # Eye Aspect Ratio (EAR) – usado en detección de fatiga
    ear = (l_eye + r_eye) / (2.0 * eye_d + 1e-6)

    # Altura de cejas relativa a los ojos (positivo = cejas altas)
    brow_l = lm["eye_left_top"][1]  - lm["brow_left"][1]
    brow_r = lm["eye_right_top"][1] - lm["brow_right"][1]
    brow_h = (brow_l + brow_r) / 2.0

    jaw_ratio    = face_w / (face_h + 1e-6)
    mouth_ratio  = mouth_w / (face_w + 1e-6)
    nose_ratio   = nose_w / (face_w + 1e-6)
    eye_ratio    = eye_d / (face_w + 1e-6)

    return {
        "face_width_px"    : round(face_w,  1),
        "face_height_px"   : round(face_h,  1),
        "face_ratio"       : round(face_w / (face_h + 1e-6), 3),
        "jaw_ratio"        : round(jaw_ratio, 3),
        "mouth_ratio"      : round(mouth_ratio, 3),
        "nose_ratio"       : round(nose_ratio, 3),
        "eye_ratio"        : round(eye_ratio, 3),
        "eye_distance_px"  : round(eye_d,   1),
        "left_eye_open_px" : round(l_eye,   1),
        "right_eye_open_px": round(r_eye,   1),
        "eye_aspect_ratio" : round(ear,      4),
        "nose_width_px"    : round(nose_w,  1),
        "mouth_width_px"   : round(mouth_w, 1),
        "mouth_open_px"    : round(mouth_o, 1),
        "brow_height_px"   : round(brow_h,  1),
        "frame_w_px"       : iw,
        "frame_h_px"       : ih,
        "timestamp"        : round(time.time(), 3),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 7. FUNCIÓN: draw_results()
# ──────────────────────────────────────────────────────────────────────────────
def draw_results(
    frame   : np.ndarray,
    bbox    : Optional[Tuple[int, int, int, int]],
    lm      : Optional[Dict],
    dists   : Optional[Dict[str, float]],
    debug   : bool = False
) -> np.ndarray:
    """
    Dibuja sobre el frame todos los elementos visuales:
      – Bounding box del rostro
      – Puntos landmarks y sus etiquetas
      – Líneas de medición
      – Panel de métricas en pantalla

    Retorna el frame anotado.
    """
    overlay = frame.copy()

    # ── 7.1 Bounding box ──────────────────────────────────────────────────────
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), COLOR_BOX, 2)
        cv2.putText(overlay, "FACE DETECTED", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BOX, 1, cv2.LINE_AA)

    # ── 7.2 Landmarks y líneas ────────────────────────────────────────────────
    if lm:
        raw = lm.get("_raw")

        # Modo depuración: dibuja los 478 puntos
        if debug and raw:
            ih, iw = frame.shape[:2]
            for landmark in raw:
                px = int(landmark.x * iw)
                py = int(landmark.y * ih)
                cv2.circle(overlay, (px, py), 1, (180, 180, 180), -1)

        # Puntos clave con nombres
        key_points = {k: v for k, v in lm.items() if not k.startswith("_")}
        for name, (px, py) in key_points.items():
            cv2.circle(overlay, (px, py), 4, COLOR_LANDMARK, -1)
            cv2.circle(overlay, (px, py), 5, (255, 255, 255), 1)  # borde blanco

        # Líneas de medición principal
        measurement_lines = [
            ("jaw_left",       "jaw_right",       "W"),
            ("forehead",       "chin",            "H"),
            ("eye_left_outer", "eye_right_outer", "E"),
            ("nose_base_left", "nose_base_right", "N"),
            ("mouth_left",     "mouth_right",     "M"),
        ]
        for p1_name, p2_name, label in measurement_lines:
            if p1_name in lm and p2_name in lm:
                p1 = lm[p1_name]
                p2 = lm[p2_name]
                cv2.line(overlay, p1, p2, COLOR_LINE, 1, cv2.LINE_AA)
                # Pequeño marcador en los extremos
                cv2.circle(overlay, p1, 3, COLOR_LINE, -1)
                cv2.circle(overlay, p2, 3, COLOR_LINE, -1)
                # Etiqueta centrada en la línea
                mx = (p1[0] + p2[0]) // 2
                my = (p1[1] + p2[1]) // 2
                cv2.putText(overlay, label, (mx + 4, my - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_LINE, 1, cv2.LINE_AA)

        # Línea vertical frente→mentón
        if "forehead" in lm and "chin" in lm:
            cv2.line(overlay, lm["forehead"], lm["chin"],
                     (120, 200, 255), 1, cv2.LINE_AA)

    # ── 7.3 Panel de métricas ─────────────────────────────────────────────────
    if dists:
        panel_x, panel_y = 10, 10
        panel_w, panel_h = 260, 290
        # Fondo semitransparente
        roi = overlay[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w]
        bg  = np.zeros_like(roi)
        bg[:] = COLOR_TEXT_BG
        cv2.addWeighted(roi, 0.3, bg, 0.7, 0, roi)
        overlay[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w] = roi

        # Borde del panel
        cv2.rectangle(overlay,
                      (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h),
                      COLOR_BOX, 1)

        # Título
        cv2.putText(overlay, " FACE METRICS",
                    (panel_x + 6, panel_y + 18),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, COLOR_BOX, 1, cv2.LINE_AA)
        cv2.line(overlay,
                 (panel_x + 4, panel_y + 24),
                 (panel_x + panel_w - 4, panel_y + 24),
                 COLOR_BOX, 1)

        # Métricas
        metrics = [
            ("Ancho cara",    f"{dists['face_width_px']} px"),
            ("Alto cara",     f"{dists['face_height_px']} px"),
            ("Ratio A/H",     f"{dists['face_ratio']}"),
            ("Dist. ojos",    f"{dists['eye_distance_px']} px"),
            ("Ojo izq.",      f"{dists['left_eye_open_px']} px"),
            ("Ojo der.",      f"{dists['right_eye_open_px']} px"),
            ("EAR (fatiga)",  f"{dists['eye_aspect_ratio']}"),
            ("Ancho nariz",   f"{dists['nose_width_px']} px"),
            ("Ancho boca",    f"{dists['mouth_width_px']} px"),
            ("Apertura boca", f"{dists['mouth_open_px']} px"),
        ]

        for i, (label, value) in enumerate(metrics):
            ty = panel_y + 42 + i * 24
            cv2.putText(overlay, label + ":",
                        (panel_x + 8, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT, 1, cv2.LINE_AA)
            cv2.putText(overlay, value,
                        (panel_x + 158, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_ACCENT, 1, cv2.LINE_AA)

        # Indicador EAR (alerta somnolencia)
        ear = dists["eye_aspect_ratio"]
        if ear < 0.15:
            cv2.putText(overlay, "⚠ EYES CLOSED",
                        (panel_x + 6, panel_y + panel_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WARN, 2, cv2.LINE_AA)

    # ── 7.4 Controles (esquina inferior izquierda) ────────────────────────────
    h_frame = frame.shape[0]
    controls = ["[Q] Salir", "[S] Snapshot+JSON", "[D] Debug landmarks"]
    for i, txt in enumerate(controls):
        cv2.putText(overlay, txt,
                    (10, h_frame - 12 - i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

    # Aplica overlay con transparencia
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    return frame


# ──────────────────────────────────────────────────────────────────────────────
# 8. FUNCIÓN: export_data()  ← extensibilidad futura (Spark AR / sockets)
# ──────────────────────────────────────────────────────────────────────────────
def export_to_json(dists: Dict, filepath: str = "face_data.json") -> None:
    """
    Exporta las métricas faciales a un archivo JSON.
    Puede ser leído en tiempo real por Spark AR u otro sistema.
    Ejemplo de integración: Spark AR → Patch Editor → Fetch API → JSON local.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(dists, f, indent=2)


def send_via_socket(
    dists      : Dict,
    host       : str = "127.0.0.1",
    port       : int = 9000,
    protocol   : str = "udp"   # "udp" o "tcp"
) -> None:
    """
    Envía las métricas por socket UDP/TCP (ej: hacia Spark AR Studio).

    Uso futuro:
      En Spark AR, habilita 'Networking > Receive from localhost'
      y escucha en el puerto definido.

    Por ahora la función falla silenciosamente si no hay receptor.
    """
    try:
        payload = json.dumps(dists).encode("utf-8")
        if protocol == "udp":
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(payload, (host, port))
            sock.close()
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.05)
            sock.connect((host, port))
            sock.sendall(payload)
            sock.close()
    except Exception:
        pass   # Sin receptor activo: ignorar silenciosamente


# ──────────────────────────────────────────────────────────────────────────────
# 9. BUCLE PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────
class FaceAnalyzer:
    def __init__(self, camera_index: int = 0, analyze_time: float = 8.0):
        """
        Inicializa la cámara, el modelo MediaPipe Tasks y las variables de estado.
        """
        self.analyze_time = analyze_time
        
        # 1. Apertura de la cámara
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print("[ERROR] No se pudo abrir la cámara. Verifica permisos o índice.")
            self.valid = False
            return
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # 2. Inicialización de MediaPipe Face Landmarker
        base_options = python.BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
        )
        
        try:
            self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
            self.valid = True
        except Exception as e:
            print(f"[ERROR] No se pudo cargar el modelo Face Landmarker: {e}")
            print("Asegúrate de haber descargado 'face_landmarker.task' en la ruta configurada.")
            self.cap.release()
            self.valid = False
            return

        # 3. Variables de estado
        self.fps_buffer: List[float] = []
        self.last_dists: Optional[Dict[str, float]] = None
        self.face_detection_start_time: Optional[float] = None
        self.debug_mode = False

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]], Optional[Dict[str, float]]]:
        """
        Procesa un frame individual: convierte formatos, extrae landmarks,
        calcula medidas y renderiza visualizaciones.
        """
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        lm = get_landmarks(frame_rgb, self.face_landmarker)
        
        bbox = None
        dists = None
        if lm:
            # Extraemos caja delimitadora a partir de los puntos extremos
            raw_lm = lm.get("_raw")
            if raw_lm:
                ih, iw = frame_rgb.shape[:2]
                x_coords = [int(p.x * iw) for p in raw_lm]
                y_coords = [int(p.y * ih) for p in raw_lm]
                
                x_min, x_max = max(0, min(x_coords)), min(iw, max(x_coords))
                y_min, y_max = max(0, min(y_coords)), min(ih, max(y_coords))
                
                margin = int((x_max - x_min) * 0.1)
                bbox = (
                    max(0, x_min - margin), 
                    max(0, y_min - margin), 
                    min(iw, x_max + margin) - max(0, x_min - margin), 
                    min(ih, y_max + margin) - max(0, y_min - margin)
                )

            dists = calculate_distances(lm, frame.shape[:2])
            self.last_dists = dists

            # Imprimir métricas en consola (~0.5 s)
            if int(time.time() * 2) % 2 == 0:   
                print(
                    f"\r[METRICS] W={dists['face_width_px']:>6.1f}px  "
                    f"H={dists['face_height_px']:>6.1f}px  "
                    f"Ratio={dists['face_ratio']:.3f}  "
                    f"EAR={dists['eye_aspect_ratio']:.4f}  "
                    f"Boca={dists['mouth_open_px']:>4.1f}px",
                    end="", flush=True
                )

        frame_rgb.flags.writeable = True
        frame = draw_results(frame, bbox, lm, dists, debug=self.debug_mode)

        if dists:
            ratio = dists["face_ratio"]
            jaw   = dists["jaw_ratio"]
            mouth = dists["mouth_ratio"]
            cv2.putText(frame, f"R:{ratio:.2f} J:{jaw:.2f} M:{mouth:.2f}",
                        (10, frame.shape[0]-100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        return frame, bbox, dists

    def check_timer_and_classify(self, frame: np.ndarray, face_detected: bool) -> bool:
        """
        Gestiona el temporizador y devuelve True si el análisis está completo.
        """
        if face_detected:
            if self.face_detection_start_time is None:
                self.face_detection_start_time = time.time()
                print("\n[INFO] Rostro detectado. Mantente quieto 8 segundos...")
                
            elapsed = time.time() - self.face_detection_start_time
            time_left = max(0.0, self.analyze_time - elapsed)
            
            cv2.putText(frame, f"Analizando: {time_left:.1f}s",
                        (frame.shape[1] // 2 - 80, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            if elapsed >= self.analyze_time:
                self._print_final_classification()
                return True
        else:
            if self.face_detection_start_time is not None:
                print("\n[WARN] Rostro perdido. Se reinició el temporizador.")
            self.face_detection_start_time = None
            
        return False

    def classify_face_pro(self, d: Dict[str, float]) -> str:
        ratio = d["face_ratio"]
        jaw   = d["jaw_ratio"]
        mouth = d["mouth_ratio"]
        nose  = d["nose_ratio"]

        # 🔴 RECTÁNGULO (cara larga)
        if ratio < 0.72:
            return "Rectángulo"

        # 🟣 OVALADO (balanceado)
        if 0.72 <= ratio <= 0.82 and 0.36 <= mouth <= 0.44:
            return "Ovalado"

        # 🔵 REDONDO (ancho + proporciones suaves)
        if 0.82 < ratio <= 0.92 and jaw > 0.80:
            return "Redondo"

        # 🟫 CUADRADO (mandíbula marcada)
        if ratio > 0.88 and jaw >= 0.90:
            return "Cuadrado"

        # 🔻 TRIÁNGULO (mandíbula dominante)
        if mouth > 0.44 and jaw > 0.85:
            return "Triángulo"

        # ❤️ CORAZÓN (frente ancha, mandíbula estrecha)
        if jaw < 0.78 and mouth < 0.38:
            return "Corazón"

        # 🔷 DIAMANTE (pómulos dominantes)
        if jaw < 0.75 and 0.38 <= mouth <= 0.42:
            return "Diamante"

        return "Indefinido"

    def _print_final_classification(self) -> None:
        """Imprime el resultado final después del tiempo de análisis."""
        print("\n" + "=" * 50)
        print("ANÁLISIS FACIAL COMPLETADO")
        print("=" * 50)
        if self.last_dists:
            ratio = self.last_dists["face_ratio"]
            print(f"Métricas finales: Ancho {self.last_dists['face_width_px']}px | Alto {self.last_dists['face_height_px']}px")
            print(f"Proporción (Ancho/Alto) = {ratio:.3f}")
            
            tipo = self.classify_face_pro(self.last_dists)
            print(f">> TIPO DE ROSTRO: {tipo}")
        else:
            print(">> ERROR: No se generaron proporciones faciales.")
        print("=" * 50 + "\n")

    def run(self) -> None:
        """
        Inicia el bucle principal de la aplicación.
        """
        if not self.valid:
            return
            
        print("[INFO] Iniciando captura de video... (Espera 8 segundos frente a la cámara)")
        
        while True:
            t_start = time.perf_counter()
            ret, frame = self.cap.read()
            if not ret:
                print("[WARN] Frame no recibido. Reintentando...")
                continue

            frame, bbox, dists = self.process_frame(frame)
            
            # FPS counter
            t_end = time.perf_counter()
            self.fps_buffer.append(1.0 / max(t_end - t_start, 1e-6))
            if len(self.fps_buffer) > 30:
                self.fps_buffer.pop(0)
            fps = np.mean(self.fps_buffer)
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 110, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_BOX, 1, cv2.LINE_AA)

            start_status = "DETECTADO" if bbox else "BUSCANDO..."
            status_color = COLOR_LINE if bbox else COLOR_WARN
            cv2.putText(frame, start_status, (frame.shape[1] - 140, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)

            # Timer y finalización
            if self.check_timer_and_classify(frame, bbox is not None):
                break

            cv2.imshow("Reconocimiento Facial | OpenCV + MediaPipe", frame)
            cv2.waitKey(1)

    def close(self) -> None:
        """Libera los recursos antes de salir."""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        if hasattr(self, 'face_landmarker') and self.face_landmarker:
            self.face_landmarker.close()
        cv2.destroyAllWindows()
        
        if hasattr(self, 'last_dists') and self.last_dists is not None:
            export_to_json(self.last_dists, "face_data_last.json")
            print("[INFO] Métricas finales guardadas en 'face_data_last.json'")
        print("[INFO] Programa finalizado correctamente.")


def main() -> None:
    """
    Punto de entrada principal refactorizado. Instancia e inicia el analizador.
    """
    print("=" * 60)
    print("  RECONOCIMIENTO FACIAL EN TIEMPO REAL")
    print("  OpenCV + MediaPipe")
    print("=" * 60)
    print("  Controles:")
    print("    El análisis dura 8 segundos automáticamente tras detectar un rostro.")
    print("=" * 60)

    # Iniciar flujo de análisis
    analyzer = FaceAnalyzer(analyze_time=8.0)
    try:
        analyzer.run()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupción por teclado.")
    finally:
        analyzer.close()


# ──────────────────────────────────────────────────────────────────────────────
# 10. PUNTO DE ENTRADA
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

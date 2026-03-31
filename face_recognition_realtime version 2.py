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

# ──────────────────────────────────────────────────────────────────────────────
# 3. LANDMARKS MEDIAPIPE FACE MESH (478 puntos)
# ──────────────────────────────────────────────────────────────────────────────
# Referencia: https://github.com/google/mediapipe/blob/master/mediapipe/modules/
#             face_geometry/data/canonical_face_model_uv_visualization.png
#
# AMPLIADOS: Ahora incluimos puntos para medir CINCO zonas horizontales
# del rostro (frente, sienes, pómulos, mandíbula baja, mentón) y tres
# tercios verticales, lo que permite una clasificación geométrica robusta.
# ──────────────────────────────────────────────────────────────────────────────

LANDMARKS_IDX = {
    # ── Contorno vertical ────────────────────────────────────────────────────
    "forehead_top"       : 10,    # punto más alto de la frente (centro)
    "chin"               : 152,   # mentón (centro inferior)

    # ── Anchuras horizontales por zona ───────────────────────────────────────
    #   Frente: usamos los extremos EXTERIORES de las cejas (46/276)
    #   porque los puntos de sien (54/284) quedan demasiado adentro
    #   y hacen que la frente mida artificialmente estrecha.
    "forehead_left"      : 46,    # extremo exterior ceja izquierda
    "forehead_right"     : 276,   # extremo exterior ceja derecha

    #   Pómulos / mejillas (zona más ancha de la cara normalmente)
    "cheekbone_left"     : 234,   # pómulo izquierdo
    "cheekbone_right"    : 454,   # pómulo derecho

    #   Mandíbula (parte media-baja, donde la mandíbula empieza a estrechar)
    "jaw_mid_left"       : 172,   # mandíbula media izquierda
    "jaw_mid_right"      : 397,   # mandíbula media derecha

    #   Mandíbula baja (cerca del mentón, la parte más estrecha)
    "jaw_low_left"       : 150,   # mandíbula baja izquierda
    "jaw_low_right"      : 379,   # mandíbula baja derecha

    #   Mandíbula ángulo (gonion – ángulo mandibular)
    "jaw_angle_left"     : 132,   # ángulo mandibular izquierdo
    "jaw_angle_right"    : 361,   # ángulo mandibular derecho

    # ── Ojos ─────────────────────────────────────────────────────────────────
    "eye_left_outer"     : 33,
    "eye_left_inner"     : 133,
    "eye_right_inner"    : 362,
    "eye_right_outer"    : 263,
    "eye_left_top"       : 159,
    "eye_left_bot"       : 145,
    "eye_right_top"      : 386,
    "eye_right_bot"      : 374,

    # ── Nariz ────────────────────────────────────────────────────────────────
    "nose_tip"           : 1,
    "nose_bridge_top"    : 6,     # puente nasal (entre los ojos)
    "nose_base_left"     : 129,
    "nose_base_right"    : 358,

    # ── Boca ─────────────────────────────────────────────────────────────────
    "mouth_left"         : 61,
    "mouth_right"        : 291,
    "mouth_top"          : 13,
    "mouth_bot"          : 14,

    # ── Cejas ────────────────────────────────────────────────────────────────
    "brow_left_inner"    : 107,   # extremo interior ceja izquierda
    "brow_left_peak"     : 70,    # punto más alto ceja izquierda
    "brow_left_outer"    : 46,    # extremo exterior ceja izquierda
    "brow_right_inner"   : 336,   # extremo interior ceja derecha
    "brow_right_peak"    : 300,   # punto más alto ceja derecha
    "brow_right_outer"   : 276,   # extremo exterior ceja derecha

    # ── Puntos auxiliares para tercios ───────────────────────────────────────
    "glabella"           : 9,     # entrecejo (base del tercio superior)
    "nose_base_center"   : 94,    # subnasale (debajo de la nariz, NO la punta)
    #   NOTA: el landmark 2 es la punta de la nariz, que queda demasiado
    #   arriba y agranda artificialmente el tercio inferior.
}

# ──────────────────────────────────────────────────────────────────────────────
# 3b. INICIALIZACIÓN DE MEDIAPIPE
# ──────────────────────────────────────────────────────────────────────────────
FACE_LANDMARKER_MODEL_PATH = r"C:\Users\Marquito\face_landmarker.task"


# ──────────────────────────────────────────────────────────────────────────────
# 4. FUNCIÓN: get_landmarks()
# ──────────────────────────────────────────────────────────────────────────────
def get_landmarks(
    frame_rgb: np.ndarray,
    face_landmarker: vision.FaceLandmarker
) -> Optional[Dict[str, Tuple[int, int]]]:
    """
    Extrae los puntos faciales clave mediante MediaPipe Face Landmarker.
    Retorna diccionario { nombre_landmark: (x_pixel, y_pixel) } o None.
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = int(time.time() * 1000)

    try:
        results = face_landmarker.detect_for_video(mp_image, timestamp_ms)
    except Exception as e:
        print(f"[WARN] Error in landmarker: {e}")
        return None

    if not results.face_landmarks:
        return None

    ih, iw = frame_rgb.shape[:2]
    face_lm = results.face_landmarks[0]

    landmarks_px: Dict[str, Tuple[int, int]] = {}
    for name, idx in LANDMARKS_IDX.items():
        if idx < len(face_lm):
            lm = face_lm[idx]
            landmarks_px[name] = (int(lm.x * iw), int(lm.y * ih))

    landmarks_px["_raw"] = face_lm  # type: ignore
    return landmarks_px


# ──────────────────────────────────────────────────────────────────────────────
# 5. FUNCIÓN: calculate_distances()  – VERSIÓN AMPLIADA
# ──────────────────────────────────────────────────────────────────────────────
def calculate_distances(
    lm: Dict[str, Tuple[int, int]],
    frame_shape: Tuple[int, int]
) -> Dict[str, float]:
    """
    Calcula todas las métricas geométricas del rostro.

    Zonas horizontales medidas (de arriba a abajo):
        1. forehead_width  : distancia entre sienes (frente)
        2. cheekbone_width : distancia entre pómulos
        3. jaw_mid_width   : mandíbula media
        4. jaw_low_width   : mandíbula baja (cerca del mentón)

    Zonas verticales (tercios del rostro):
        1. upper_third  : forehead_top → glabella (frente)
        2. middle_third : glabella → nose_base_center (nariz)
        3. lower_third  : nose_base_center → chin (boca + mentón)

    Métricas derivadas adicionales:
        - brow_to_mouth   : distancia vertical cejas → boca
        - brow_to_chin    : distancia vertical cejas → mentón
        - chin_taper      : jaw_low_width / jaw_mid_width (qué tanto se angosta)
        - jaw_angle_width : distancia entre ángulos mandibulares (gonion)

    Todo se normaliza respecto a face_height para ser invariante a escala.
    """
    def dist(p1: str, p2: str) -> float:
        a = np.array(lm[p1], dtype=np.float64)
        b = np.array(lm[p2], dtype=np.float64)
        return float(np.linalg.norm(a - b))

    def vertical_dist(p1: str, p2: str) -> float:
        """Distancia solo en el eje Y (vertical)."""
        return abs(float(lm[p1][1]) - float(lm[p2][1]))

    ih, iw = frame_shape

    # ── Dimensiones absolutas (px) ───────────────────────────────────────────
    face_height       = dist("forehead_top", "chin")
    forehead_width    = dist("forehead_left", "forehead_right")
    cheekbone_width   = dist("cheekbone_left", "cheekbone_right")
    jaw_mid_width     = dist("jaw_mid_left", "jaw_mid_right")
    jaw_low_width     = dist("jaw_low_left", "jaw_low_right")
    jaw_angle_width   = dist("jaw_angle_left", "jaw_angle_right")

    eye_dist          = dist("eye_left_outer", "eye_right_outer")
    l_eye_open        = dist("eye_left_top", "eye_left_bot")
    r_eye_open        = dist("eye_right_top", "eye_right_bot")
    nose_width        = dist("nose_base_left", "nose_base_right")
    mouth_width       = dist("mouth_left", "mouth_right")
    mouth_open        = dist("mouth_top", "mouth_bot")

    # ── Tercios verticales del rostro (px) ───────────────────────────────────
    upper_third       = vertical_dist("forehead_top", "glabella")
    middle_third      = vertical_dist("glabella", "nose_base_center")
    lower_third       = vertical_dist("nose_base_center", "chin")

    # ── Distancias verticales cejas ↔ boca ───────────────────────────────────
    brow_avg_y  = (lm["brow_left_peak"][1] + lm["brow_right_peak"][1]) / 2.0
    mouth_avg_y = (lm["mouth_top"][1] + lm["mouth_bot"][1]) / 2.0
    chin_y      = float(lm["chin"][1])

    brow_to_mouth = abs(mouth_avg_y - brow_avg_y)
    brow_to_chin  = abs(chin_y - brow_avg_y)

    # ── Altura de cejas respecto a ojos ──────────────────────────────────────
    brow_l = lm["eye_left_top"][1] - lm["brow_left_peak"][1]
    brow_r = lm["eye_right_top"][1] - lm["brow_right_peak"][1]
    brow_height = (brow_l + brow_r) / 2.0

    # ── EAR (Eye Aspect Ratio) ───────────────────────────────────────────────
    ear = (l_eye_open + r_eye_open) / (2.0 * eye_dist + 1e-6)

    # ── Factor de seguridad contra divisiones por cero ───────────────────────
    fh = face_height + 1e-6
    cw = cheekbone_width + 1e-6

    # ══════════════════════════════════════════════════════════════════════════
    #  RATIOS NORMALIZADOS (invariantes a escala / distancia de cámara)
    # ══════════════════════════════════════════════════════════════════════════

    # --- Ratio principal ancho/alto (usando la zona MÁS ancha como referencia)
    max_width   = max(forehead_width, cheekbone_width, jaw_angle_width)
    face_ratio  = max_width / fh                    # >1 = cara ancha, <1 = cara larga

    # --- Anchuras normalizadas respecto a face_height
    forehead_n  = forehead_width / fh               # ancho frente normalizado
    cheekbone_n = cheekbone_width / fh              # ancho pómulos normalizado
    jaw_mid_n   = jaw_mid_width / fh                # ancho mandíbula media norm.
    jaw_angle_n = jaw_angle_width / fh              # ancho ángulo mandibular norm.

    # --- Relaciones entre zonas horizontales (quién es más ancho que quién)
    forehead_to_cheek   = forehead_width / cw       # >1 = frente más ancha que pómulos
    jaw_to_cheek        = jaw_angle_width / cw      # >1 = mandíbula más ancha que pómulos
    forehead_to_jaw     = forehead_width / (jaw_angle_width + 1e-6)

    # --- Estrechamiento del mentón
    chin_taper  = jaw_low_width / (jaw_mid_width + 1e-6)  # <1 = mentón puntiagudo

    # --- Tercios normalizados respecto a face_height
    upper_n     = upper_third / fh
    middle_n    = middle_third / fh
    lower_n     = lower_third / fh

    # --- Proporciones verticales cejas-boca
    brow_mouth_n = brow_to_mouth / fh
    brow_chin_n  = brow_to_chin / fh

    # --- Proporciones de boca y nariz respecto a ancho de cara
    mouth_ratio = mouth_width / cw
    nose_ratio  = nose_width / cw

    return {
        # Valores absolutos en px (para visualización)
        "face_height_px"      : round(face_height, 1),
        "face_width_px"       : round(max_width, 1),
        "forehead_width_px"   : round(forehead_width, 1),
        "cheekbone_width_px"  : round(cheekbone_width, 1),
        "jaw_mid_width_px"    : round(jaw_mid_width, 1),
        "jaw_angle_width_px"  : round(jaw_angle_width, 1),
        "jaw_low_width_px"    : round(jaw_low_width, 1),
        "eye_distance_px"     : round(eye_dist, 1),
        "left_eye_open_px"    : round(l_eye_open, 1),
        "right_eye_open_px"   : round(r_eye_open, 1),
        "nose_width_px"       : round(nose_width, 1),
        "mouth_width_px"      : round(mouth_width, 1),
        "mouth_open_px"       : round(mouth_open, 1),
        "brow_height_px"      : round(brow_height, 1),
        "brow_to_mouth_px"    : round(brow_to_mouth, 1),
        "brow_to_chin_px"     : round(brow_to_chin, 1),
        "upper_third_px"      : round(upper_third, 1),
        "middle_third_px"     : round(middle_third, 1),
        "lower_third_px"      : round(lower_third, 1),

        # Ratios normalizados (para clasificación – INVARIANTES A ESCALA)
        "face_ratio"          : round(face_ratio, 4),
        "forehead_n"          : round(forehead_n, 4),
        "cheekbone_n"         : round(cheekbone_n, 4),
        "jaw_mid_n"           : round(jaw_mid_n, 4),
        "jaw_angle_n"         : round(jaw_angle_n, 4),
        "forehead_to_cheek"   : round(forehead_to_cheek, 4),
        "jaw_to_cheek"        : round(jaw_to_cheek, 4),
        "forehead_to_jaw"     : round(forehead_to_jaw, 4),
        "chin_taper"          : round(chin_taper, 4),
        "upper_n"             : round(upper_n, 4),
        "middle_n"            : round(middle_n, 4),
        "lower_n"             : round(lower_n, 4),
        "brow_mouth_n"        : round(brow_mouth_n, 4),
        "brow_chin_n"         : round(brow_chin_n, 4),
        "mouth_ratio"         : round(mouth_ratio, 4),
        "nose_ratio"          : round(nose_ratio, 4),

        # EAR y meta
        "eye_aspect_ratio"    : round(ear, 4),
        "frame_w_px"          : iw,
        "frame_h_px"          : ih,
        "timestamp"           : round(time.time(), 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. CLASIFICADOR DE TIPO DE ROSTRO – SISTEMA DE SCORING VECTORIAL
# ══════════════════════════════════════════════════════════════════════════════
#
#  En lugar de una cadena rígida de if/else, este clasificador:
#
#   1) Extrae un VECTOR DE CARACTERÍSTICAS de 8 dimensiones del rostro.
#   2) Define un PERFIL PROTOTIPO para cada tipo de rostro.
#   3) Calcula un SCORE PONDERADO de similitud entre el vector medido
#      y cada prototipo.
#   4) El tipo con mayor score gana. Si hay empate o ambigüedad,
#      se aplican reglas de desempate basadas en características
#      discriminantes.
#
#  Esto tiene varias ventajas:
#    - No hay "huecos" donde un rostro caiga en "Indefinido".
#    - Es invariante a escala (todo normalizado por face_height).
#    - Funciona con rostros chicos porque los ratios se mantienen.
#    - Los pesos permiten ajustar qué característica importa más.
#
# ══════════════════════════════════════════════════════════════════════════════

class FaceShapeClassifier:
    """
    Clasificador geométrico de tipo de rostro basado en scoring vectorial.

    Tipos soportados:
        Ovalado, Redondo, Cuadrado, Rectangular, Triangular, Corazón, Diamante

    Cada tipo tiene un perfil definido por rangos [min, ideal, max] en
    8 dimensiones. El score se calcula como la suma ponderada de qué tan
    cerca está cada métrica del valor ideal, penalizando si sale del rango.
    """

    # ── Dimensiones del vector de características ────────────────────────────
    # Cada feature tiene un peso que indica su importancia discriminante.
    FEATURE_WEIGHTS = {
        "face_ratio"        : 3.0,   # ancho/alto global – MUY discriminante
        "forehead_to_cheek" : 2.0,   # frente vs pómulos
        "jaw_to_cheek"      : 2.5,   # mandíbula vs pómulos – clave para triángulo/corazón
        "forehead_to_jaw"   : 2.5,   # frente vs mandíbula – discrimina corazón vs triángulo
        "chin_taper"        : 2.0,   # qué tan puntiagudo es el mentón
        "lower_n"           : 1.5,   # proporción del tercio inferior
        "brow_mouth_n"      : 1.0,   # distancia vertical cejas→boca normalizada
        "cheekbone_n"       : 1.5,   # prominencia de pómulos respecto a altura
    }

    # ── Perfiles prototipo ───────────────────────────────────────────────────
    #  Formato: { feature: (min_aceptable, valor_ideal, max_aceptable) }
    #
    #  - Si la métrica cae dentro de [min, max], recibe score positivo.
    #  - Si cae exactamente en ideal, recibe score máximo.
    #  - Si cae fuera de [min, max], recibe penalización proporcional.
    #
    #  Los valores fueron calibrados a partir de referencias antropométricas
    #  de clasificación facial y ajustados para landmarks de MediaPipe.
    # ─────────────────────────────────────────────────────────────────────────

    PROFILES = {
        "Ovalado": {
            # Cara ligeramente más larga que ancha, bien balanceada.
            # Pómulos son la parte más ancha, frente ≈ mandíbula.
            # Mentón se angosta gradualmente. Tercios equilibrados.
            # CALIBRADO: con landmarks 46/276 (cejas) y 94 (subnasale).
            "face_ratio"        : (0.70, 0.80, 0.90),
            "forehead_to_cheek" : (0.88, 0.96, 1.04),
            "jaw_to_cheek"      : (0.82, 0.92, 1.02),
            "forehead_to_jaw"   : (0.92, 1.04, 1.16),
            "chin_taper"        : (0.55, 0.70, 0.82),
            "lower_n"           : (0.28, 0.33, 0.38),
            "brow_mouth_n"      : (0.34, 0.42, 0.50),
            "cheekbone_n"       : (0.62, 0.72, 0.82),
        },

        "Redondo": {
            # Cara casi tan ancha como larga. Pómulos amplios.
            # Mandíbula redondeada, suave. Frente ≈ mandíbula.
            # Mentón corto, redondeado. face_ratio alto (>0.85).
            "face_ratio"        : (0.85, 0.94, 1.04),
            "forehead_to_cheek" : (0.90, 0.97, 1.04),
            "jaw_to_cheek"      : (0.88, 0.95, 1.02),
            "forehead_to_jaw"   : (0.92, 1.02, 1.12),
            "chin_taper"        : (0.68, 0.80, 0.92),
            "lower_n"           : (0.28, 0.32, 0.37),
            "brow_mouth_n"      : (0.33, 0.40, 0.47),
            "cheekbone_n"       : (0.68, 0.78, 0.86),
        },

        "Cuadrado": {
            # Cara ancha. Mandíbula MUY marcada y angular, casi igual
            # que pómulos. Frente amplia. Mentón cuadrado (chin_taper alto).
            # face_ratio alto. La clave: jaw_to_cheek cercano a 1 Y
            # chin_taper alto (no se angosta).
            "face_ratio"        : (0.85, 0.95, 1.06),
            "forehead_to_cheek" : (0.92, 0.99, 1.06),
            "jaw_to_cheek"      : (0.94, 1.02, 1.10),
            "forehead_to_jaw"   : (0.88, 0.98, 1.08),
            "chin_taper"        : (0.78, 0.88, 0.98),
            "lower_n"           : (0.30, 0.34, 0.38),
            "brow_mouth_n"      : (0.34, 0.40, 0.46),
            "cheekbone_n"       : (0.68, 0.76, 0.84),
        },

        "Rectángulo": {
            # Cara notablemente más larga que ancha. Angular.
            # Frente, pómulos y mandíbula de ancho similar (como cuadrado
            # pero estirado). face_ratio bajo (<0.78). chin_taper alto.
            "face_ratio"        : (0.60, 0.70, 0.78),
            "forehead_to_cheek" : (0.90, 0.98, 1.06),
            "jaw_to_cheek"      : (0.88, 0.96, 1.06),
            "forehead_to_jaw"   : (0.90, 1.02, 1.12),
            "chin_taper"        : (0.72, 0.84, 0.96),
            "lower_n"           : (0.30, 0.35, 0.40),
            "brow_mouth_n"      : (0.38, 0.45, 0.52),
            "cheekbone_n"       : (0.52, 0.60, 0.70),
        },

        "Triángulo": {
            # Mandíbula CLARAMENTE más ancha que la frente.
            # forehead_to_jaw < 0.85 es la señal inequívoca.
            # jaw_to_cheek > 1.0 (mandíbula domina).
            # NOTA: con los landmarks corregidos, la frente ya mide bien.
            # Un rostro normal NO debería caer aquí a menos que la
            # mandíbula sea genuinamente más ancha que la frente.
            "face_ratio"        : (0.75, 0.86, 0.98),
            "forehead_to_cheek" : (0.70, 0.80, 0.88),
            "jaw_to_cheek"      : (1.02, 1.12, 1.22),
            "forehead_to_jaw"   : (0.62, 0.74, 0.85),
            "chin_taper"        : (0.72, 0.84, 0.96),
            "lower_n"           : (0.30, 0.35, 0.40),
            "brow_mouth_n"      : (0.36, 0.42, 0.48),
            "cheekbone_n"       : (0.62, 0.70, 0.80),
        },

        "Corazón": {
            # Frente CLARAMENTE más ancha que mandíbula.
            # forehead_to_jaw > 1.18 es la señal clave.
            # Mentón puntiagudo (chin_taper bajo).
            "face_ratio"        : (0.72, 0.82, 0.92),
            "forehead_to_cheek" : (0.98, 1.08, 1.18),
            "jaw_to_cheek"      : (0.64, 0.76, 0.88),
            "forehead_to_jaw"   : (1.18, 1.35, 1.52),
            "chin_taper"        : (0.40, 0.55, 0.68),
            "lower_n"           : (0.26, 0.31, 0.36),
            "brow_mouth_n"      : (0.36, 0.43, 0.50),
            "cheekbone_n"       : (0.60, 0.68, 0.76),
        },

        "Diamante": {
            # Pómulos son CLARAMENTE la zona más ancha.
            # Tanto frente como mandíbula son estrechas respecto a pómulos.
            # forehead_to_cheek < 0.88 Y jaw_to_cheek < 0.85.
            # Mentón puntiagudo.
            "face_ratio"        : (0.72, 0.82, 0.92),
            "forehead_to_cheek" : (0.74, 0.84, 0.92),
            "jaw_to_cheek"      : (0.68, 0.78, 0.88),
            "forehead_to_jaw"   : (0.94, 1.08, 1.20),
            "chin_taper"        : (0.42, 0.58, 0.72),
            "lower_n"           : (0.28, 0.33, 0.38),
            "brow_mouth_n"      : (0.36, 0.42, 0.48),
            "cheekbone_n"       : (0.60, 0.68, 0.78),
        },
    }

    @classmethod
    def _score_feature(cls, value: float, profile_range: Tuple[float, float, float]) -> float:
        """
        Calcula el score de una feature individual dado su valor y el rango
        del perfil (min, ideal, max).

        Retorna un valor entre -1.0 y +1.0:
            +1.0  → valor exactamente en el ideal
             0.0  → valor en el borde del rango aceptable
            -1.0  → valor muy fuera de rango

        Usa una función triangular suave:
            - Dentro del rango [min, max]: interpolación lineal hacia ideal = 1.0
            - Fuera del rango: penalización proporcional a la distancia
        """
        lo, ideal, hi = profile_range
        half_range = (hi - lo) / 2.0 + 1e-6

        if lo <= value <= hi:
            # Dentro del rango aceptable: score entre 0.0 y 1.0
            if value <= ideal:
                score = (value - lo) / (ideal - lo + 1e-6)
            else:
                score = (hi - value) / (hi - ideal + 1e-6)
            return max(0.0, min(1.0, score))
        else:
            # Fuera de rango: penalización proporcional
            if value < lo:
                penalty = (lo - value) / half_range
            else:
                penalty = (value - hi) / half_range
            return max(-1.0, -penalty)

    @classmethod
    def classify(cls, d: Dict[str, float]) -> Dict[str, any]:
        """
        Clasifica el tipo de rostro usando scoring vectorial.

        Parámetros
        ----------
        d : dict con todas las métricas de calculate_distances()

        Retorna
        -------
        {
            "tipo"        : str,          # nombre del tipo ganador
            "confianza"   : float,        # 0.0 – 1.0
            "scores"      : dict,         # score de cada tipo
            "features"    : dict,         # vector de features extraído
            "detalle"     : str,          # descripción legible
        }
        """
        # ── Paso 1: Extraer vector de features ──────────────────────────────
        features = {}
        for feat_name in cls.FEATURE_WEIGHTS:
            if feat_name in d:
                features[feat_name] = d[feat_name]
            else:
                features[feat_name] = 0.0

        # ── Paso 2: Calcular score ponderado para cada tipo ─────────────────
        scores = {}
        score_details = {}

        for face_type, profile in cls.PROFILES.items():
            total_score = 0.0
            total_weight = 0.0
            feat_scores = {}

            for feat_name, weight in cls.FEATURE_WEIGHTS.items():
                if feat_name in profile:
                    value = features[feat_name]
                    feat_score = cls._score_feature(value, profile[feat_name])
                    weighted = feat_score * weight
                    total_score += weighted
                    total_weight += weight
                    feat_scores[feat_name] = round(feat_score, 3)

            # Normalizar a [0, 1]
            if total_weight > 0:
                normalized = (total_score / total_weight + 1.0) / 2.0
            else:
                normalized = 0.0

            scores[face_type] = round(max(0.0, min(1.0, normalized)), 4)
            score_details[face_type] = feat_scores

        # ── Paso 3: Reglas de desempate discriminantes ──────────────────────
        #
        #  Cuando dos tipos tienen scores muy cercanos (diferencia < 0.05),
        #  aplicamos criterios binarios duros que son inequívocos para
        #  ciertos tipos. Esto resuelve las zonas grises.
        # ────────────────────────────────────────────────────────────────────

        sorted_types = sorted(scores.items(), key=lambda x: -x[1])
        best_type, best_score = sorted_types[0]
        second_type, second_score = sorted_types[1]

        # Solo aplica desempate si la diferencia es menor al 5%
        if (best_score - second_score) < 0.05:
            candidates = {best_type, second_type}

            # Desempate Cuadrado vs Redondo → chin_taper alto = cuadrado
            if candidates == {"Cuadrado", "Redondo"}:
                if features.get("chin_taper", 0) > 0.82:
                    best_type = "Cuadrado"
                else:
                    best_type = "Redondo"
                best_score = scores[best_type]

            # Desempate Ovalado vs Rectángulo → face_ratio decide
            elif candidates == {"Ovalado", "Rectángulo"}:
                if features.get("face_ratio", 0) < 0.73:
                    best_type = "Rectángulo"
                else:
                    best_type = "Ovalado"
                best_score = scores[best_type]

            # Desempate Corazón vs Diamante → forehead_to_cheek decide
            elif candidates == {"Corazón", "Diamante"}:
                if features.get("forehead_to_cheek", 0) > 0.98:
                    best_type = "Corazón"
                else:
                    best_type = "Diamante"
                best_score = scores[best_type]

            # Desempate Triángulo vs Cuadrado → forehead_to_jaw decide
            elif candidates == {"Triángulo", "Cuadrado"}:
                if features.get("forehead_to_jaw", 0) < 0.82:
                    best_type = "Triángulo"
                else:
                    best_type = "Cuadrado"
                best_score = scores[best_type]

            # Desempate Ovalado vs Redondo → face_ratio decide
            elif candidates == {"Ovalado", "Redondo"}:
                if features.get("face_ratio", 0) > 0.85:
                    best_type = "Redondo"
                else:
                    best_type = "Ovalado"
                best_score = scores[best_type]

        # ── Paso 4: Calcular confianza ──────────────────────────────────────
        #  La confianza refleja qué tan separado está el ganador del segundo.
        #  Si el margen es grande, alta confianza. Si es bajo, baja confianza.
        margin = best_score - second_score
        confianza = min(1.0, best_score * 0.6 + margin * 4.0)
        confianza = round(max(0.0, min(1.0, confianza)), 3)

        # ── Paso 5: Generar descripción ─────────────────────────────────────
        desc_map = {
            "Ovalado"    : "Rostro equilibrado, ligeramente más largo que ancho. "
                           "Pómulos son la zona más ancha, mentón se suaviza gradualmente.",
            "Redondo"    : "Rostro casi tan ancho como largo, líneas suaves. "
                           "Pómulos amplios y mentón redondeado.",
            "Cuadrado"   : "Rostro de proporciones anchas con mandíbula angular y marcada. "
                           "Frente, pómulos y mandíbula de ancho similar.",
            "Rectángulo" : "Rostro notablemente más largo que ancho, angular. "
                           "Línea de mandíbula definida, frente amplia.",
            "Triángulo"  : "Mandíbula es la zona más ancha del rostro. "
                           "Frente más estrecha que la línea mandibular.",
            "Corazón"    : "Frente amplia que se angosta hacia un mentón puntiagudo. "
                           "La mandíbula es significativamente más estrecha que la frente.",
            "Diamante"   : "Pómulos prominentes dominan el ancho facial. "
                           "Tanto frente como mandíbula son estrechas.",
        }

        return {
            "tipo"       : best_type,
            "confianza"  : confianza,
            "scores"     : scores,
            "features"   : features,
            "detalle"    : desc_map.get(best_type, ""),
            "ranking"    : [(t, s) for t, s in sorted_types],
        }


# ──────────────────────────────────────────────────────────────────────────────
# 7. FUNCIÓN: draw_results()
# ──────────────────────────────────────────────────────────────────────────────
def draw_results(
    frame   : np.ndarray,
    bbox    : Optional[Tuple[int, int, int, int]],
    lm      : Optional[Dict],
    dists   : Optional[Dict[str, float]],
    classification: Optional[Dict] = None,
    debug   : bool = False
) -> np.ndarray:
    """
    Dibuja sobre el frame todos los elementos visuales.
    Ahora incluye el panel de clasificación con scores.
    """
    overlay = frame.copy()

    # ── Bounding box ─────────────────────────────────────────────────────────
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), COLOR_BOX, 2)
        cv2.putText(overlay, "FACE DETECTED", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BOX, 1, cv2.LINE_AA)

    # ── Landmarks y líneas ───────────────────────────────────────────────────
    if lm:
        raw = lm.get("_raw")

        if debug and raw:
            ih, iw = frame.shape[:2]
            for landmark in raw:
                px = int(landmark.x * iw)
                py = int(landmark.y * ih)
                cv2.circle(overlay, (px, py), 1, (180, 180, 180), -1)

        key_points = {k: v for k, v in lm.items() if not k.startswith("_")}
        for name, (px, py) in key_points.items():
            cv2.circle(overlay, (px, py), 4, COLOR_LANDMARK, -1)
            cv2.circle(overlay, (px, py), 5, (255, 255, 255), 1)

        # Líneas de medición por zona
        measurement_lines = [
            ("forehead_left",    "forehead_right",    "FH"),
            ("cheekbone_left",   "cheekbone_right",   "CK"),
            ("jaw_angle_left",   "jaw_angle_right",   "JW"),
            ("jaw_low_left",     "jaw_low_right",     "JL"),
            ("forehead_top",     "chin",              "H"),
            ("eye_left_outer",   "eye_right_outer",   "E"),
            ("nose_base_left",   "nose_base_right",   "N"),
            ("mouth_left",       "mouth_right",       "M"),
        ]
        for p1_name, p2_name, label in measurement_lines:
            if p1_name in lm and p2_name in lm:
                p1 = lm[p1_name]
                p2 = lm[p2_name]
                cv2.line(overlay, p1, p2, COLOR_LINE, 1, cv2.LINE_AA)
                cv2.circle(overlay, p1, 3, COLOR_LINE, -1)
                cv2.circle(overlay, p2, 3, COLOR_LINE, -1)
                mx = (p1[0] + p2[0]) // 2
                my = (p1[1] + p2[1]) // 2
                cv2.putText(overlay, label, (mx + 4, my - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_LINE, 1, cv2.LINE_AA)

        # Línea vertical frente→mentón
        if "forehead_top" in lm and "chin" in lm:
            cv2.line(overlay, lm["forehead_top"], lm["chin"],
                     (120, 200, 255), 1, cv2.LINE_AA)

    # ── Panel de métricas ────────────────────────────────────────────────────
    if dists:
        panel_x, panel_y = 10, 10
        panel_w, panel_h = 280, 340
        roi = overlay[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w]
        bg = np.zeros_like(roi)
        bg[:] = COLOR_TEXT_BG
        cv2.addWeighted(roi, 0.3, bg, 0.7, 0, roi)
        overlay[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w] = roi

        cv2.rectangle(overlay,
                      (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h),
                      COLOR_BOX, 1)

        cv2.putText(overlay, " FACE METRICS",
                    (panel_x + 6, panel_y + 18),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, COLOR_BOX, 1, cv2.LINE_AA)
        cv2.line(overlay,
                 (panel_x + 4, panel_y + 24),
                 (panel_x + panel_w - 4, panel_y + 24),
                 COLOR_BOX, 1)

        metrics = [
            ("Altura cara",       f"{dists['face_height_px']} px"),
            ("Ancho max",         f"{dists['face_width_px']} px"),
            ("Ratio A/H",         f"{dists['face_ratio']:.3f}"),
            ("Frente ancho",      f"{dists['forehead_width_px']} px"),
            ("Pomulos ancho",     f"{dists['cheekbone_width_px']} px"),
            ("Mandibula ancho",   f"{dists['jaw_angle_width_px']} px"),
            ("Frente/Pomulos",    f"{dists['forehead_to_cheek']:.3f}"),
            ("Mand./Pomulos",     f"{dists['jaw_to_cheek']:.3f}"),
            ("Frente/Mand.",      f"{dists['forehead_to_jaw']:.3f}"),
            ("Estrechamiento",    f"{dists['chin_taper']:.3f}"),
            ("Cejas→Boca norm",   f"{dists['brow_mouth_n']:.3f}"),
            ("Tercio inf. norm",  f"{dists['lower_n']:.3f}"),
            ("EAR (fatiga)",      f"{dists['eye_aspect_ratio']}"),
        ]

        for i, (label, value) in enumerate(metrics):
            ty = panel_y + 42 + i * 22
            cv2.putText(overlay, label + ":",
                        (panel_x + 8, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_TEXT, 1, cv2.LINE_AA)
            cv2.putText(overlay, value,
                        (panel_x + 170, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, COLOR_ACCENT, 1, cv2.LINE_AA)

        ear = dists["eye_aspect_ratio"]
        if ear < 0.15:
            cv2.putText(overlay, "EYES CLOSED",
                        (panel_x + 6, panel_y + panel_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WARN, 2, cv2.LINE_AA)

    # ── Panel de clasificación (lado derecho) ────────────────────────────────
    if classification:
        fw = frame.shape[1]
        cp_w, cp_h = 260, 260
        cp_x = fw - cp_w - 10
        cp_y = 10

        # Fondo
        roi2 = overlay[cp_y:cp_y + cp_h, cp_x:cp_x + cp_w]
        bg2 = np.zeros_like(roi2)
        bg2[:] = COLOR_TEXT_BG
        cv2.addWeighted(roi2, 0.3, bg2, 0.7, 0, roi2)
        overlay[cp_y:cp_y + cp_h, cp_x:cp_x + cp_w] = roi2

        cv2.rectangle(overlay,
                      (cp_x, cp_y), (cp_x + cp_w, cp_y + cp_h),
                      (0, 200, 200), 1)

        cv2.putText(overlay, " FACE SHAPE",
                    (cp_x + 6, cp_y + 18),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 200, 200), 1, cv2.LINE_AA)
        cv2.line(overlay,
                 (cp_x + 4, cp_y + 24),
                 (cp_x + cp_w - 4, cp_y + 24),
                 (0, 200, 200), 1)

        tipo = classification["tipo"]
        conf = classification["confianza"]

        # Tipo ganador grande
        cv2.putText(overlay, tipo.upper(),
                    (cp_x + 10, cp_y + 52),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 200), 2, cv2.LINE_AA)

        # Barra de confianza
        bar_x = cp_x + 10
        bar_y = cp_y + 62
        bar_w = cp_w - 20
        bar_h = 12
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (80, 80, 80), -1)
        fill_w = int(bar_w * conf)
        color_bar = (0, 255, 100) if conf > 0.6 else (0, 200, 255) if conf > 0.35 else (0, 80, 255)
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                      color_bar, -1)
        cv2.putText(overlay, f"{conf*100:.0f}%",
                    (bar_x + bar_w + 4, bar_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color_bar, 1, cv2.LINE_AA)

        # Ranking de todos los tipos
        ranking = classification.get("ranking", [])
        for i, (t, s) in enumerate(ranking[:7]):
            ty = cp_y + 90 + i * 22
            # Nombre
            col = (0, 255, 200) if t == tipo else (180, 180, 180)
            cv2.putText(overlay, t,
                        (cp_x + 10, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
            # Mini barra
            mini_w = int(120 * s)
            cv2.rectangle(overlay,
                          (cp_x + 110, ty - 8),
                          (cp_x + 110 + mini_w, ty),
                          col, -1)
            cv2.putText(overlay, f"{s:.2f}",
                        (cp_x + 235, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1, cv2.LINE_AA)

    # ── Controles ────────────────────────────────────────────────────────────
    h_frame = frame.shape[0]
    controls = ["[Q] Salir", "[S] Snapshot+JSON", "[D] Debug landmarks"]
    for i, txt in enumerate(controls):
        cv2.putText(overlay, txt,
                    (10, h_frame - 12 - i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    return frame


# ──────────────────────────────────────────────────────────────────────────────
# 8. FUNCIÓN: export_data()
# ──────────────────────────────────────────────────────────────────────────────
def export_to_json(dists: Dict, filepath: str = "face_data.json") -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(dists, f, indent=2, ensure_ascii=False)


def send_via_socket(
    dists: Dict, host: str = "127.0.0.1", port: int = 9000, protocol: str = "udp"
) -> None:
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
        pass


# ──────────────────────────────────────────────────────────────────────────────
# 9. CLASE PRINCIPAL: FaceAnalyzer
# ──────────────────────────────────────────────────────────────────────────────
class FaceAnalyzer:
    def __init__(self, camera_index: int = 0, analyze_time: float = 8.0):
        self.analyze_time = analyze_time
        self.classifier = FaceShapeClassifier()

        # Buffer de clasificaciones para promediar durante los 8 segundos
        self.classification_buffer: List[Dict] = []

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print("[ERROR] No se pudo abrir la cámara. Verifica permisos o índice.")
            self.valid = False
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

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

        self.fps_buffer: List[float] = []
        self.last_dists: Optional[Dict[str, float]] = None
        self.last_classification: Optional[Dict] = None
        self.face_detection_start_time: Optional[float] = None
        self.debug_mode = False

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple], Optional[Dict]]:
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        lm = get_landmarks(frame_rgb, self.face_landmarker)

        bbox = None
        dists = None
        classification = None

        if lm:
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

            # Clasificar en cada frame y acumular
            classification = FaceShapeClassifier.classify(dists)
            self.last_classification = classification
            self.classification_buffer.append(classification)

            # Consola cada ~0.5s
            if int(time.time() * 2) % 2 == 0:
                print(
                    f"\r[METRICS] H={dists['face_height_px']:>6.1f}px  "
                    f"W={dists['face_width_px']:>6.1f}px  "
                    f"Ratio={dists['face_ratio']:.3f}  "
                    f"FH/CK={dists['forehead_to_cheek']:.3f}  "
                    f"JW/CK={dists['jaw_to_cheek']:.3f}  "
                    f"→ {classification['tipo']} ({classification['confianza']:.0%})",
                    end="", flush=True
                )

        frame_rgb.flags.writeable = True
        frame = draw_results(frame, bbox, lm, dists, classification, debug=self.debug_mode)

        return frame, bbox, dists

    def _get_consensus_classification(self) -> Optional[Dict]:
        """
        Calcula la clasificación final promediando los scores acumulados
        durante los 8 segundos de análisis.

        Esto elimina fluctuaciones frame a frame y da un resultado
        mucho más estable que tomar solo el último frame.
        """
        if not self.classification_buffer:
            return self.last_classification

        # Promediar los scores de todos los frames
        all_types = list(FaceShapeClassifier.PROFILES.keys())
        avg_scores = {t: 0.0 for t in all_types}
        avg_features = {f: 0.0 for f in FaceShapeClassifier.FEATURE_WEIGHTS}

        n = len(self.classification_buffer)
        for clf in self.classification_buffer:
            for t in all_types:
                avg_scores[t] += clf["scores"].get(t, 0.0)
            for f in avg_features:
                avg_features[f] += clf["features"].get(f, 0.0)

        for t in all_types:
            avg_scores[t] /= n
        for f in avg_features:
            avg_features[f] /= n

        # Determinar ganador por promedio
        sorted_types = sorted(avg_scores.items(), key=lambda x: -x[1])
        best_type, best_score = sorted_types[0]
        second_type, second_score = sorted_types[1]

        margin = best_score - second_score
        confianza = min(1.0, best_score * 0.6 + margin * 4.0)
        confianza = round(max(0.0, min(1.0, confianza)), 3)

        # Conteo de votos (cuántos frames dieron cada tipo)
        vote_count = {}
        for clf in self.classification_buffer:
            t = clf["tipo"]
            vote_count[t] = vote_count.get(t, 0) + 1

        return {
            "tipo"       : best_type,
            "confianza"  : confianza,
            "scores"     : {t: round(s, 4) for t, s in avg_scores.items()},
            "features"   : {f: round(v, 4) for f, v in avg_features.items()},
            "ranking"    : [(t, round(s, 4)) for t, s in sorted_types],
            "n_frames"   : n,
            "votes"      : vote_count,
        }

    def check_timer_and_classify(self, frame: np.ndarray, face_detected: bool) -> bool:
        if face_detected:
            if self.face_detection_start_time is None:
                self.face_detection_start_time = time.time()
                self.classification_buffer.clear()
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
            self.classification_buffer.clear()

        return False

    def _print_final_classification(self) -> None:
        print("\n" + "=" * 60)
        print("  ANÁLISIS FACIAL COMPLETADO")
        print("=" * 60)

        consensus = self._get_consensus_classification()
        if consensus:
            print(f"\n  TIPO DE ROSTRO: {consensus['tipo']}")
            print(f"  CONFIANZA:      {consensus['confianza']:.0%}")
            print(f"  FRAMES ANALIZADOS: {consensus.get('n_frames', '?')}")

            print(f"\n  Votos por tipo:")
            for t, v in sorted(consensus.get("votes", {}).items(), key=lambda x: -x[1]):
                print(f"    {t:15s} → {v} frames")

            print(f"\n  Scores promedio:")
            for t, s in consensus["ranking"]:
                bar = "█" * int(s * 30)
                print(f"    {t:15s} {s:.4f}  {bar}")

            print(f"\n  Features promedio:")
            for f, v in consensus["features"].items():
                print(f"    {f:22s} = {v:.4f}")

            # Guardar resultado completo
            self.last_consensus = consensus
        else:
            print("  ERROR: No se generaron clasificaciones.")

        print("=" * 60 + "\n")

    def run(self) -> None:
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

            if self.check_timer_and_classify(frame, bbox is not None):
                break

            cv2.imshow("Reconocimiento Facial | OpenCV + MediaPipe", frame)
            cv2.waitKey(1)

    def close(self) -> None:
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        if hasattr(self, 'face_landmarker') and self.face_landmarker:
            self.face_landmarker.close()
        cv2.destroyAllWindows()

        # Exportar resultado final
        export_data = {}
        if hasattr(self, 'last_dists') and self.last_dists is not None:
            export_data["metrics"] = self.last_dists
        if hasattr(self, 'last_consensus'):
            export_data["classification"] = {
                "tipo"      : self.last_consensus["tipo"],
                "confianza" : self.last_consensus["confianza"],
                "scores"    : self.last_consensus["scores"],
                "features"  : self.last_consensus["features"],
                "votes"     : self.last_consensus.get("votes", {}),
                "n_frames"  : self.last_consensus.get("n_frames", 0),
            }

        if export_data:
            export_to_json(export_data, "face_data_last.json")
            print("[INFO] Métricas y clasificación guardadas en 'face_data_last.json'")
        print("[INFO] Programa finalizado correctamente.")


def main() -> None:
    print("=" * 60)
    print("  RECONOCIMIENTO FACIAL EN TIEMPO REAL")
    print("  OpenCV + MediaPipe  |  Clasificador Vectorial v2")
    print("=" * 60)
    print("  Controles:")
    print("    El análisis dura 8 segundos automáticamente")
    print("    tras detectar un rostro.")
    print("=" * 60)

    analyzer = FaceAnalyzer(analyze_time=8.0)
    try:
        analyzer.run()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupción por teclado.")
    finally:
        analyzer.close()


if __name__ == "__main__":
    main()

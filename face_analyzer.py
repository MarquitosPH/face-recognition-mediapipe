"""
face_analyzer.py — Módulo de análisis facial.
Procesa imágenes estáticas (subidas por el usuario) para clasificar la forma del rostro.
Usa MediaPipe Face Mesh + clasificador vectorial.

TIPOS DE ROSTRO SOPORTADOS (6):
    Ovalado · Redondo · Cuadrado · Largo · Corazón · Diamante

CAMBIOS v3 (diagnóstico con 18 celebridades — sesgo sistémico a Diamante):
    CAUSA RAÍZ: Los landmarks 116/345 daban lecturas de pómulos más anchas
    de lo real para caras "normales", empujándolas a Diamante. Al mismo tiempo,
    caras con pómulos verdaderamente prominentes (Diamante real: Pattinson,
    Murphy) quedaban FUERA del rango mínimo y se clasificaban como Ovalado.
    La paradoja: el mismo bug que sobreclasificaba Diamante, expulsaba a los
    Diamante reales.

    FIX 1 — Landmarks cheekbone: 116/345 → 227/447
        Puntos 227/447 corresponden al arco cigomático lateral en MediaPipe,
        dan lecturas más calibradas de la anchura real del pómulo sin capturar
        tejido blando extra del carrillo.
        También: forehead ahora usa 54/284 (cola de ceja, más exterior)
        en lugar de 46/276 para una lectura de frente más amplia y estable.

    FIX 2 — cheekbone_width = promedio de dos pares de landmarks
        Para reducir el impacto de un único punto ruidoso, cheekbone_width
        se calcula como promedio de (227→447) y (116→345). Más robusto.

    FIX 3 — Perfiles redibujados con rangos exclusivos por tipo
        Diamante ahora requiere cheekbone_n ESTRICTAMENTE alto (>0.68 mínimo)
        Y forehead_to_cheek ESTRICTAMENTE bajo (<0.88 máximo).
        Los otros tipos amplían sus rangos para no colapsar hacia Diamante.

    FIX 4 — Regla hard-exclusion en classify()
        Antes de scoring, si forehead_to_cheek > 1.05 → excluir Diamante.
        Esto previene que caras con frente claramente más ancha que pómulos
        sean forzadas a Diamante por otras features secundarias.

CAMBIOS v2:
    - Landmarks cheekbone: 234/454 → 116/345
    - Se eliminó "Triángulo", "Rectángulo" → "Largo"
    - Perfiles recalibrados con tabla comparativa
    - forehead_n añadido como feature de scoring
    - Reglas de desempate de 5 → 10 pares
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from typing import Optional, Tuple, Dict

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── Landmarks MediaPipe Face Mesh (478 puntos) ──────────────────────────────
#
# CHEEKBONE — dos pares para promedio robusto:
#   Principal  : 227 / 447  → arco cigomático lateral (lectura calibrada)
#   Secundario : 116 / 345  → zygomatic arch medio
#   cheekbone_width = promedio de ambas distancias
#   Esto reduce el impacto de un solo punto ruidoso en perfiles de iluminación
#   o angulación de cámara.
#
# FOREHEAD — 54/284 (cola exterior de ceja):
#   Más estable que 46/276 porque está más separado del centro y produce
#   lecturas de frente consistentes con las comparaciones de la tabla.
# ─────────────────────────────────────────────────────────────────────────────
LANDMARKS_IDX = {
    # ── Contorno vertical ─────────────────────────────────────────────────
    "forehead_top"        : 10,    # punto más alto de la frente (centro)
    "chin"                : 152,   # mentón (centro inferior)

    # ── Anchuras horizontales — frente ────────────────────────────────────
    "forehead_left"       : 54,    # cola exterior ceja izquierda (v3: era 46)
    "forehead_right"      : 284,   # cola exterior ceja derecha   (v3: era 276)

    # ── Anchuras horizontales — pómulos (dos pares para promedio) ─────────
    "cheekbone_left"      : 227,   # arco cigomático lateral izq  (v3: PRINCIPAL)
    "cheekbone_right"     : 447,   # arco cigomático lateral der  (v3: PRINCIPAL)
    "cheekbone2_left"     : 116,   # arco cigomático medio izq    (v3: SECUNDARIO)
    "cheekbone2_right"    : 345,   # arco cigomático medio der    (v3: SECUNDARIO)

    # ── Anchuras horizontales — mandíbula ─────────────────────────────────
    "jaw_mid_left"        : 172,   # mandíbula media izquierda
    "jaw_mid_right"       : 397,   # mandíbula media derecha
    "jaw_low_left"        : 150,   # mandíbula baja izquierda
    "jaw_low_right"       : 379,   # mandíbula baja derecha
    "jaw_angle_left"      : 132,   # ángulo mandibular izquierdo (gonion)
    "jaw_angle_right"     : 361,   # ángulo mandibular derecho  (gonion)

    # ── Ojos ──────────────────────────────────────────────────────────────
    "eye_left_outer"      : 33,
    "eye_left_inner"      : 133,
    "eye_right_inner"     : 362,
    "eye_right_outer"     : 263,
    "eye_left_top"        : 159,
    "eye_left_bot"        : 145,
    "eye_right_top"       : 386,
    "eye_right_bot"       : 374,

    # ── Nariz ─────────────────────────────────────────────────────────────
    "nose_tip"            : 1,
    "nose_bridge_top"     : 6,
    "nose_base_left"      : 129,
    "nose_base_right"     : 358,

    # ── Boca ──────────────────────────────────────────────────────────────
    "mouth_left"          : 61,
    "mouth_right"         : 291,
    "mouth_top"           : 13,
    "mouth_bot"           : 14,

    # ── Cejas / tercios ───────────────────────────────────────────────────
    "brow_left_inner"     : 107,
    "brow_left_peak"      : 70,
    "brow_left_outer"     : 46,
    "brow_right_inner"    : 336,
    "brow_right_peak"     : 300,
    "brow_right_outer"    : 276,
    "glabella"            : 9,
    "nose_base_center"    : 94,
}


def _find_model_path():
    """Busca el modelo face_landmarker.task en varias ubicaciones."""
    possible = [
        os.path.join(os.path.dirname(__file__), "face_landmarker.task"),
        os.path.join(os.path.dirname(__file__), "models", "face_landmarker.task"),
        "face_landmarker.task",
    ]
    for p in possible:
        if os.path.exists(p):
            return p
    return None


def get_landmarks_from_image(frame_rgb: np.ndarray) -> Optional[Dict[str, Tuple[int, int]]]:
    """
    Extrae landmarks de una imagen estática usando MediaPipe FaceMesh (legacy API).
    Compatible sin necesidad del archivo .task.
    """
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return None

        ih, iw = frame_rgb.shape[:2]
        face_lm = results.multi_face_landmarks[0].landmark

        landmarks_px = {}
        for name, idx in LANDMARKS_IDX.items():
            if idx < len(face_lm):
                lm = face_lm[idx]
                landmarks_px[name] = (int(lm.x * iw), int(lm.y * ih))

        return landmarks_px


def calculate_distances(lm: Dict[str, Tuple[int, int]], frame_shape: Tuple[int, int]) -> Dict[str, float]:
    """Calcula todas las métricas geométricas del rostro."""

    def dist(p1: str, p2: str) -> float:
        a = np.array(lm[p1], dtype=np.float64)
        b = np.array(lm[p2], dtype=np.float64)
        return float(np.linalg.norm(a - b))

    def vertical_dist(p1: str, p2: str) -> float:
        return abs(float(lm[p1][1]) - float(lm[p2][1]))

    ih, iw = frame_shape

    face_height       = dist("forehead_top",   "chin")
    forehead_width    = dist("forehead_left",   "forehead_right")

    # Cheekbone = promedio de par principal (227/447) y secundario (116/345)
    cheek_main        = dist("cheekbone_left",  "cheekbone_right")
    cheek_secondary   = dist("cheekbone2_left", "cheekbone2_right")
    cheekbone_width   = (cheek_main + cheek_secondary) / 2.0

    jaw_mid_width     = dist("jaw_mid_left",    "jaw_mid_right")
    jaw_low_width     = dist("jaw_low_left",    "jaw_low_right")
    jaw_angle_width   = dist("jaw_angle_left",  "jaw_angle_right")
    eye_dist          = dist("eye_left_outer",  "eye_right_outer")
    l_eye_open        = dist("eye_left_top",    "eye_left_bot")
    r_eye_open        = dist("eye_right_top",   "eye_right_bot")
    nose_width        = dist("nose_base_left",  "nose_base_right")
    mouth_width       = dist("mouth_left",      "mouth_right")
    mouth_open        = dist("mouth_top",       "mouth_bot")

    upper_third  = vertical_dist("forehead_top",    "glabella")
    middle_third = vertical_dist("glabella",         "nose_base_center")
    lower_third  = vertical_dist("nose_base_center", "chin")

    brow_avg_y   = (lm["brow_left_peak"][1] + lm["brow_right_peak"][1]) / 2.0
    mouth_avg_y  = (lm["mouth_top"][1]       + lm["mouth_bot"][1])       / 2.0
    chin_y       = float(lm["chin"][1])

    brow_to_mouth = abs(mouth_avg_y - brow_avg_y)
    brow_to_chin  = abs(chin_y      - brow_avg_y)

    brow_l      = lm["eye_left_top"][1]  - lm["brow_left_peak"][1]
    brow_r      = lm["eye_right_top"][1] - lm["brow_right_peak"][1]
    brow_height = (brow_l + brow_r) / 2.0

    ear = (l_eye_open + r_eye_open) / (2.0 * eye_dist + 1e-6)

    fh = face_height    + 1e-6
    cw = cheekbone_width + 1e-6

    max_width  = max(forehead_width, cheekbone_width, jaw_angle_width)
    face_ratio = max_width / fh

    forehead_n  = forehead_width  / fh
    cheekbone_n = cheekbone_width / fh
    jaw_mid_n   = jaw_mid_width   / fh
    jaw_angle_n = jaw_angle_width / fh

    forehead_to_cheek = forehead_width   / cw
    jaw_to_cheek      = jaw_angle_width  / cw
    forehead_to_jaw   = forehead_width   / (jaw_angle_width + 1e-6)

    chin_taper  = jaw_low_width / (jaw_mid_width + 1e-6)

    upper_n      = upper_third  / fh
    middle_n     = middle_third / fh
    lower_n      = lower_third  / fh

    brow_mouth_n = brow_to_mouth / fh
    brow_chin_n  = brow_to_chin  / fh

    mouth_ratio  = mouth_width / cw
    nose_ratio   = nose_width  / cw

    return {
        # ── Absolutas (px) ────────────────────────────────────────────────
        "face_height_px"      : round(face_height,    1),
        "face_width_px"       : round(max_width,      1),
        "forehead_width_px"   : round(forehead_width, 1),
        "cheekbone_width_px"  : round(cheekbone_width,1),
        "jaw_mid_width_px"    : round(jaw_mid_width,  1),
        "jaw_angle_width_px"  : round(jaw_angle_width,1),
        "jaw_low_width_px"    : round(jaw_low_width,  1),
        "eye_distance_px"     : round(eye_dist,       1),
        "nose_width_px"       : round(nose_width,     1),
        "mouth_width_px"      : round(mouth_width,    1),
        # ── Ratios / normalizadas ─────────────────────────────────────────
        "face_ratio"          : round(face_ratio,         4),
        "forehead_n"          : round(forehead_n,         4),
        "cheekbone_n"         : round(cheekbone_n,        4),
        "jaw_mid_n"           : round(jaw_mid_n,          4),
        "jaw_angle_n"         : round(jaw_angle_n,        4),
        "forehead_to_cheek"   : round(forehead_to_cheek,  4),
        "jaw_to_cheek"        : round(jaw_to_cheek,       4),
        "forehead_to_jaw"     : round(forehead_to_jaw,    4),
        "chin_taper"          : round(chin_taper,         4),
        "upper_n"             : round(upper_n,            4),
        "middle_n"            : round(middle_n,           4),
        "lower_n"             : round(lower_n,            4),
        "brow_mouth_n"        : round(brow_mouth_n,       4),
        "brow_chin_n"         : round(brow_chin_n,        4),
        "mouth_ratio"         : round(mouth_ratio,        4),
        "nose_ratio"          : round(nose_ratio,         4),
        "eye_aspect_ratio"    : round(ear,                4),
    }


class FaceShapeClassifier:
    """
    Clasificador geométrico de tipo de rostro basado en scoring vectorial.

    v3 — Perfiles redibujados para eliminar sesgo sistémico hacia Diamante.
    Principio clave: Diamante es el tipo MÁS RESTRICTIVO (rangos estrechos,
    hard-exclusion si forehead_to_cheek > 1.05). Ovalado es el MÁS AMPLIO.
    """

    # ── Pesos de features ─────────────────────────────────────────────────────
    FEATURE_WEIGHTS = {
        "face_ratio"        : 3.0,
        "forehead_to_jaw"   : 3.5,   # KEY: separa Corazón del resto
        "jaw_to_cheek"      : 3.5,   # KEY: separa Cuadrado del resto
        "forehead_to_cheek" : 3.5,   # KEY: separa Diamante/Corazón
        "chin_taper"        : 2.5,   # KEY: Cuadrado plano vs puntudos
        "brow_mouth_n"      : 2.5,   # KEY v4: Redondo vs Largo (cara compacta vs larga)
        "cheekbone_n"       : 2.0,   # Diamante/Redondo vs estrechos
        "lower_n"           : 2.0,   # KEY v4: Redondo (mentón corto) vs Largo
        "face_ratio"        : 1.0,   # mínimo: solo desempate grueso, no discrimina solo
        "forehead_n"        : 1.0,   # anchura frente directa
    }

    # ── Perfiles por tipo de rostro ───────────────────────────────────────────
    #
    # v5 — Fix sesgo a Redondo.
    #
    # DIAGNÓSTICO v4:
    #   Se desplazó face_ratio ~0.12 hacia abajo para todos los tipos.
    #   El rango de Redondo (0.74-0.98) quedó tan central que captura
    #   la mayoría de rostros. Redondo se convirtió en el nuevo Ovalado.
    #
    # SOLUCIÓN v5:
    #   1. Redondo ya NO depende de face_ratio como KEY.
    #      Sus discriminadores son brow_mouth_n BAJO y lower_n BAJO.
    #      (cara compacta = distancia ceja→boca corta, mentón corto).
    #   2. Ovalado recupera face_ratio centrado y brow_mouth_n moderado.
    #   3. Se añade exclusión: brow_mouth_n > 0.44 → excluir Redondo.
    #   4. face_ratio baja a peso 1.0 — ya no es discriminador primario.
    #   5. brow_mouth_n y lower_n suben a 2.5 y 2.0 respectivamente.
    #
    # IMPORTANTE: usar /api/debug-metrics para ver valores reales de tu sistema
    # y afinar los rangos con datos reales en lugar de estimados.
    PROFILES = {

        # ── OVALADO ──────────────────────────────────────────────────────────
        # Tipo equilibrado. brow_mouth_n MODERADO.
        # Frente ≈ pómulos, mandíbula gradual.
        "Ovalado": {
            "face_ratio"        : (0.62, 0.74, 0.86),
            "forehead_n"        : (0.30, 0.40, 0.50),
            "forehead_to_cheek" : (0.90, 1.00, 1.10),
            "jaw_to_cheek"      : (0.84, 0.94, 1.04),
            "forehead_to_jaw"   : (0.94, 1.06, 1.18),
            "chin_taper"        : (0.58, 0.70, 0.82),
            "lower_n"           : (0.27, 0.34, 0.41),
            "brow_mouth_n"      : (0.37, 0.44, 0.51),  # MODERADO — no compacto
            "cheekbone_n"       : (0.52, 0.62, 0.72),
        },

        # ── REDONDO ──────────────────────────────────────────────────────────
        # KEY discriminadores: brow_mouth_n MUY BAJO (cara compacta)
        # y lower_n MUY BAJO (mentón corto). NO depende de face_ratio.
        # Si brow_mouth_n > 0.44 → se excluye en classify().
        "Redondo": {
            "face_ratio"        : (0.72, 0.84, 0.96),
            "forehead_n"        : (0.40, 0.50, 0.60),
            "forehead_to_cheek" : (0.86, 0.97, 1.08),
            "jaw_to_cheek"      : (0.88, 0.99, 1.10),
            "forehead_to_jaw"   : (0.82, 0.96, 1.10),
            "chin_taper"        : (0.70, 0.82, 0.94),
            "lower_n"           : (0.18, 0.25, 0.32),  # KEY: mentón MUY corto
            "brow_mouth_n"      : (0.24, 0.32, 0.40),  # KEY: cara MUY compacta
            "cheekbone_n"       : (0.58, 0.70, 0.82),
        },

        # ── CUADRADO ─────────────────────────────────────────────────────────
        # jaw_to_cheek ALTO + chin_taper ALTO = mandíbula cuadrada y plana.
        "Cuadrado": {
            "face_ratio"        : (0.68, 0.80, 0.92),
            "forehead_n"        : (0.38, 0.48, 0.58),
            "forehead_to_cheek" : (0.88, 1.00, 1.12),
            "jaw_to_cheek"      : (0.94, 1.05, 1.16),  # KEY
            "forehead_to_jaw"   : (0.80, 0.94, 1.08),
            "chin_taper"        : (0.82, 0.92, 1.02),  # KEY
            "lower_n"           : (0.24, 0.31, 0.38),
            "brow_mouth_n"      : (0.32, 0.40, 0.48),
            "cheekbone_n"       : (0.56, 0.66, 0.76),
        },

        # ── LARGO ────────────────────────────────────────────────────────────
        # brow_mouth_n ALTO (cara larga) + face_ratio BAJO.
        "Largo": {
            "face_ratio"        : (0.44, 0.57, 0.70),
            "forehead_n"        : (0.26, 0.35, 0.44),
            "forehead_to_cheek" : (0.84, 0.96, 1.08),
            "jaw_to_cheek"      : (0.76, 0.89, 1.02),
            "forehead_to_jaw"   : (0.88, 1.04, 1.20),
            "chin_taper"        : (0.60, 0.74, 0.88),
            "lower_n"           : (0.30, 0.38, 0.46),
            "brow_mouth_n"      : (0.48, 0.56, 0.64),  # KEY: cara muy larga
            "cheekbone_n"       : (0.44, 0.55, 0.66),
        },

        # ── CORAZÓN ──────────────────────────────────────────────────────────
        # forehead_to_jaw ALTO + chin_taper BAJO = señal inequívoca.
        "Corazón": {
            "face_ratio"        : (0.58, 0.70, 0.82),
            "forehead_n"        : (0.40, 0.52, 0.64),
            "forehead_to_cheek" : (1.00, 1.13, 1.26),  # KEY
            "jaw_to_cheek"      : (0.50, 0.63, 0.76),  # KEY
            "forehead_to_jaw"   : (1.16, 1.38, 1.60),  # KEY ABSOLUTO
            "chin_taper"        : (0.32, 0.47, 0.62),  # KEY
            "lower_n"           : (0.27, 0.34, 0.41),
            "brow_mouth_n"      : (0.34, 0.42, 0.50),
            "cheekbone_n"       : (0.50, 0.62, 0.74),
        },

        # ── DIAMANTE ─────────────────────────────────────────────────────────
        # cheekbone_n ALTO + forehead_to_cheek BAJO = pómulos dominantes.
        "Diamante": {
            "face_ratio"        : (0.64, 0.78, 0.92),
            "forehead_n"        : (0.30, 0.40, 0.50),
            "forehead_to_cheek" : (0.66, 0.78, 0.90),  # KEY
            "jaw_to_cheek"      : (0.58, 0.70, 0.82),  # KEY
            "forehead_to_jaw"   : (0.90, 1.06, 1.22),
            "chin_taper"        : (0.40, 0.55, 0.70),
            "lower_n"           : (0.27, 0.35, 0.43),
            "brow_mouth_n"      : (0.33, 0.42, 0.51),
            "cheekbone_n"       : (0.64, 0.76, 0.88),  # KEY
        },
    }

    # ── Perfiles por tipo de rostro ───────────────────────────────────────────
    #
    # v4 — Recalibración completa basada en análisis de 18 celebridades.
    #
    # DIAGNÓSTICO v3:
    #   face_ratio tenía peso 3.0 pero MediaPipe produce valores ~0.10-0.15
    #   más bajos de lo esperado (los landmarks de ancho son conservadores).
    #   Resultado: Redondo/Cuadrado nunca alcanzaban sus rangos mínimos.
    #   Todos los rostros caían en Ovalado/Largo que tienen rangos centrales.
    #
    # SOLUCIÓN v4:
    #   1. face_ratio bajado a peso 1.5 — deja de ser el discriminador primario.
    #   2. Los 3 ratios de forma (ftc, jtc, ftj) subieron a peso 3.5 cada uno
    #      — estos miden relaciones entre anchuras, que son más estables.
    #   3. Rangos de face_ratio desplazados ~0.10-0.14 hacia abajo en todos.
    #   4. Ovalado más estrecho — ya no es el "catch-all".
    #
    # RANGOS FACE_RATIO CALIBRADOS PARA MEDIAPIPE:
    #   Landmark 10 (forehead_top) está en el centro-alto de la frente,
    #   no en el borde del cabello. Los puntos de ancho (54, 284, 227, 345)
    #   quedan más adentro que el contorno visual. Resultado: face_ratio
    #   real de MediaPipe ≈ visual - 0.10 para la mayoría de rostros.
    PROFILES = {

        # ── OVALADO ──────────────────────────────────────────────────────────
        # Tipo equilibrado. Frente ≈ pómulos ≈ mandíbula (todo moderado).
        # face_ratio calibrado: 0.62-0.86 en MediaPipe para rostros ovalados.
        # cheekbone_n moderado. Rangos MÁS ESTRECHOS que v3 para no
        # capturar tipos que deberían ir a Redondo/Cuadrado.
        "Ovalado": {
            "face_ratio"        : (0.62, 0.74, 0.86),
            "forehead_n"        : (0.30, 0.40, 0.50),
            "forehead_to_cheek" : (0.90, 1.00, 1.10),  # frente ≈ pómulos
            "jaw_to_cheek"      : (0.84, 0.94, 1.04),  # mandíbula ≈ pómulos
            "forehead_to_jaw"   : (0.94, 1.06, 1.18),
            "chin_taper"        : (0.58, 0.70, 0.82),
            "lower_n"           : (0.26, 0.33, 0.40),
            "brow_mouth_n"      : (0.34, 0.42, 0.50),
            "cheekbone_n"       : (0.52, 0.62, 0.72),
        },

        # ── REDONDO ──────────────────────────────────────────────────────────
        # face_ratio MAYOR que Ovalado pero MediaPipe lo sub-mide.
        # Calibrado: 0.74-0.98 en MediaPipe (era 0.88-1.12, demasiado alto).
        # KEY: brow_mouth_n BAJO (cara compacta), lower_n BAJO (mentón corto).
        # cheekbone_n relativamente alto (pómulos amplios).
        "Redondo": {
            "face_ratio"        : (0.74, 0.86, 0.98),  # RECALIBRADO ↓0.14
            "forehead_n"        : (0.40, 0.50, 0.60),
            "forehead_to_cheek" : (0.86, 0.97, 1.08),
            "jaw_to_cheek"      : (0.88, 0.99, 1.10),
            "forehead_to_jaw"   : (0.82, 0.96, 1.10),
            "chin_taper"        : (0.70, 0.82, 0.94),
            "lower_n"           : (0.22, 0.27, 0.33),  # KEY: mentón muy corto
            "brow_mouth_n"      : (0.28, 0.35, 0.42),  # KEY: cara compacta
            "cheekbone_n"       : (0.58, 0.70, 0.82),
        },

        # ── CUADRADO ─────────────────────────────────────────────────────────
        # jaw_to_cheek ALTO: mandíbula ≈ pómulos (KEY absoluto).
        # chin_taper ALTO: mentón plano, no se angosta.
        # face_ratio recalibrado: 0.70-0.94 (era 0.82-1.06, demasiado alto).
        # forehead_to_jaw cercano a 1.0 (frente ≈ mandíbula, todo igual de ancho).
        "Cuadrado": {
            "face_ratio"        : (0.70, 0.82, 0.94),  # RECALIBRADO ↓0.12
            "forehead_n"        : (0.38, 0.48, 0.58),
            "forehead_to_cheek" : (0.88, 1.00, 1.12),
            "jaw_to_cheek"      : (0.94, 1.05, 1.16),  # KEY: mandíbula ≈ pómulos
            "forehead_to_jaw"   : (0.80, 0.94, 1.08),
            "chin_taper"        : (0.82, 0.92, 1.02),  # KEY: mentón plano
            "lower_n"           : (0.24, 0.31, 0.38),
            "brow_mouth_n"      : (0.30, 0.38, 0.46),
            "cheekbone_n"       : (0.56, 0.66, 0.76),
        },

        # ── LARGO ────────────────────────────────────────────────────────────
        # face_ratio MÁS BAJO de todos. brow_mouth_n ALTO (cara larga).
        # cheekbone_n BAJO (estrecha). Rangos recalibrados con desplazamiento menor
        # porque ya capturaba bien (Benedict/Adam/Keanu: 100% en v3).
        "Largo": {
            "face_ratio"        : (0.44, 0.57, 0.70),  # RECALIBRADO ↓0.06
            "forehead_n"        : (0.26, 0.35, 0.44),
            "forehead_to_cheek" : (0.84, 0.96, 1.08),
            "jaw_to_cheek"      : (0.76, 0.89, 1.02),
            "forehead_to_jaw"   : (0.88, 1.04, 1.20),
            "chin_taper"        : (0.60, 0.74, 0.88),
            "lower_n"           : (0.29, 0.37, 0.45),
            "brow_mouth_n"      : (0.42, 0.52, 0.62),  # KEY: cara larga
            "cheekbone_n"       : (0.44, 0.55, 0.66),  # KEY: estrecha
        },

        # ── CORAZÓN ──────────────────────────────────────────────────────────
        # forehead_to_jaw MUY ALTO: KEY absoluto e irremplazable.
        # forehead_to_cheek > 1.0: frente > pómulos.
        # chin_taper BAJO: mentón puntiagudo.
        # face_ratio recalibrado: 0.58-0.82.
        "Corazón": {
            "face_ratio"        : (0.58, 0.70, 0.82),  # RECALIBRADO ↓0.08
            "forehead_n"        : (0.40, 0.52, 0.64),
            "forehead_to_cheek" : (1.00, 1.13, 1.26),  # KEY: frente > pómulos
            "jaw_to_cheek"      : (0.50, 0.63, 0.76),  # KEY: mandíbula << pómulos
            "forehead_to_jaw"   : (1.16, 1.38, 1.60),  # KEY ABSOLUTO
            "chin_taper"        : (0.32, 0.47, 0.62),  # KEY: mentón puntiagudo
            "lower_n"           : (0.27, 0.34, 0.41),
            "brow_mouth_n"      : (0.33, 0.42, 0.51),
            "cheekbone_n"       : (0.50, 0.62, 0.74),
        },

        # ── DIAMANTE ─────────────────────────────────────────────────────────
        # cheekbone_n ALTO: pómulos dominantes (KEY).
        # forehead_to_cheek BAJO: frente < pómulos (KEY).
        # jaw_to_cheek BAJO: mandíbula < pómulos (KEY).
        # face_ratio recalibrado: 0.64-0.90.
        # Exclusión más permisiva en classify() para capturar Pattinson/Murphy.
        "Diamante": {
            "face_ratio"        : (0.64, 0.78, 0.92),  # RECALIBRADO ↓0.08
            "forehead_n"        : (0.30, 0.40, 0.50),
            "forehead_to_cheek" : (0.66, 0.78, 0.90),  # KEY: pómulos >> frente
            "jaw_to_cheek"      : (0.58, 0.70, 0.82),  # KEY: pómulos >> mandíbula
            "forehead_to_jaw"   : (0.90, 1.06, 1.22),
            "chin_taper"        : (0.40, 0.55, 0.70),
            "lower_n"           : (0.27, 0.35, 0.43),
            "brow_mouth_n"      : (0.33, 0.42, 0.51),
            "cheekbone_n"       : (0.64, 0.76, 0.88),  # KEY: alto
        },
    }

    @classmethod
    def _score_feature(cls, value: float, profile_range: Tuple[float, float, float]) -> float:
        """
        Score de una feature individual dado su valor y el rango del perfil
        (mínimo, ideal, máximo).

        Retorna entre -1.0 y +1.0:
            +1.0 → valor exactamente en el ideal
             0.0 → valor en el borde del rango aceptable
            -1.0 → valor muy fuera de rango
        """
        lo, ideal, hi = profile_range
        half_range = (hi - lo) / 2.0 + 1e-6

        if lo <= value <= hi:
            if value <= ideal:
                score = (value - lo) / (ideal - lo + 1e-6)
            else:
                score = (hi - value) / (hi - ideal + 1e-6)
            return max(0.0, min(1.0, score))
        else:
            penalty = (lo - value) / half_range if value < lo else (value - hi) / half_range
            return max(-1.0, -penalty)

    @classmethod
    def _pairwise_pass(
        cls,
        features: Dict[str, float],
        cand_a: str,
        cand_b: str,
        scores: Dict[str, float]
    ) -> str:
        """
        Segunda pasada: comparación directa entre los dos candidatos finalistas.

        Usa solo las features MÁS DISCRIMINANTES para cada par específico,
        ignorando features compartidas o ruidosas. Si ninguna regla aplica,
        devuelve el ganador del pase 1 (sin cambio).

        Se activa cuando margen < 8% O confianza < 52% en el pase 1.
        """
        pair = frozenset({cand_a, cand_b})
        f    = features

        # ── Ovalado vs Cuadrado ───────────────────────────────────────────────
        # Cuadrado: mandíbula marcada (jaw_to_cheek alto) y mentón plano.
        if pair == frozenset({"Ovalado", "Cuadrado"}):
            score_sq = (
                (f.get("jaw_to_cheek", 0) - 0.92) * 3.0 +
                (f.get("chin_taper",   0) - 0.74) * 2.5 +  # pivote 0.74 (antes 0.80)
                (f.get("face_ratio",   0) - 0.88) * 1.5
            )
            return "Cuadrado" if score_sq > 0 else "Ovalado"

        # ── Ovalado vs Corazón ────────────────────────────────────────────────
        # Corazón: frente >> mandíbula y mentón puntiagudo.
        if pair == frozenset({"Ovalado", "Corazón"}):
            score_h = (
                (f.get("forehead_to_jaw",   0) - 1.12) * 3.5 +
                (0.62 - f.get("chin_taper", 0))        * 2.0 +
                (f.get("forehead_to_cheek", 0) - 1.00) * 1.5
            )
            return "Corazón" if score_h > 0 else "Ovalado"

        # ── Ovalado vs Diamante ───────────────────────────────────────────────
        # Diamante: pómulos dominantes, frente y mandíbula más estrechas.
        if pair == frozenset({"Ovalado", "Diamante"}):
            score_d = (
                (f.get("cheekbone_n",       0) - 0.68) * 3.0 +
                (0.90 - f.get("forehead_to_cheek", 0)) * 3.0 +
                (0.86 - f.get("jaw_to_cheek",      0)) * 2.0
            )
            return "Diamante" if score_d > 0 else "Ovalado"

        # ── Redondo vs Cuadrado ───────────────────────────────────────────────
        # Cuadrado: mandíbula marcada. Redondo: mandíbula suave.
        if pair == frozenset({"Redondo", "Cuadrado"}):
            score_sq = (
                (f.get("jaw_to_cheek", 0) - 0.95) * 3.0 +
                (f.get("chin_taper",   0) - 0.86) * 2.5
            )
            return "Cuadrado" if score_sq > 0 else "Redondo"

        # ── Cuadrado vs Largo ─────────────────────────────────────────────────
        # Largo: face_ratio bajo. Cuadrado: face_ratio alto.
        if pair == frozenset({"Cuadrado", "Largo"}):
            return "Largo" if f.get("face_ratio", 0.8) < 0.80 else "Cuadrado"

        # ── Corazón vs Largo ──────────────────────────────────────────────────
        # Corazón: frente muy ancha respecto a mandíbula.
        if pair == frozenset({"Corazón", "Largo"}):
            score_h = (
                (f.get("forehead_to_jaw",  0) - 1.18) * 3.0 +
                (0.65 - f.get("chin_taper",0))        * 2.0
            )
            return "Corazón" if score_h > 0 else "Largo"

        # ── Corazón vs Diamante ───────────────────────────────────────────────
        # Corazón: frente > pómulos. Diamante: pómulos > frente.
        if pair == frozenset({"Corazón", "Diamante"}):
            return "Corazón" if f.get("forehead_to_cheek", 1.0) > 1.00 else "Diamante"

        # ── Diamante vs Redondo ───────────────────────────────────────────────
        # Diamante: pómulos prominentes, mentón más fino.
        if pair == frozenset({"Diamante", "Redondo"}):
            score_d = (
                (f.get("cheekbone_n",       0) - 0.72) * 3.0 +
                (0.90 - f.get("forehead_to_cheek", 0)) * 2.5 +
                (0.74 - f.get("chin_taper",        0)) * 1.5
            )
            return "Diamante" if score_d > 0 else "Redondo"

        # ── Largo vs Ovalado ──────────────────────────────────────────────────
        # Largo: face_ratio bajo y cara claramente alargada.
        if pair == frozenset({"Largo", "Ovalado"}):
            score_l = (
                (0.78 - f.get("face_ratio",    0)) * 3.0 +
                (f.get("brow_mouth_n",         0) - 0.43) * 2.0
            )
            return "Largo" if score_l > 0 else "Ovalado"

        # ── Redondo vs Ovalado ────────────────────────────────────────────────
        if pair == frozenset({"Redondo", "Ovalado"}):
            return "Redondo" if f.get("face_ratio", 0) > 0.90 else "Ovalado"

        # Sin regla para este par → mantener ganador del pase 1
        return cand_a if scores.get(cand_a, 0) >= scores.get(cand_b, 0) else cand_b

    @classmethod
    def classify(cls, d: Dict[str, float]) -> Dict:
        """
        Clasifica el tipo de rostro usando scoring vectorial ponderado.
        """
        # ── Paso 1: extraer vector de features ──────────────────────────────
        features = {f: d.get(f, 0.0) for f in cls.FEATURE_WEIGHTS}

        ftc  = features.get("forehead_to_cheek", 1.0)   # frente / pómulo
        jtc  = features.get("jaw_to_cheek",      1.0)   # mandíbula / pómulo
        ftj  = features.get("forehead_to_jaw",   1.0)   # frente / mandíbula
        fr   = features.get("face_ratio",        0.8)   # ancho / alto
        ct   = features.get("chin_taper",        0.7)   # adelgazamiento mentón
        ckn  = features.get("cheekbone_n",       0.6)   # ancho pómulo / alto cara

        # ── Hard-exclusion: eliminar tipos imposibles antes del scoring ──────
        #
        # Estas reglas previenen que el scoring secundario fuerce
        # clasificaciones que violan las características definitorias del tipo.
        #
        # DIAMANTE: pómulos dominantes.
        exclude = set()
        if ftc > 1.08 or jtc > 0.98:
            exclude.add("Diamante")

        # CORAZÓN: frente >> mandíbula y mentón puntiagudo.
        if ftj < 1.02 or ct > 0.80:
            exclude.add("Corazón")

        # CUADRADO: mandíbula ≈ pómulos, mentón plano.
        if jtc < 0.76 or ct < 0.65:
            exclude.add("Cuadrado")

        # REDONDO: cara compacta. brow_mouth_n ALTO → no es Redondo.
        # Esta es la exclusión más importante de v5.
        bm = features.get("brow_mouth_n", 0.5)
        ln = features.get("lower_n", 0.5)
        if fr < 0.66 or bm > 0.44 or ln > 0.34:
            exclude.add("Redondo")

        # LARGO: cara alargada.
        if fr > 0.90 or bm < 0.36:
            exclude.add("Largo")

        # Fallback: si quedan < 2 tipos, ignorar exclusiones
        remaining = [t for t in cls.PROFILES if t not in exclude]
        if len(remaining) < 2:
            exclude.clear()

        # ── Paso 2: score ponderado por tipo ────────────────────────────────
        scores = {}
        for face_type, profile in cls.PROFILES.items():
            if face_type in exclude:
                scores[face_type] = 0.0
                continue
            total_score  = 0.0
            total_weight = 0.0
            for feat_name, weight in cls.FEATURE_WEIGHTS.items():
                if feat_name in profile:
                    feat_score    = cls._score_feature(features[feat_name], profile[feat_name])
                    total_score  += feat_score * weight
                    total_weight += weight
            normalized = (total_score / total_weight + 1.0) / 2.0 if total_weight > 0 else 0.0
            scores[face_type] = round(max(0.0, min(1.0, normalized)), 4)

        sorted_types               = sorted(scores.items(), key=lambda x: -x[1])
        best_type,   best_score    = sorted_types[0]
        second_type, second_score  = sorted_types[1]

        # ── Paso 2b: TWO-PASS para confianza baja (< 52%) ───────────────────
        #
        # Si el ganador tiene poca ventaja sobre el segundo, ejecutamos un
        # segundo pase de comparación directa entre los dos candidatos usando
        # solo las features MÁS DISCRIMINANTES para ese par específico.
        # Esto evita que un tipo "promedio" (Ovalado) gane por defecto cuando
        # la cara es atípica pero claramente pertenece a otro tipo.
        #
        # Se activa cuando: margen < 0.08 O best_score < 0.52
        margin_pass1 = best_score - second_score
        if margin_pass1 < 0.08 or best_score < 0.52:
            best_type = cls._pairwise_pass(
                features, best_type, second_type, scores
            )
            best_score = scores[best_type]

        # ── Paso 3: reglas de desempate (margen < 5%) ───────────────────────
        #
        # Se aplican criterios duros sobre las features más discriminantes
        # cuando dos tipos quedan muy próximos en score.
        # Cubre 10 pares para reducir casos sin resolver.
        if (best_score - second_score) < 0.05:
            candidates = {best_type, second_type}

            # Cuadrado vs Redondo → chin_taper decide (cuadrado no se angosta)
            if candidates == {"Cuadrado", "Redondo"}:
                best_type = "Cuadrado" if features.get("chin_taper", 0) > 0.86 else "Redondo"

            # Ovalado vs Largo → face_ratio decide (largo es claramente más angosto)
            elif candidates == {"Ovalado", "Largo"}:
                best_type = "Largo" if features.get("face_ratio", 0) < 0.78 else "Ovalado"

            # Corazón vs Diamante → forehead_to_cheek decide
            elif candidates == {"Corazón", "Diamante"}:
                best_type = "Corazón" if features.get("forehead_to_cheek", 0) > 1.02 else "Diamante"

            # Ovalado vs Redondo → face_ratio decide
            elif candidates == {"Ovalado", "Redondo"}:
                best_type = "Redondo" if features.get("face_ratio", 0) > 0.88 else "Ovalado"

            # Corazón vs Ovalado → forehead_to_jaw es la señal inequívoca
            elif candidates == {"Corazón", "Ovalado"}:
                best_type = "Corazón" if features.get("forehead_to_jaw", 0) > 1.18 else "Ovalado"

            # Diamante vs Ovalado → forehead_to_cheek decide (diamante tiene pómulos dominantes)
            elif candidates == {"Diamante", "Ovalado"}:
                best_type = "Diamante" if features.get("forehead_to_cheek", 0) < 0.88 else "Ovalado"

            # Cuadrado vs Largo → face_ratio decide
            elif candidates == {"Cuadrado", "Largo"}:
                best_type = "Largo" if features.get("face_ratio", 0) < 0.80 else "Cuadrado"

            # Diamante vs Redondo → forehead_to_cheek decide
            elif candidates == {"Diamante", "Redondo"}:
                best_type = "Diamante" if features.get("forehead_to_cheek", 0) < 0.88 else "Redondo"

            # Corazón vs Largo → forehead_to_jaw decide
            elif candidates == {"Corazón", "Largo"}:
                best_type = "Corazón" if features.get("forehead_to_jaw", 0) > 1.20 else "Largo"

            # Cuadrado vs Ovalado → jaw_to_cheek decide (cuadrado tiene mandíbula prominente)
            elif candidates == {"Cuadrado", "Ovalado"}:
                best_type = "Cuadrado" if features.get("jaw_to_cheek", 0) > 0.96 else "Ovalado"

            best_score = scores[best_type]

        # ── Paso 4: confianza ────────────────────────────────────────────────
        margin    = best_score - second_score
        confianza = min(1.0, best_score * 0.6 + margin * 4.0)
        confianza = round(max(0.0, min(1.0, confianza)), 3)

        # ── Paso 5: descripciones y características ──────────────────────────
        desc_map = {
            "Ovalado"  : "Rostro equilibrado, ligeramente más largo que ancho. Los pómulos son la zona más ancha y la mandíbula se reduce gradualmente hacia el mentón.",
            "Redondo"  : "Rostro casi tan ancho como largo, con líneas suaves y curvas. Pómulos amplios y mentón redondeado.",
            "Cuadrado" : "Rostro ancho con mandíbula angular y muy marcada. Frente, pómulos y mandíbula tienen anchos similares.",
            "Largo"    : "Rostro notablemente más largo que ancho. Mandíbula definida, frente amplia y proporciones alargadas.",
            "Corazón"  : "Frente amplia que se angosta hacia un mentón puntiagudo. La mandíbula es significativamente más estrecha que la frente.",
            "Diamante" : "Pómulos prominentes y dominantes. Tanto la frente como la mandíbula son más estrechas que los pómulos.",
        }

        characteristics = {
            "Ovalado"  : ["Proporciones equilibradas", "Pómulos prominentes", "Mentón suave", "Frente proporcionada", "Mandíbula gradual"],
            "Redondo"  : ["Líneas suaves y curvas", "Pómulos amplios", "Mentón redondeado", "Ancho similar al alto", "Mandíbula suave"],
            "Cuadrado" : ["Mandíbula angular y marcada", "Frente amplia", "Proporciones anchas", "Mentón cuadrado", "Líneas definidas"],
            "Largo"    : ["Cara alargada", "Mandíbula definida", "Frente amplia", "Proporciones verticales", "Ángulos marcados"],
            "Corazón"  : ["Frente amplia", "Mentón puntiagudo", "Pómulos altos", "Mandíbula estrecha", "Forma de V invertida"],
            "Diamante" : ["Pómulos dominantes", "Frente estrecha", "Mandíbula estrecha", "Forma angular", "Mentón definido"],
        }

        return {
            "tipo"           : best_type,
            "confianza"      : confianza,
            "scores"         : scores,
            "features"       : features,
            "detalle"        : desc_map.get(best_type, ""),
            "characteristics": characteristics.get(best_type, []),
            "ranking"        : [(t, s) for t, s in sorted_types],
        }


def analyze_face_image(image_bytes: bytes) -> Optional[Dict]:
    """
    Analiza una imagen (bytes) y retorna la clasificación facial completa.
    Punto de entrada principal para el backend.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    lm = get_landmarks_from_image(frame_rgb)
    if lm is None:
        return None

    dists          = calculate_distances(lm, frame_rgb.shape[:2])
    classification = FaceShapeClassifier.classify(dists)

    return {
        "success"        : True,
        "face_shape"     : classification["tipo"],
        "confidence"     : int(classification["confianza"] * 100),
        "description"    : classification["detalle"],
        "characteristics": classification["characteristics"],
        "scores"         : classification["scores"],
        "metrics"        : {
            "face_ratio"       : dists["face_ratio"],
            "forehead_n"       : dists["forehead_n"],
            "forehead_to_cheek": dists["forehead_to_cheek"],
            "jaw_to_cheek"     : dists["jaw_to_cheek"],
            "forehead_to_jaw"  : dists["forehead_to_jaw"],
            "chin_taper"       : dists["chin_taper"],
            "cheekbone_n"      : dists["cheekbone_n"],
        },
    }

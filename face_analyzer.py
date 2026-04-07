"""
face_analyzer.py — Módulo de análisis facial adaptado de face_recognition_realtime_version_2.py
Procesa imágenes estáticas (subidas por el usuario) para clasificar la forma del rostro.
Usa MediaPipe Face Mesh + clasificador vectorial.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os
from typing import Optional, Tuple, Dict

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── Landmarks MediaPipe Face Mesh (478 puntos) ──────────────────────────────
LANDMARKS_IDX = {
    "forehead_top"       : 10,
    "chin"               : 152,
    "forehead_left"      : 46,
    "forehead_right"     : 276,
    "cheekbone_left"     : 234,
    "cheekbone_right"    : 454,
    "jaw_mid_left"       : 172,
    "jaw_mid_right"      : 397,
    "jaw_low_left"       : 150,
    "jaw_low_right"      : 379,
    "jaw_angle_left"     : 132,
    "jaw_angle_right"    : 361,
    "eye_left_outer"     : 33,
    "eye_left_inner"     : 133,
    "eye_right_inner"    : 362,
    "eye_right_outer"    : 263,
    "eye_left_top"       : 159,
    "eye_left_bot"       : 145,
    "eye_right_top"      : 386,
    "eye_right_bot"      : 374,
    "nose_tip"           : 1,
    "nose_bridge_top"    : 6,
    "nose_base_left"     : 129,
    "nose_base_right"    : 358,
    "mouth_left"         : 61,
    "mouth_right"        : 291,
    "mouth_top"          : 13,
    "mouth_bot"          : 14,
    "brow_left_inner"    : 107,
    "brow_left_peak"     : 70,
    "brow_left_outer"    : 46,
    "brow_right_inner"   : 336,
    "brow_right_peak"    : 300,
    "brow_right_outer"   : 276,
    "glabella"           : 9,
    "nose_base_center"   : 94,
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

    upper_third       = vertical_dist("forehead_top", "glabella")
    middle_third      = vertical_dist("glabella", "nose_base_center")
    lower_third       = vertical_dist("nose_base_center", "chin")

    brow_avg_y  = (lm["brow_left_peak"][1] + lm["brow_right_peak"][1]) / 2.0
    mouth_avg_y = (lm["mouth_top"][1] + lm["mouth_bot"][1]) / 2.0
    chin_y      = float(lm["chin"][1])

    brow_to_mouth = abs(mouth_avg_y - brow_avg_y)
    brow_to_chin  = abs(chin_y - brow_avg_y)

    brow_l = lm["eye_left_top"][1] - lm["brow_left_peak"][1]
    brow_r = lm["eye_right_top"][1] - lm["brow_right_peak"][1]
    brow_height = (brow_l + brow_r) / 2.0

    ear = (l_eye_open + r_eye_open) / (2.0 * eye_dist + 1e-6)

    fh = face_height + 1e-6
    cw = cheekbone_width + 1e-6

    max_width   = max(forehead_width, cheekbone_width, jaw_angle_width)
    face_ratio  = max_width / fh

    forehead_n  = forehead_width / fh
    cheekbone_n = cheekbone_width / fh
    jaw_mid_n   = jaw_mid_width / fh
    jaw_angle_n = jaw_angle_width / fh

    forehead_to_cheek   = forehead_width / cw
    jaw_to_cheek        = jaw_angle_width / cw
    forehead_to_jaw     = forehead_width / (jaw_angle_width + 1e-6)

    chin_taper  = jaw_low_width / (jaw_mid_width + 1e-6)

    upper_n     = upper_third / fh
    middle_n    = middle_third / fh
    lower_n     = lower_third / fh

    brow_mouth_n = brow_to_mouth / fh
    brow_chin_n  = brow_to_chin / fh

    mouth_ratio = mouth_width / cw
    nose_ratio  = nose_width / cw

    return {
        "face_height_px": round(face_height, 1),
        "face_width_px": round(max_width, 1),
        "forehead_width_px": round(forehead_width, 1),
        "cheekbone_width_px": round(cheekbone_width, 1),
        "jaw_mid_width_px": round(jaw_mid_width, 1),
        "jaw_angle_width_px": round(jaw_angle_width, 1),
        "jaw_low_width_px": round(jaw_low_width, 1),
        "eye_distance_px": round(eye_dist, 1),
        "nose_width_px": round(nose_width, 1),
        "mouth_width_px": round(mouth_width, 1),
        "face_ratio": round(face_ratio, 4),
        "forehead_n": round(forehead_n, 4),
        "cheekbone_n": round(cheekbone_n, 4),
        "jaw_mid_n": round(jaw_mid_n, 4),
        "jaw_angle_n": round(jaw_angle_n, 4),
        "forehead_to_cheek": round(forehead_to_cheek, 4),
        "jaw_to_cheek": round(jaw_to_cheek, 4),
        "forehead_to_jaw": round(forehead_to_jaw, 4),
        "chin_taper": round(chin_taper, 4),
        "upper_n": round(upper_n, 4),
        "middle_n": round(middle_n, 4),
        "lower_n": round(lower_n, 4),
        "brow_mouth_n": round(brow_mouth_n, 4),
        "brow_chin_n": round(brow_chin_n, 4),
        "mouth_ratio": round(mouth_ratio, 4),
        "nose_ratio": round(nose_ratio, 4),
        "eye_aspect_ratio": round(ear, 4),
    }


class FaceShapeClassifier:
    """Clasificador geométrico de tipo de rostro basado en scoring vectorial."""

    FEATURE_WEIGHTS = {
        "face_ratio"        : 3.0,
        "forehead_to_cheek" : 2.0,
        "jaw_to_cheek"      : 2.5,
        "forehead_to_jaw"   : 2.5,
        "chin_taper"        : 2.0,
        "lower_n"           : 1.5,
        "brow_mouth_n"      : 1.0,
        "cheekbone_n"       : 1.5,
    }

    PROFILES = {
        "Ovalado": {
            "face_ratio": (0.70, 0.80, 0.90),
            "forehead_to_cheek": (0.88, 0.96, 1.04),
            "jaw_to_cheek": (0.82, 0.92, 1.02),
            "forehead_to_jaw": (0.92, 1.04, 1.16),
            "chin_taper": (0.55, 0.70, 0.82),
            "lower_n": (0.28, 0.33, 0.38),
            "brow_mouth_n": (0.34, 0.42, 0.50),
            "cheekbone_n": (0.62, 0.72, 0.82),
        },
        "Redondo": {
            "face_ratio": (0.85, 0.94, 1.04),
            "forehead_to_cheek": (0.90, 0.97, 1.04),
            "jaw_to_cheek": (0.88, 0.95, 1.02),
            "forehead_to_jaw": (0.92, 1.02, 1.12),
            "chin_taper": (0.68, 0.80, 0.92),
            "lower_n": (0.28, 0.32, 0.37),
            "brow_mouth_n": (0.33, 0.40, 0.47),
            "cheekbone_n": (0.68, 0.78, 0.86),
        },
        "Cuadrado": {
            "face_ratio": (0.85, 0.95, 1.06),
            "forehead_to_cheek": (0.92, 0.99, 1.06),
            "jaw_to_cheek": (0.94, 1.02, 1.10),
            "forehead_to_jaw": (0.88, 0.98, 1.08),
            "chin_taper": (0.78, 0.88, 0.98),
            "lower_n": (0.30, 0.34, 0.38),
            "brow_mouth_n": (0.34, 0.40, 0.46),
            "cheekbone_n": (0.68, 0.76, 0.84),
        },
        "Rectángulo": {
            "face_ratio": (0.60, 0.70, 0.78),
            "forehead_to_cheek": (0.90, 0.98, 1.06),
            "jaw_to_cheek": (0.88, 0.96, 1.06),
            "forehead_to_jaw": (0.90, 1.02, 1.12),
            "chin_taper": (0.72, 0.84, 0.96),
            "lower_n": (0.30, 0.35, 0.40),
            "brow_mouth_n": (0.38, 0.45, 0.52),
            "cheekbone_n": (0.52, 0.60, 0.70),
        },
        "Triángulo": {
            "face_ratio": (0.75, 0.86, 0.98),
            "forehead_to_cheek": (0.70, 0.80, 0.88),
            "jaw_to_cheek": (1.02, 1.12, 1.22),
            "forehead_to_jaw": (0.62, 0.74, 0.85),
            "chin_taper": (0.72, 0.84, 0.96),
            "lower_n": (0.30, 0.35, 0.40),
            "brow_mouth_n": (0.36, 0.42, 0.48),
            "cheekbone_n": (0.62, 0.70, 0.80),
        },
        "Corazón": {
            "face_ratio": (0.72, 0.82, 0.92),
            "forehead_to_cheek": (0.98, 1.08, 1.18),
            "jaw_to_cheek": (0.64, 0.76, 0.88),
            "forehead_to_jaw": (1.18, 1.35, 1.52),
            "chin_taper": (0.40, 0.55, 0.68),
            "lower_n": (0.26, 0.31, 0.36),
            "brow_mouth_n": (0.36, 0.43, 0.50),
            "cheekbone_n": (0.60, 0.68, 0.76),
        },
        "Diamante": {
            "face_ratio": (0.72, 0.82, 0.92),
            "forehead_to_cheek": (0.74, 0.84, 0.92),
            "jaw_to_cheek": (0.68, 0.78, 0.88),
            "forehead_to_jaw": (0.94, 1.08, 1.20),
            "chin_taper": (0.42, 0.58, 0.72),
            "lower_n": (0.28, 0.33, 0.38),
            "brow_mouth_n": (0.36, 0.42, 0.48),
            "cheekbone_n": (0.60, 0.68, 0.78),
        },
    }

    @classmethod
    def _score_feature(cls, value, profile_range):
        lo, ideal, hi = profile_range
        half_range = (hi - lo) / 2.0 + 1e-6
        if lo <= value <= hi:
            if value <= ideal:
                score = (value - lo) / (ideal - lo + 1e-6)
            else:
                score = (hi - value) / (hi - ideal + 1e-6)
            return max(0.0, min(1.0, score))
        else:
            if value < lo:
                penalty = (lo - value) / half_range
            else:
                penalty = (value - hi) / half_range
            return max(-1.0, -penalty)

    @classmethod
    def classify(cls, d: Dict[str, float]) -> Dict:
        features = {f: d.get(f, 0.0) for f in cls.FEATURE_WEIGHTS}

        scores = {}
        for face_type, profile in cls.PROFILES.items():
            total_score = 0.0
            total_weight = 0.0
            for feat_name, weight in cls.FEATURE_WEIGHTS.items():
                if feat_name in profile:
                    value = features[feat_name]
                    feat_score = cls._score_feature(value, profile[feat_name])
                    total_score += feat_score * weight
                    total_weight += weight
            if total_weight > 0:
                normalized = (total_score / total_weight + 1.0) / 2.0
            else:
                normalized = 0.0
            scores[face_type] = round(max(0.0, min(1.0, normalized)), 4)

        sorted_types = sorted(scores.items(), key=lambda x: -x[1])
        best_type, best_score = sorted_types[0]
        second_type, second_score = sorted_types[1]

        # Desempate
        if (best_score - second_score) < 0.05:
            candidates = {best_type, second_type}
            if candidates == {"Cuadrado", "Redondo"}:
                best_type = "Cuadrado" if features.get("chin_taper", 0) > 0.82 else "Redondo"
            elif candidates == {"Ovalado", "Rectángulo"}:
                best_type = "Rectángulo" if features.get("face_ratio", 0) < 0.73 else "Ovalado"
            elif candidates == {"Corazón", "Diamante"}:
                best_type = "Corazón" if features.get("forehead_to_cheek", 0) > 0.98 else "Diamante"
            elif candidates == {"Triángulo", "Cuadrado"}:
                best_type = "Triángulo" if features.get("forehead_to_jaw", 0) < 0.82 else "Cuadrado"
            elif candidates == {"Ovalado", "Redondo"}:
                best_type = "Redondo" if features.get("face_ratio", 0) > 0.85 else "Ovalado"
            best_score = scores[best_type]

        margin = best_score - second_score
        confianza = min(1.0, best_score * 0.6 + margin * 4.0)
        confianza = round(max(0.0, min(1.0, confianza)), 3)

        desc_map = {
            "Ovalado":    "Rostro equilibrado, ligeramente más largo que ancho. Pómulos son la zona más ancha.",
            "Redondo":    "Rostro casi tan ancho como largo, líneas suaves y mentón redondeado.",
            "Cuadrado":   "Rostro con mandíbula angular y marcada. Frente, pómulos y mandíbula de ancho similar.",
            "Rectángulo": "Rostro más largo que ancho, angular. Mandíbula definida, frente amplia.",
            "Triángulo":  "Mandíbula más ancha que la frente. Línea mandibular dominante.",
            "Corazón":    "Frente amplia con mentón puntiagudo. Mandíbula más estrecha que la frente.",
            "Diamante":   "Pómulos prominentes. Frente y mandíbula estrechas.",
        }

        characteristics = {
            "Ovalado":    ["Proporciones equilibradas", "Pómulos prominentes", "Mentón suave", "Frente proporcionada", "Mandíbula gradual"],
            "Redondo":    ["Líneas suaves", "Pómulos amplios", "Mentón redondeado", "Ancho similar al alto", "Mandíbula suave"],
            "Cuadrado":   ["Mandíbula angular", "Frente amplia", "Proporciones anchas", "Mentón cuadrado", "Líneas definidas"],
            "Rectángulo": ["Cara alargada", "Mandíbula definida", "Frente amplia", "Proporciones verticales", "Ángulos marcados"],
            "Triángulo":  ["Mandíbula ancha", "Frente estrecha", "Base amplia", "Pómulos medios", "Mentón ancho"],
            "Corazón":    ["Frente amplia", "Mentón puntiagudo", "Pómulos altos", "Mandíbula estrecha", "Forma de V invertida"],
            "Diamante":   ["Pómulos dominantes", "Frente estrecha", "Mandíbula estrecha", "Forma angular", "Mentón definido"],
        }

        return {
            "tipo": best_type,
            "confianza": confianza,
            "scores": scores,
            "features": features,
            "detalle": desc_map.get(best_type, ""),
            "characteristics": characteristics.get(best_type, []),
            "ranking": [(t, s) for t, s in sorted_types],
        }


def analyze_face_image(image_bytes: bytes) -> Optional[Dict]:
    """
    Analiza una imagen (bytes) y retorna la clasificación facial completa.
    Punto de entrada principal para el backend.
    """
    # Decodificar imagen
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Extraer landmarks
    lm = get_landmarks_from_image(frame_rgb)
    if lm is None:
        return None

    # Calcular distancias
    dists = calculate_distances(lm, frame_rgb.shape[:2])

    # Clasificar
    classification = FaceShapeClassifier.classify(dists)

    return {
        "success": True,
        "face_shape": classification["tipo"],
        "confidence": int(classification["confianza"] * 100),
        "description": classification["detalle"],
        "characteristics": classification["characteristics"],
        "scores": classification["scores"],
        "metrics": {
            "face_ratio": dists["face_ratio"],
            "forehead_to_cheek": dists["forehead_to_cheek"],
            "jaw_to_cheek": dists["jaw_to_cheek"],
            "forehead_to_jaw": dists["forehead_to_jaw"],
            "chin_taper": dists["chin_taper"],
        }
    }

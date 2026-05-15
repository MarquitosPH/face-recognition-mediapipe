"""
face_analyzer.py — Clasificador facial con modelo ONNX YOLOv8s
Reemplaza el clasificador vectorial de MediaPipe por el modelo
preentrenado v1 (80.8% accuracy) exportado a ONNX.

TIPOS DE ROSTRO (6):
    Ovalado · Redondo · Cuadrado · Largo · Corazón · Diamante
"""

import cv2
import numpy as np
import os
import onnxruntime as ort
from typing import Optional, Dict

IMG_SIZE = 224

CLASS_NAMES = ["Cuadrado", "Diamante", "Largo", "Ovalado", "Redondo"]

DISPLAY_NAMES = {
    "Cuadrado" : "Cuadrado",
    "Diamante" : "Diamante",
    "Largo"    : "Largo",
    "Ovalado"  : "Ovalado",
    "Redondo"  : "Redondo",
}

DESC_MAP = {
    "Corazon"  : "Frente amplia que se angosta hacia un mentón puntiagudo. La mandíbula es significativamente más estrecha que la frente.",
    "Cuadrado" : "Rostro ancho con mandíbula angular y muy marcada. Frente, pómulos y mandíbula tienen anchos similares.",
    "Diamante" : "Pómulos prominentes y dominantes. Tanto la frente como la mandíbula son más estrechas que los pómulos.",
    "Largo"    : "Rostro notablemente más largo que ancho. Mandíbula definida, frente amplia y proporciones alargadas.",
    "Ovalado"  : "Rostro equilibrado, ligeramente más largo que ancho. Los pómulos son la zona más ancha y la mandíbula se reduce gradualmente hacia el mentón.",
    "Redondo"  : "Rostro casi tan ancho como largo, con líneas suaves y curvas. Pómulos amplios y mentón redondeado.",
}

CHARACTERISTICS_MAP = {
    "Corazon"  : ["Frente amplia", "Mentón puntiagudo", "Pómulos altos", "Mandíbula estrecha", "Forma de V invertida"],
    "Cuadrado" : ["Mandíbula angular y marcada", "Frente amplia", "Proporciones anchas", "Mentón cuadrado", "Líneas definidas"],
    "Diamante" : ["Pómulos dominantes", "Frente estrecha", "Mandíbula estrecha", "Forma angular", "Mentón definido"],
    "Largo"    : ["Cara alargada", "Mandíbula definida", "Frente amplia", "Proporciones verticales", "Ángulos marcados"],
    "Ovalado"  : ["Proporciones equilibradas", "Pómulos prominentes", "Mentón suave", "Frente proporcionada", "Mandíbula gradual"],
    "Redondo"  : ["Líneas suaves y curvas", "Pómulos amplios", "Mentón redondeado", "Ancho similar al alto", "Mandíbula suave"],
}

_ort_session  = None
_face_cascade = None


def _find_onnx_model() -> Optional[str]:
    candidates = [
        os.path.join(os.path.dirname(__file__), "model", "face_shape_classifier.onnx"),
        os.path.join(os.path.dirname(__file__), "face_shape_classifier.onnx"),
        "model/face_shape_classifier.onnx",
        "face_shape_classifier.onnx",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _get_session() -> Optional[ort.InferenceSession]:
    global _ort_session
    if _ort_session is None:
        model_path = _find_onnx_model()
        if model_path is None:
            print("[face_analyzer] ERROR: No se encontro face_shape_classifier.onnx")
            return None
        _ort_session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        print(f"[face_analyzer] Modelo ONNX cargado: {model_path}")
    return _ort_session


def _get_cascade() -> cv2.CascadeClassifier:
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def _detect_and_crop(img_bgr: np.ndarray) -> np.ndarray:
    cascade = _get_cascade()
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces   = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    if len(faces) > 0:
        x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        margin = int(max(w, h) * 0.20)
        x1 = max(0, x - margin);          y1 = max(0, y - margin)
        x2 = min(img_bgr.shape[1], x+w+margin); y2 = min(img_bgr.shape[0], y+h+margin)
        crop = img_bgr[y1:y2, x1:x2]
    else:
        h, w = img_bgr.shape[:2]; m = min(h, w)
        crop = img_bgr[(h-m)//2:(h+m)//2, (w-m)//2:(w+m)//2]

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return cv2.resize(crop_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)


def _preprocess(img_rgb: np.ndarray) -> np.ndarray:
    arr = img_rgb.astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return np.expand_dims(arr, axis=0)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def analyze_face_image(image_bytes: bytes) -> Optional[Dict]:
    """
    Analiza una imagen (bytes) y retorna la clasificacion facial.
    Compatible con el endpoint /api/upload-photo existente.
    """
    try:
        nparr   = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None

        face_rgb = _detect_and_crop(img_bgr)
        tensor   = _preprocess(face_rgb)

        session = _get_session()
        if session is None:
            return None

        input_name = session.get_inputs()[0].name
        output     = session.run(None, {input_name: tensor})[0][0]
        probs      = _softmax(output)

        top_idx   = int(np.argmax(probs))
        top_class = CLASS_NAMES[top_idx]
        top_conf  = float(probs[top_idx])

        scores  = {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))}
        ranking = [(CLASS_NAMES[i], round(float(probs[i]), 4)) for i in np.argsort(probs)[::-1]]

        return {
            "success"        : True,
            "face_shape"     : DISPLAY_NAMES[top_class],
            "confidence"     : int(top_conf * 100),
            "description"    : DESC_MAP.get(top_class, ""),
            "characteristics": CHARACTERISTICS_MAP.get(top_class, []),
            "scores"         : scores,
            "metrics"        : {},
            "ranking"        : ranking,
        }

    except Exception as e:
        print(f"[face_analyzer] Error: {e}")
        return None

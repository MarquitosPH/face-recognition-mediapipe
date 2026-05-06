"""
LensMatch — Sistema de Recomendación de Lentes basado en Reconocimiento Facial
FastAPI Backend + Jinja2 Templates + SQLite + OpenCV/MediaPipe
"""

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import os

from database import (
    get_all_glasses, get_glass_by_id, get_glasses_by_face_shape,
    get_all_face_shapes, get_face_shape_by_name
)
from face_analyzer import analyze_face_image
from face_analyzer import (
    get_landmarks_from_image, calculate_distances, FaceShapeClassifier
)
import cv2, numpy as np

app = FastAPI(title="LensMatch", version="3.0.0")

# --- Router de renderizado 3D server-side ---
try:
    from api_renderer import router as renderer_router
    app.include_router(renderer_router)
    print("[OK] Renderizador 3D server-side habilitado (/api/tryon/ws)")
except ImportError as e:
    print(f"[INFO] Renderizador server-side no disponible: {e}")
    print("[INFO] Instala: pip install pyrender trimesh PyOpenGL")

# --- Archivos estáticos y templates ---
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# =============================================
# RUTAS DE PÁGINAS (Templates HTML)
# =============================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/analisis", response_class=HTMLResponse)
async def face_capture(request: Request):
    return templates.TemplateResponse("face_capture.html", {"request": request})


@app.get("/procesando", response_class=HTMLResponse)
async def processing(request: Request):
    return templates.TemplateResponse("processing.html", {"request": request})


@app.get("/resultados", response_class=HTMLResponse)
async def results(request: Request):
    glasses = get_all_glasses()
    return templates.TemplateResponse("results.html", {
        "request": request,
        "glasses": glasses,
    })


@app.get("/detalle/{glass_id}", response_class=HTMLResponse)
async def lens_detail(request: Request, glass_id: str):
    glass = get_glass_by_id(glass_id)
    if not glass:
        glasses = get_all_glasses()
        glass = glasses[0] if glasses else {}
    return templates.TemplateResponse("lens_detail.html", {
        "request": request,
        "glass": glass,
    })


@app.get("/prueba-virtual/{glass_id}", response_class=HTMLResponse)
async def virtual_tryon(request: Request, glass_id: str):
    glass = get_glass_by_id(glass_id)
    if not glass:
        glasses = get_all_glasses()
        glass = glasses[0] if glasses else {}
    all_glasses = get_all_glasses()
    similar = [g for g in all_glasses if g["id"] != glass_id][:6]
    return templates.TemplateResponse("virtual_tryon.html", {
        "request": request,
        "glass": glass,
        "similar_glasses": similar,
        "all_glasses": all_glasses,
    })


@app.get("/catalogo", response_class=HTMLResponse)
async def catalog(request: Request):
    glasses = get_all_glasses()
    return templates.TemplateResponse("catalog.html", {
        "request": request,
        "glasses": glasses,
    })


@app.get("/guia-rostros", response_class=HTMLResponse)
async def face_guide(request: Request):
    return templates.TemplateResponse("face_guide.html", {"request": request})


@app.get("/analisis-completo", response_class=HTMLResponse)
async def face_analysis(request: Request):
    return templates.TemplateResponse("face_analysis.html", {"request": request})


# =============================================
# API ENDPOINTS
# =============================================

@app.post("/api/upload-photo")
async def upload_photo(file: UploadFile = File(...)):
    """Analiza una foto del rostro usando OpenCV + MediaPipe."""
    try:
        image_bytes = await file.read()
        result = analyze_face_image(image_bytes)

        if result is None:
            return JSONResponse({
                "success": False,
                "error": "No se detectó un rostro en la imagen. Intenta con mejor iluminación y mirando de frente."
            }, status_code=400)

        return JSONResponse(result)

    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": f"Error al procesar la imagen: {str(e)}"
        }, status_code=500)


@app.get("/api/glasses")
async def api_get_glasses():
    """Retorna todos los lentes desde SQLite."""
    return get_all_glasses()


@app.get("/api/glasses/{glass_id}")
async def api_get_glass(glass_id: str):
    """Retorna un lente por ID."""
    glass = get_glass_by_id(glass_id)
    if glass:
        return glass
    return JSONResponse({"error": "No encontrado"}, status_code=404)


@app.get("/api/recommendations/{face_shape}")
async def api_get_recommendations(face_shape: str):
    """Retorna lentes recomendados para una forma de rostro desde SQLite."""
    return get_glasses_by_face_shape(face_shape)


@app.get("/api/face-shapes")
async def api_get_face_shapes():
    """Retorna todas las formas de rostro con sus descripciones."""
    return get_all_face_shapes()


@app.post("/api/debug-metrics")
async def debug_metrics(file: UploadFile = File(...)):
    """
    Endpoint de diagnóstico — devuelve TODAS las métricas crudas que
    MediaPipe mide para una foto dada.

    Usar durante calibración del clasificador para ver exactamente qué
    valores produce el sistema antes de pasarlos al scoring.

    Acceso: POST /api/debug-metrics  (multipart/form-data, campo: file)
    También disponible en: GET /api/debug  (página HTML de diagnóstico)
    """
    try:
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse({"error": "No se pudo decodificar la imagen"}, status_code=400)

        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lm = get_landmarks_from_image(frame_rgb)
        if lm is None:
            return JSONResponse({"error": "No se detectó rostro"}, status_code=400)

        dists = calculate_distances(lm, frame_rgb.shape[:2])

        # Scores por tipo para ver cuál gana y por cuánto
        features = {f: dists.get(f, 0.0) for f in FaceShapeClassifier.FEATURE_WEIGHTS}
        scores   = {}
        for face_type, profile in FaceShapeClassifier.PROFILES.items():
            total, weight = 0.0, 0.0
            for feat, w in FaceShapeClassifier.FEATURE_WEIGHTS.items():
                if feat in profile:
                    s      = FaceShapeClassifier._score_feature(features[feat], profile[feat])
                    total += s * w
                    weight += w
            norm = (total / weight + 1.0) / 2.0 if weight > 0 else 0.0
            scores[face_type] = round(max(0.0, min(1.0, norm)), 4)

        return JSONResponse({
            "raw_metrics"   : dists,
            "scoring_input" : {k: round(v, 4) for k, v in features.items()},
            "scores_by_type": scores,
            "scores_ranked" : sorted(scores.items(), key=lambda x: -x[1]),
            "diagnostico": {
                "face_ratio_nota"  : "Si es < 0.68 → Largo. Si > 0.84 → Redondo/Cuadrado.",
                "brow_mouth_n_nota": "Si < 0.36 → cara compacta (Redondo). Si > 0.48 → cara larga.",
                "jaw_to_cheek_nota": "Si > 0.96 → Cuadrado. Si < 0.72 → Diamante/Corazón.",
                "forehead_to_jaw_nota": "Si > 1.18 → Corazón. Si < 0.96 → Cuadrado/Redondo.",
                "cheekbone_n_nota" : "Si > 0.68 → pómulos prominentes (Diamante/Redondo).",
            }
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/debug", response_class=HTMLResponse)
async def debug_page(request: Request):
    """Página HTML para usar el diagnóstico de métricas desde el navegador."""
    html = """
    <!DOCTYPE html><html lang="es"><head><meta charset="UTF-8">
    <title>Debug Métricas — LensMatch</title>
    <style>
      body{font-family:monospace;max-width:860px;margin:2rem auto;padding:1rem;background:#f8f8f8}
      h1{font-size:1.2rem;margin-bottom:1rem}
      input[type=file]{margin:.5rem 0}
      button{padding:.5rem 1.2rem;background:#333;color:#fff;border:none;cursor:pointer;border-radius:4px}
      pre{background:#1e1e1e;color:#d4d4d4;padding:1rem;border-radius:6px;overflow-x:auto;font-size:12px}
      .bar{display:flex;align-items:center;gap:8px;margin:3px 0;font-size:12px}
      .bar-track{width:300px;background:#ddd;border-radius:3px;height:14px}
      .bar-fill{height:100%;border-radius:3px;background:#4a9eff}
      label{font-size:14px;font-weight:bold}
    </style></head><body>
    <h1>🔬 Debug Métricas — LensMatch</h1>
    <p style="font-size:13px;color:#555">Sube una foto de frente para ver TODOS los valores crudos que MediaPipe mide.<br>
    Usa estos valores para calibrar los perfiles del clasificador.</p>
    <input type="file" id="f" accept="image/*">
    <button onclick="analyze()">Analizar</button>
    <div id="bars" style="margin:1rem 0"></div>
    <pre id="out">Esperando imagen...</pre>
    <script>
    async function analyze(){
      const file = document.getElementById('f').files[0];
      if(!file) return;
      const fd = new FormData(); fd.append('file', file);
      document.getElementById('out').textContent = 'Procesando...';
      const r = await fetch('/api/debug-metrics', {method:'POST', body:fd});
      const d = await r.json();
      document.getElementById('out').textContent = JSON.stringify(d, null, 2);
      // Barras de scores
      const scores = d.scores_ranked || [];
      const barsDiv = document.getElementById('bars');
      barsDiv.innerHTML = '<label>Scores por tipo:</label><br>';
      scores.forEach(([tipo, score]) => {
        barsDiv.innerHTML += `<div class="bar">
          <span style="min-width:90px">${tipo}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${score*100}%"></div></div>
          <span>${(score*100).toFixed(1)}%</span>
        </div>`;
      });
    }
    </script></body></html>
    """
    return HTMLResponse(html)


# =============================================
# RUN
# =============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
"""
OptiMatch — Sistema de Recomendación de Lentes basado en Reconocimiento Facial
FastAPI Backend + Jinja2 Templates + SQLite + OpenCV/MediaPipe
"""

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import Request  
import os

from database import (
    get_all_glasses, get_glass_by_id, get_glasses_by_face_shape,
    get_all_face_shapes, get_face_shape_by_name, save_glass_config, get_glass_config
)
from face_analyzer import analyze_face_image

app = FastAPI(title="OptiMatch", version="3.0.0")

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

    # Config de posicion/escala para el modelo inicial
    GLASS_CONFIGS = {
        "LentesPrueba1" : {"scale": "0.064 0.064 0.064", "position": "0 -0.34 -0.33", "rotation": "0 0 0"},
        "LentesAviador" : {"scale": "700 700 700",        "position": "0 0 0",          "rotation": "0 0 0"},
        "Browline"      : {"scale": "0.5 0.5 0.5", "position": "0 -0.03 -0.44", "rotation": "0 0 0"}, #classic oval
        "CatEye"        : {"scale": "3.66 3.66 3.66", "position": "0 -0.16 0.07", "rotation": "0 0 0"},
        "GruesoDecorado": {"scale": "1.85 1.85 1.85", "position": "-0.03 0.06 -0.29", "rotation": "0 180 0"}, # elegance frame
        "redondos"      : {"scale": "0.06 0.06 0.06", "position": "-0.02 -0.19 -0.25", "rotation": "0 96 0"}, #redondos
        "square"        : {"scale": "1.075 1.075 1.075", "position": "0.5 -1.1 -1.4", "rotation": "-9 -115 0"}, #titanium air
        "square2"       : {"scale": "0.72 0.72 0.72", "position": "-0.01 -0.01 -0.04", "rotation": "0 0 0"}, # moder square
    }
    default_config = {"scale": "0.064 0.064 0.064", "position": "0 -0.34 -0.33", "rotation": "0 0 0"}
    glass_config   = GLASS_CONFIGS.get(glass.get("model_3d", ""), default_config)
    saved_config = get_glass_config(glass_id)

    return templates.TemplateResponse("virtual_tryon.html", {
        "request"       : request,
        "glass"         : glass,
        "similar_glasses": similar,
        "all_glasses"   : all_glasses,
        "glass_config"  : glass_config,
        "saved_config": saved_config,  
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

@app.post("/api/glasses/{glass_id}/config")
async def save_config(glass_id: str, request: Request):
    data = await request.json()
    save_glass_config(
        glass_id,
        scale=data.get("scale", 0.5),
        pos_x=data.get("x", 0),
        pos_y=data.get("y", 0),
        pos_z=data.get("z", 0),
    )
    return {"ok": True}

@app.get("/api/glasses/{glass_id}/config")
async def get_config(glass_id: str):
    config = get_glass_config(glass_id)
    if config:
        return config
    return {}

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


# =============================================
# RUN
# =============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

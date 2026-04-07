"""
api_renderer.py — Endpoints de FastAPI para renderizado server-side de lentes 3D
Incluye WebSocket para tiempo real y REST para imágenes individuales.
"""

import base64
import json
import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from glasses_renderer import get_renderer

router = APIRouter(prefix="/api/tryon", tags=["Virtual Try-On"])


# ═══════════════════════════════════════════════════════════
# REST — Procesar una imagen individual (base64)
# ═══════════════════════════════════════════════════════════

@router.post("/render")
async def render_glasses_on_photo(data: dict):
    """
    Recibe una imagen en base64 y retorna la imagen con lentes superpuestos.

    Body JSON:
    {
        "image": "data:image/jpeg;base64,...",
        "model": "LentesPrueba1"
    }

    Response:
    {
        "success": true,
        "image": "data:image/jpeg;base64,...",
        "face_detected": true
    }
    """
    try:
        image_b64 = data.get("image", "")
        model_name = data.get("model", "LentesPrueba1")

        # Decodificar base64
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]

        img_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return JSONResponse({"success": False, "error": "Imagen inválida"})

        # Renderizar
        renderer = get_renderer(frame.shape[1], frame.shape[0])
        if model_name not in renderer.loaded_glasses:
            return JSONResponse({
                "success": False,
                "error": f"Modelo '{model_name}' no encontrado"
            })

        renderer.set_glasses(model_name)
        result, face_detected = renderer.render_frame(frame, model_name)

        # Codificar resultado a base64
        _, buffer = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 85])
        result_b64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "success": True,
            "image": f"data:image/jpeg;base64,{result_b64}",
            "face_detected": face_detected,
        }

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ═══════════════════════════════════════════════════════════
# WEBSOCKET — Streaming en tiempo real
# ═══════════════════════════════════════════════════════════

@router.websocket("/ws")
async def websocket_tryon(websocket: WebSocket):
    """
    WebSocket para prueba virtual en tiempo real.

    Protocolo:
    1. Cliente envía: {"type": "set_model", "model": "LentesPrueba1"}
    2. Cliente envía: {"type": "frame", "image": "base64..."}
    3. Servidor responde: {"type": "result", "image": "base64...", "face": true/false}

    El cliente captura frames del webcam, los envía como base64,
    el servidor renderiza los lentes y devuelve el resultado.
    """
    await websocket.accept()

    renderer = get_renderer()
    current_model = "LentesPrueba1"
    frame_count = 0

    # Establecer modelo inicial si hay lentes cargados
    if current_model in renderer.loaded_glasses:
        renderer.set_glasses(current_model)

    try:
        while True:
            # Recibir mensaje del cliente
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            msg_type = msg.get("type", "")

            if msg_type == "set_model":
                # Cambiar modelo de lentes
                new_model = msg.get("model", current_model)
                if new_model in renderer.loaded_glasses:
                    current_model = new_model
                    renderer.set_glasses(current_model)
                    await websocket.send_json({
                        "type": "model_set",
                        "model": current_model,
                        "success": True,
                    })
                else:
                    await websocket.send_json({
                        "type": "model_set",
                        "model": new_model,
                        "success": False,
                        "error": f"Modelo '{new_model}' no encontrado",
                    })

            elif msg_type == "frame":
                # Procesar frame
                image_b64 = msg.get("image", "")
                if "," in image_b64:
                    image_b64 = image_b64.split(",")[1]

                try:
                    img_bytes = base64.b64decode(image_b64)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if frame is not None:
                        result, face_detected = renderer.render_frame(
                            frame, current_model
                        )

                        # Codificar resultado
                        _, buffer = cv2.imencode(
                            '.jpg', result,
                            [cv2.IMWRITE_JPEG_QUALITY, 75]  # calidad más baja para velocidad
                        )
                        result_b64 = base64.b64encode(buffer).decode('utf-8')

                        await websocket.send_json({
                            "type": "result",
                            "image": result_b64,
                            "face": face_detected,
                            "frame": frame_count,
                        })

                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e),
                    })

                frame_count += 1

            elif msg_type == "config":
                # Actualizar configuración del modelo en tiempo real
                model = msg.get("model", current_model)
                config = msg.get("config", {})
                renderer.add_config(model, config)
                await websocket.send_json({
                    "type": "config_updated",
                    "model": model,
                })

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        print(f"[WS] Cliente desconectado (frames procesados: {frame_count})")
    except Exception as e:
        print(f"[WS] Error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass

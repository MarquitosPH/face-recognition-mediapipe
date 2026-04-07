"""
glasses_renderer.py — Renderizador 3D de lentes con Head Occluder
Usa pyrender + trimesh + MediaPipe para composición en tiempo real.

Flujo:
  1. Recibe frame de cámara + 468 landmarks de MediaPipe
  2. Calcula Matriz de Pose 4x4 (Procrustes alignment)
  3. Renderiza lentes 3D + cabeza fantasma (oclusor invisible)
  4. Compone el resultado sobre el frame original

Dependencias:
  pip install pyrender trimesh numpy opencv-python mediapipe PyOpenGL

Nota: pyrender necesita un backend OpenGL. En servidores sin pantalla:
  pip install PyOpenGL-osmesa   (para OSMesa)
  O configurar EGL si tienes GPU
"""

import os
import numpy as np
import trimesh
import cv2
import mediapipe as mp
from typing import Optional, Dict, Tuple, List

# Backend para renderizado offscreen (sin pantalla)
# Intentar EGL primero (GPU), luego OSMesa (CPU)
os.environ['PYOPENGL_PLATFORM'] = 'egl'
try:
    import pyrender
except Exception:
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
    import pyrender


# ═══════════════════════════════════════════════════════════
# CANONICAL FACE MODEL — Puntos de referencia estáticos
# Subset de los 468 landmarks del modelo canónico de MediaPipe
# Usados para calcular la transformación Procrustes
# ═══════════════════════════════════════════════════════════

# Índices de landmarks estables para calcular la pose
# (puntos que no se mueven mucho con expresiones faciales)
STABLE_LANDMARK_INDICES = [
    1,    # punta de nariz
    33,   # ojo izquierdo exterior
    263,  # ojo derecho exterior
    61,   # boca izquierda
    291,  # boca derecha
    199,  # mentón inferior
    10,   # frente superior
    152,  # mentón
    234,  # pómulo izquierdo
    454,  # pómulo derecho
    130,  # borde izquierdo cara
    359,  # borde derecho cara
    6,    # puente nasal
    168,  # entre cejas
]

# Coordenadas 3D del modelo canónico de MediaPipe (en cm, subset)
# Estos valores corresponden a los STABLE_LANDMARK_INDICES
# Fuente: MediaPipe canonical_face_model
CANONICAL_POINTS = np.array([
    [0.0, -0.0636, 0.0625],       # 1  nariz punta
    [-0.0345, 0.0283, 0.0365],    # 33 ojo izq exterior
    [0.0345, 0.0283, 0.0365],     # 263 ojo der exterior
    [-0.0236, -0.0404, 0.0286],   # 61 boca izq
    [0.0236, -0.0404, 0.0286],    # 291 boca der
    [0.0, -0.0985, 0.0175],       # 199 mentón inf
    [0.0, 0.0704, 0.0315],        # 10 frente
    [0.0, -0.0877, 0.0207],       # 152 mentón
    [-0.0678, 0.0072, 0.0116],    # 234 pómulo izq
    [0.0678, 0.0072, 0.0116],     # 454 pómulo der
    [-0.0586, -0.0068, 0.0171],   # 130 borde izq
    [0.0586, -0.0068, 0.0171],    # 359 borde der
    [0.0, 0.0236, 0.0540],        # 6  puente nasal
    [0.0, 0.0465, 0.0442],        # 168 entre cejas
], dtype=np.float64)


# ═══════════════════════════════════════════════════════════
# FUNCIONES DE POSE
# ═══════════════════════════════════════════════════════════

def landmarks_to_3d_points(
    landmarks: list,
    indices: List[int],
    frame_w: int,
    frame_h: int
) -> np.ndarray:
    """
    Extrae puntos 3D de los landmarks de MediaPipe.
    Convierte de coordenadas normalizadas (0-1) a coordenadas de imagen.
    """
    points = np.zeros((len(indices), 3), dtype=np.float64)
    for i, idx in enumerate(indices):
        lm = landmarks[idx]
        points[i, 0] = lm.x * frame_w
        points[i, 1] = lm.y * frame_h
        points[i, 2] = lm.z * frame_w  # Z está en misma escala que X
    return points


def compute_pose_matrix(
    landmarks: list,
    frame_w: int,
    frame_h: int
) -> np.ndarray:
    """
    Calcula la Matriz de Pose 4x4 usando Procrustes alignment.
    Alinea el modelo canónico a los landmarks detectados.

    Returns: Matriz 4x4 de transformación [R|t; 0 0 0 1]
    """
    # Puntos detectados en el frame
    detected = landmarks_to_3d_points(
        landmarks, STABLE_LANDMARK_INDICES, frame_w, frame_h
    )

    # Escalar puntos canónicos al tamaño de la cara detectada
    # Usar distancia entre landmarks 130 y 359 (ancho de cara)
    lm_130 = landmarks[130]
    lm_359 = landmarks[359]
    face_width_px = np.sqrt(
        (lm_359.x * frame_w - lm_130.x * frame_w) ** 2 +
        (lm_359.y * frame_h - lm_130.y * frame_h) ** 2
    )

    # Ancho canónico (distancia entre puntos 130 y 359 en el modelo)
    canon_width = np.linalg.norm(CANONICAL_POINTS[10] - CANONICAL_POINTS[11])
    scale = face_width_px / (canon_width + 1e-6)

    canonical_scaled = CANONICAL_POINTS * scale

    # Procrustes: encontrar R, t que minimiza ||detected - (R @ canonical + t)||
    # Centrar ambos conjuntos de puntos
    centroid_det = np.mean(detected, axis=0)
    centroid_can = np.mean(canonical_scaled, axis=0)

    det_centered = detected - centroid_det
    can_centered = canonical_scaled - centroid_can

    # SVD para encontrar la rotación óptima
    H = can_centered.T @ det_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Corregir reflexión si determinante es negativo
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Traslación
    t = centroid_det - R @ centroid_can

    # Construir matriz 4x4
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R
    pose[:3, 3] = t

    return pose, scale


# ═══════════════════════════════════════════════════════════
# GLASSES RENDERER CLASS
# ═══════════════════════════════════════════════════════════

class GlassesRenderer:
    """
    Renderizador de lentes 3D con Head Occluder.

    Usa pyrender para renderizado offscreen y trimesh para carga de modelos.
    La "cabeza fantasma" (canonical face model) se renderiza como oclusor
    invisible (color_mask=False) que bloquea el Depth Buffer.

    Usage:
        renderer = GlassesRenderer(width=1280, height=720)
        renderer.load_glasses("static/models/LentesPrueba1.glb")

        # En cada frame:
        result = renderer.render_frame(frame_bgr, landmarks)
    """

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height

        # Renderer offscreen de pyrender
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=width,
            viewport_height=height,
            point_size=1.0
        )

        # Escena base
        self.scene = None

        # Nodos
        self.glasses_node = None
        self.occluder_node = None
        self.camera_node = None
        self.light_node = None

        # Modelos cargados (cache)
        self.loaded_glasses: Dict[str, trimesh.Trimesh] = {}

        # Malla del oclusor (cabeza fantasma)
        self.occluder_mesh = None

        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Configuraciones por modelo
        self.glass_configs: Dict[str, dict] = {
            "LentesPrueba1": {
                "scale": 1.0,
                "offset_y": 0.0,
                "offset_z": 0.0,
                "rot_x": 0, "rot_y": 0, "rot_z": 0,
            },
        }
        self.default_config = {
            "scale": 1.0, "offset_y": 0.0, "offset_z": 0.0,
            "rot_x": 0, "rot_y": 0, "rot_z": 0,
        }

        self._init_scene()
        self._load_occluder()

    def _init_scene(self):
        """Inicializa la escena de pyrender con cámara y luces."""
        self.scene = pyrender.Scene(
            bg_color=[0, 0, 0, 0],  # fondo transparente
            ambient_light=[0.4, 0.4, 0.4]
        )

        # Cámara ortográfica que coincide con el viewport del video
        camera = pyrender.IntrinsicsCamera(
            fx=self.width, fy=self.width,
            cx=self.width / 2, cy=self.height / 2,
            znear=0.01, zfar=1000.0
        )
        cam_pose = np.eye(4)
        self.camera_node = self.scene.add(camera, pose=cam_pose)

        # Luces
        dir_light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0)
        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, 0, 1]
        self.light_node = self.scene.add(dir_light, pose=light_pose)

        # Luz ambiental adicional
        point_light = pyrender.PointLight(color=[1, 1, 1], intensity=2.0)
        pl_pose = np.eye(4)
        pl_pose[:3, 3] = [self.width / 2, 0, 500]
        self.scene.add(point_light, pose=pl_pose)

    def _load_occluder(self, obj_path: Optional[str] = None):
        """
        Carga el oclusor de cabeza (cabeza fantasma).
        Si no hay archivo .obj, genera una malla básica desde los puntos canónicos.
        """
        if obj_path and os.path.exists(obj_path):
            # Cargar canonical_face_model.obj de MediaPipe
            occluder_trimesh = trimesh.load(obj_path, force='mesh')
        else:
            # Generar malla simple desde los puntos canónicos
            # Usamos una esfera deformada como aproximación
            occluder_trimesh = self._generate_head_mesh()

        # Material invisible: NO dibuja color, SÍ escribe profundidad
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.0, 0.0, 0.0, 0.0],
            alphaMode='BLEND',
        )

        self.occluder_mesh = pyrender.Mesh.from_trimesh(
            occluder_trimesh,
            material=material,
        )

    def _generate_head_mesh(self) -> trimesh.Trimesh:
        """
        Genera una malla de cabeza simplificada a partir de
        los vértices del modelo canónico de MediaPipe.
        Esto se usa si no tienes el archivo canonical_face_model.obj
        """
        # Crear una esfera deformada como aproximación de cabeza
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.08)
        vertices = sphere.vertices.copy()

        # Deformar: aplanar en Z, alargar en Y
        vertices[:, 2] *= 0.7   # aplanar frente-atrás
        vertices[:, 1] *= 1.15  # alargar arriba-abajo

        sphere.vertices = vertices
        return sphere

    def load_glasses(self, glb_path: str, model_name: str = "") -> bool:
        """
        Carga un modelo de lentes .glb/.gltf.
        Aplica correcciones de orientación y escala para modelos de Blender.
        """
        if not os.path.exists(glb_path):
            print(f"[ERROR] No se encontró: {glb_path}")
            return False

        try:
            # trimesh carga .glb como Scene, extraer la malla
            loaded = trimesh.load(glb_path, force='mesh')

            if isinstance(loaded, trimesh.Scene):
                # Combinar todas las mallas de la escena (marcos, cristales, etc.)
                meshes = []
                for geom in loaded.geometry.values():
                    if isinstance(geom, trimesh.Trimesh):
                        meshes.append(geom)
                if meshes:
                    loaded = trimesh.util.concatenate(meshes)
                else:
                    print(f"[ERROR] No se encontraron mallas en: {glb_path}")
                    return False

            # ═══════════════════════════════════════════════════════════
            # CORRECCIÓN DE ORIENTACIÓN (BLENDER -> MEDIAPIPE)
            # ═══════════════════════════════════════════════════════════
            # 1. Rotación: 90 grados en X para que miren al frente
            rot_matrix = trimesh.transformations.rotation_matrix(
                np.radians(90), [1, 0, 0]
            )
            loaded.apply_transform(rot_matrix)

            # 2. Centrado: Asegurar que el "puente" sea el origen (0,0,0)
            # Esto evita que los lentes salgan volando fuera de la cara
            loaded.vertices -= loaded.center_mass 

            # 3. Guardar en el cache
            name = model_name or os.path.splitext(os.path.basename(glb_path))[0]
            self.loaded_glasses[name] = loaded
            
            print(f"[OK] Modelo '{name}' cargado, rotado y centrado.")
            return True

        except Exception as e:
            print(f"[ERROR] No se pudo cargar {glb_path}: {e}")
            return False

    def set_glasses(self, model_name: str) -> bool:
        """Establece qué modelo de lentes renderizar."""
        if model_name not in self.loaded_glasses:
            print(f"[WARN] Modelo '{model_name}' no cargado")
            return False

        # Remover lentes anteriores
        if self.glasses_node is not None:
            self.scene.remove_node(self.glasses_node)
            self.glasses_node = None

        # Remover oclusor anterior
        if self.occluder_node is not None:
            self.scene.remove_node(self.occluder_node)
            self.occluder_node = None

        # Agregar lentes con pose identidad (se actualiza en render)
        glasses_trimesh = self.loaded_glasses[model_name]
        glasses_mesh = pyrender.Mesh.from_trimesh(glasses_trimesh)
        self.glasses_node = self.scene.add(glasses_mesh, pose=np.eye(4))

        # Agregar oclusor (cabeza fantasma)
        if self.occluder_mesh:
            self.occluder_node = pyrender.Node(
                mesh=self.occluder_mesh,
                matrix=np.eye(4),
            )
            self.scene.add_node(self.occluder_node)

        return True

    def get_config(self, model_name: str) -> dict:
        return self.glass_configs.get(model_name, self.default_config)

    def add_config(self, model_name: str, config: dict):
        """Agrega o actualiza configuración de un modelo."""
        self.glass_configs[model_name] = {**self.default_config, **config}

    def detect_landmarks(self, frame_rgb: np.ndarray):
        """Detecta landmarks faciales con MediaPipe."""
        results = self.mp_face_mesh.process(frame_rgb)
        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
            return results.multi_face_landmarks[0].landmark
        return None

    def render_frame(
        self,
        frame_bgr: np.ndarray,
        model_name: str = "LentesPrueba1",
        landmarks=None,
    ) -> Tuple[np.ndarray, bool]:
        """
        Pipeline completo: detecta rostro, renderiza lentes + oclusor, compone.

        Args:
            frame_bgr: Frame de la cámara en BGR (OpenCV)
            model_name: Nombre del modelo de lentes a usar
            landmarks: Landmarks pre-detectados (opcional, si no se pasan se detectan)

        Returns:
            (frame_resultado_bgr, rostro_detectado)
        """
        h, w = frame_bgr.shape[:2]

        # Redimensionar renderer si cambió el tamaño
        if w != self.width or h != self.height:
            self.width = w
            self.height = h
            self.renderer.viewport_width = w
            self.renderer.viewport_height = h

        # Detectar landmarks si no se proporcionaron
        if landmarks is None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            landmarks = self.detect_landmarks(frame_rgb)

        if landmarks is None:
            return frame_bgr, False

        # Asegurar que los lentes estén en la escena
        if self.glasses_node is None or model_name not in self.loaded_glasses:
            if model_name in self.loaded_glasses:
                self.set_glasses(model_name)
            else:
                return frame_bgr, False

        # Calcular pose
        pose, scale = compute_pose_matrix(landmarks, w, h)
        config = self.get_config(model_name)

        # Aplicar offsets de configuración
        offset_matrix = np.eye(4)
        offset_matrix[1, 3] = config["offset_y"] * scale
        offset_matrix[2, 3] = config["offset_z"] * scale

        # Aplicar rotaciones extra de configuración
        rx = np.radians(config["rot_x"])
        ry = np.radians(config["rot_y"])
        rz = np.radians(config["rot_z"])

        Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
        Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
        Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])

        rot_extra = np.eye(4)
        rot_extra[:3, :3] = Rz @ Ry @ Rx

        # Escala del modelo
        scale_factor = config["scale"] * scale * 0.01  # ajuste base
        scale_matrix = np.eye(4)
        scale_matrix[0, 0] = scale_factor
        scale_matrix[1, 1] = scale_factor
        scale_matrix[2, 2] = scale_factor

        # Pose final = pose_cara × offset × rotación_extra × escala
        glasses_pose = pose @ offset_matrix @ rot_extra @ scale_matrix
        occluder_pose = pose.copy()

        # Actualizar poses en la escena
        self.scene.set_pose(self.glasses_node, glasses_pose)
        if self.occluder_node:
            self.scene.set_pose(self.occluder_node, occluder_pose)

        # Renderizar (color + depth)
        try:
            color, depth = self.renderer.render(
                self.scene,
                flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES
            )
        except Exception as e:
            print(f"[ERROR] Render failed: {e}")
            return frame_bgr, True

        # Composición: superponer lentes sobre el frame original
        result = self._composite(frame_bgr, color)

        return result, True

    def _composite(self, background: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
        """
        Compone el overlay RGBA sobre el background BGR.
        Usa alpha blending para transparencia.
        """
        if overlay_rgba.shape[2] == 4:
            alpha = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0
            overlay_rgb = overlay_rgba[:, :, :3].astype(np.float32)
            bg = background.astype(np.float32)

            # Alpha blending: result = overlay * alpha + background * (1 - alpha)
            # Convertir overlay de RGB a BGR para OpenCV
            overlay_bgr = overlay_rgb[:, :, ::-1]

            result = overlay_bgr * alpha + bg * (1.0 - alpha)
            return result.astype(np.uint8)
        else:
            return background

    def cleanup(self):
        """Libera recursos."""
        if self.renderer:
            self.renderer.delete()
        if self.mp_face_mesh:
            self.mp_face_mesh.close()


# ═══════════════════════════════════════════════════════════
# SINGLETON GLOBAL (para usar en FastAPI sin recrear)
# ═══════════════════════════════════════════════════════════

_renderer_instance: Optional[GlassesRenderer] = None

def get_renderer(width: int = 1280, height: int = 720) -> GlassesRenderer:
    """Obtiene o crea el singleton del renderer."""
    global _renderer_instance
    if _renderer_instance is None:
        _renderer_instance = GlassesRenderer(width, height)
        # Pre-cargar modelos del directorio models/
        models_dir = os.path.join(os.path.dirname(__file__), "static", "models")
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                if f.endswith('.glb') or f.endswith('.gltf'):
                    name = os.path.splitext(f)[0]
                    _renderer_instance.load_glasses(
                        os.path.join(models_dir, f), name
                    )
    return _renderer_instance

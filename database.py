"""
database.py — Módulo de base de datos SQLite para LensMatch
Gestiona la tabla de lentes y las recomendaciones por forma de rostro.
"""

import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "lensmatch.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Crea las tablas si no existen y carga datos iniciales."""
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS glasses (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            brand       TEXT NOT NULL,
            style       TEXT NOT NULL,
            material    TEXT NOT NULL,
            category    TEXT NOT NULL,
            gender      TEXT NOT NULL,
            compatibility INTEGER DEFAULT 0,
            compatible_faces TEXT NOT NULL,
            description TEXT,
            image       TEXT,
            image_hover TEXT,
            images      TEXT,
            tags        TEXT,
            model_3d    TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS face_shapes (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT UNIQUE NOT NULL,
            description TEXT,
            tips        TEXT
        )
    """)

    c.execute("SELECT COUNT(*) FROM glasses")
    if c.fetchone()[0] == 0:
        _seed_glasses(c)

    c.execute("SELECT COUNT(*) FROM face_shapes")
    if c.fetchone()[0] == 0:
        _seed_face_shapes(c)

    conn.commit()
    conn.close()


def _seed_glasses(cursor):
    # ====================================================================
    # IMÁGENES - FILTRADAS: solo fotos de LENTES PRODUCTO sobre fondo
    # claro/neutro (evitamos modelos, exteriores, playa, etc.)
    # ====================================================================
    # Estructura dual:
    #   image       → foto principal (reposo)
    #   image_hover → foto alternativa (al pasar el cursor)
    # ====================================================================

    # Locales
    IMG_CLASICO = "/static/images/clasicos.jpg"
    IMG_AVIADOR_1 = "/static/images/LentesAviador01.jpg"
    IMG_AVIADOR_2 = "/static/images/LentesAviador02.jpg"
    IMG_AVIADOR_3 = "/static/images/LentesAviador03.jpg"
    IMG_AVIADOR_4 = "/static/images/LentesAviador04.jpg"

    # Externas — todas fotos de lentes en fondo blanco/neutro (tipo catálogo e-commerce)
    # Seleccionadas para look uniforme (no modelos, no exteriores)
    UX_GLASSES_01 = "https://images.unsplash.com/photo-1511499767150-a48a237f0083?w=800&q=80&auto=format"
    UX_GLASSES_02 = "https://images.unsplash.com/photo-1577803645773-f96470509666?w=800&q=80&auto=format"
    UX_GLASSES_03 = "https://images.unsplash.com/photo-1574258495973-f010dfbb5371?w=800&q=80&auto=format"
    UX_GLASSES_04 = "https://images.unsplash.com/photo-1572635196237-14b3f281503f?w=800&q=80&auto=format"
    UX_GLASSES_05 = "https://images.unsplash.com/photo-1556306535-0f09a537f0a3?w=800&q=80&auto=format"
    UX_GLASSES_06 = "https://images.unsplash.com/photo-1583394838336-acd977736f90?w=800&q=80&auto=format"
    UX_GLASSES_07 = "https://images.unsplash.com/photo-1604785555196-1023b6e22eaa?w=800&q=80&auto=format"
    UX_GLASSES_08 = "https://images.unsplash.com/photo-1591076482161-42ce6da69f67?w=800&q=80&auto=format"
    UX_GLASSES_09 = "https://images.unsplash.com/photo-1508296695146-257a814070b4?w=800&q=80&auto=format"
    UX_GLASSES_10 = "https://images.unsplash.com/photo-1577744486770-020ab432da65?w=800&q=80&auto=format"
    UX_GLASSES_11 = "https://images.unsplash.com/photo-1600208681548-2a5a8cba86a4?w=800&q=80&auto=format"
    UX_GLASSES_12 = "https://images.unsplash.com/photo-1619449284321-09a3d58bd8d5?w=800&q=80&auto=format"

    glasses = [
        {
            "id": "1", "name": "Classic Oval A123", "brand": "VisionPlus",
            "style": "Clásico", "material": "Acetato", "category": "Diario", "gender": "Unisex",
            "compatibility": 94, "compatible_faces": ["Ovalado", "Triangular"],
            "description": "Armazón clásico de acetato con líneas suaves y elegantes.",
            "image": UX_GLASSES_01, "image_hover": UX_GLASSES_02,
            "images": [UX_GLASSES_01, UX_GLASSES_02, UX_GLASSES_03],
            "tags": ["Clásico", "Acetato"], "model_3d": "LentesPrueba1"
        },
        {
            "id": "2", "name": "Modern Square B456", "brand": "OptiFrame",
            "style": "Moderno", "material": "Metal", "category": "Profesional", "gender": "Hombre",
            "compatibility": 88, "compatible_faces": ["Ovalado", "Redondo"],
            "description": "Diseño moderno con líneas cuadradas definidas en metal ligero.",
            "image": UX_GLASSES_02, "image_hover": UX_GLASSES_05,
            "images": [UX_GLASSES_02, UX_GLASSES_05, UX_GLASSES_01],
            "tags": ["Moderno", "Metal"], "model_3d": "LentesPrueba1"
        },
        {
            "id": "3", "name": "Cat Eye Luxe C789", "brand": "FemmeVision",
            "style": "Elegante", "material": "Acetato", "category": "Moda", "gender": "Mujer",
            "compatibility": 91, "compatible_faces": ["Ovalado", "Cuadrado"],
            "description": "Sofisticado armazón cat-eye en acetato premium.",
            "image": UX_GLASSES_03, "image_hover": UX_GLASSES_07,
            "images": [UX_GLASSES_03, UX_GLASSES_07, UX_GLASSES_08],
            "tags": ["Cat-eye", "Acetato"], "model_3d": "LentesPrueba1"
        },
        {
            "id": "4", "name": "Aviador Pro D012", "brand": "SkyLens",
            "style": "Clásico", "material": "Metal", "category": "Diario", "gender": "Unisex",
            "compatibility": 85, "compatible_faces": ["Cuadrado", "Corazón"],
            "description": "El clásico estilo aviador rediseñado con materiales modernos.",
            "image": IMG_AVIADOR_1, "image_hover": IMG_AVIADOR_2,
            "images": [IMG_AVIADOR_1, IMG_AVIADOR_2, IMG_AVIADOR_3, IMG_AVIADOR_4],
            "tags": ["Aviador", "Metal"], "model_3d": "LentesPrueba1"
        },
        {
            "id": "5", "name": "Round Retro E345", "brand": "RetroSpec",
            "style": "Retro", "material": "Acetato", "category": "Moda", "gender": "Unisex",
            "compatibility": 78, "compatible_faces": ["Cuadrado", "Triangular"],
            "description": "Inspiración vintage con forma perfectamente redonda.",
            "image": UX_GLASSES_09, "image_hover": UX_GLASSES_11,
            "images": [UX_GLASSES_09, UX_GLASSES_11, UX_GLASSES_06],
            "tags": ["Redondo", "Acetato"], "model_3d": "LentesPrueba1"
        },
        {
            "id": "6", "name": "Titanium Air F678", "brand": "LightFrame",
            "style": "Moderno", "material": "Titanio", "category": "Profesional", "gender": "Hombre",
            "compatibility": 90, "compatible_faces": ["Ovalado", "Corazón", "Redondo"],
            "description": "Ultraligero y resistente, diseñado para quienes buscan comodidad sin sacrificar estilo.",
            "image": UX_GLASSES_04, "image_hover": UX_GLASSES_12,
            "images": [UX_GLASSES_04, UX_GLASSES_12, UX_GLASSES_02],
            "tags": ["Moderno", "Titanio"], "model_3d": "LentesPrueba1"
        },
        {
            "id": "7", "name": "Browline Vintage G901", "brand": "HeritageLens",
            "style": "Retro", "material": "Mixto", "category": "Diario", "gender": "Unisex",
            "compatibility": 72, "compatible_faces": ["Cuadrado", "Ovalado"],
            "description": "Fusión de materiales con frente de acetato y varillas metálicas.",
            "image": UX_GLASSES_06, "image_hover": UX_GLASSES_09,
            "images": [UX_GLASSES_06, UX_GLASSES_09, UX_GLASSES_05],
            "tags": ["Retro", "Mixto"], "model_3d": "LentesPrueba1"
        },
        {
            "id": "8", "name": "Elegance Frame H234", "brand": "LuxeOptic",
            "style": "Elegante", "material": "Metal", "category": "Moda", "gender": "Mujer",
            "compatibility": 86, "compatible_faces": ["Redondo", "Corazón"],
            "description": "Marco metálico delicado con detalles refinados.",
            "image": UX_GLASSES_07, "image_hover": UX_GLASSES_03,
            "images": [UX_GLASSES_07, UX_GLASSES_03, UX_GLASSES_08],
            "tags": ["Elegante", "Metal"], "model_3d": "LentesPrueba1"
        },
        {
            "id": "9", "name": "Sport Active I567", "brand": "FlexView",
            "style": "Deportivo", "material": "Mixto", "category": "Deportivo", "gender": "Unisex",
            "compatibility": 65, "compatible_faces": ["Ovalado", "Cuadrado", "Triangular"],
            "description": "Diseñado para la vida activa con materiales flexibles.",
            "image": UX_GLASSES_10, "image_hover": UX_GLASSES_04,
            "images": [UX_GLASSES_10, UX_GLASSES_04, UX_GLASSES_12],
            "tags": ["Deportivo", "Mixto"], "model_3d": "LentesPrueba1"
        },
        {
            "id": "10", "name": "Rimless Pure J890", "brand": "ClearSight",
            "style": "Moderno", "material": "Titanio", "category": "Profesional", "gender": "Unisex",
            "compatibility": 82, "compatible_faces": ["Ovalado", "Corazón"],
            "description": "Sin montura para quienes prefieren discreción.",
            "image": UX_GLASSES_12, "image_hover": UX_GLASSES_04,
            "images": [UX_GLASSES_12, UX_GLASSES_04, UX_GLASSES_02],
            "tags": ["Sin montura", "Titanio"], "model_3d": "LentesPrueba1"
        },
        {
            "id": "11", "name": "Bold Acetate K123", "brand": "UrbanFrame",
            "style": "Moderno", "material": "Acetato", "category": "Diario", "gender": "Hombre",
            "compatibility": 76, "compatible_faces": ["Redondo", "Triangular"],
            "description": "Acetato grueso con colores vibrantes y presencia audaz.",
            "image": UX_GLASSES_05, "image_hover": UX_GLASSES_10,
            "images": [UX_GLASSES_05, UX_GLASSES_10, UX_GLASSES_02],
            "tags": ["Cuadrado", "Acetato"], "model_3d": "LentesPrueba1"
        },
        {
            "id": "12", "name": "Classic Metal L456", "brand": "VisionPlus",
            "style": "Clásico", "material": "Metal", "category": "Profesional", "gender": "Unisex",
            "compatibility": 89, "compatible_faces": ["Ovalado", "Cuadrado", "Redondo"],
            "description": "Diseño clásico de metal que equilibra forma y función.",
            "image": UX_GLASSES_11, "image_hover": UX_GLASSES_09,
            "images": [UX_GLASSES_11, UX_GLASSES_09, UX_GLASSES_01],
            "tags": ["Clásico", "Metal"], "model_3d": "LentesPrueba1"
        },
        {
            "id": "13", "name": "Lentes Prueba 1", "brand": "Prueba3D",
            "style": "Moderno", "material": "Acetato", "category": "Diario", "gender": "Unisex",
            "compatibility": 90, "compatible_faces": ["Ovalado", "Redondo", "Cuadrado", "Corazón"],
            "description": "Primer modelo 3D real integrado al sistema de prueba virtual con renderizado Three.js.",
            "image": UX_GLASSES_08, "image_hover": UX_GLASSES_01,
            "images": [UX_GLASSES_08, UX_GLASSES_01, UX_GLASSES_03],
            "tags": ["3D", "Moderno"], "model_3d": "LentesPrueba1"
        },
        {
            "id": "14", "name": "Lentes Aviador", "brand": "Sungait",
            "style": "Moderno", "material": "Metal", "category": "Diario", "gender": "Unisex",
            "compatibility": 80, "compatible_faces": ["Redondo", "Ovalado", "Triangular"],
            "description": "Estilo clásico combinado: gafas cuadradas clásicas nunca pasan de moda, adecuadas para la mayoría de las caras.",
            "image": IMG_AVIADOR_1, "image_hover": IMG_AVIADOR_2,
            "images": [IMG_AVIADOR_1, IMG_AVIADOR_2, IMG_AVIADOR_3, IMG_AVIADOR_4],
            "tags": ["Metal", "Moderno"], "model_3d": "LentesAviador"
        },
    ]

    for g in glasses:
        cursor.execute("""
            INSERT INTO glasses (id, name, brand, style, material, category, gender,
                                 compatibility, compatible_faces, description, image, image_hover,
                                 images, tags, model_3d)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            g["id"], g["name"], g["brand"], g["style"], g["material"],
            g["category"], g["gender"], g["compatibility"],
            json.dumps(g["compatible_faces"]),
            g["description"], g["image"], g["image_hover"],
            json.dumps(g["images"]),
            json.dumps(g["tags"]),
            g.get("model_3d", "")
        ))


def _seed_face_shapes(cursor):
    shapes = [
        ("Ovalado", "Rostro equilibrado, ligeramente más largo que ancho. Pómulos son la zona más ancha.",
         ["Casi todos los estilos te favorecen", "Prueba formas rectangulares o cuadradas", "Los cat-eye resaltan tus pómulos"]),
        ("Redondo", "Rostro casi tan ancho como largo, líneas suaves y mentón redondeado.",
         ["Busca armazones angulares para dar estructura", "Los marcos cuadrados y rectangulares te favorecen", "Evita marcos redondos que acentúen la redondez"]),
        ("Cuadrado", "Rostro con mandíbula angular y marcada. Frente, pómulos y mandíbula de ancho similar.",
         ["Armazones redondos u ovalados suavizan tus ángulos", "Los marcos delgados equilibran tus rasgos", "Evita marcos cuadrados muy gruesos"]),
        ("Rectángulo", "Rostro notablemente más largo que ancho, angular y con mandíbula definida.",
         ["Marcos anchos acortan visualmente el rostro", "Prueba marcos decorativos o de color", "Los marcos gruesos aportan equilibrio"]),
        ("Triangular", "Mandíbula es la zona más ancha. Frente más estrecha que la línea mandibular.",
         ["Marcos anchos arriba equilibran la mandíbula", "Cat-eye y browline son ideales", "Evita marcos estrechos en la parte superior"]),
        ("Corazón", "Frente amplia que se angosta hacia un mentón puntiagudo.",
         ["Marcos que sean más anchos abajo", "Aviadores y marcos ligeros funcionan bien", "Evita marcos muy decorados arriba"]),
        ("Diamante", "Pómulos prominentes con frente y mandíbula estrechas.",
         ["Cat-eye y marcos ovalados resaltan tus pómulos", "Busca marcos con detalle en la ceja", "Evita marcos muy estrechos"]),
    ]
    for name, desc, tips in shapes:
        cursor.execute("INSERT INTO face_shapes (name, description, tips) VALUES (?, ?, ?)",
                       (name, desc, json.dumps(tips)))


def _row_to_dict(row):
    d = dict(row)
    for field in ("compatible_faces", "images", "tags", "tips"):
        if field in d and d[field]:
            try:
                d[field] = json.loads(d[field])
            except (json.JSONDecodeError, TypeError):
                pass
    return d


def get_all_glasses():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM glasses ORDER BY compatibility DESC").fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def get_glass_by_id(glass_id: str):
    conn = get_connection()
    row = conn.execute("SELECT * FROM glasses WHERE id = ?", (glass_id,)).fetchone()
    conn.close()
    return _row_to_dict(row) if row else None


def get_glasses_by_face_shape(face_shape: str):
    conn = get_connection()
    rows = conn.execute("SELECT * FROM glasses WHERE compatible_faces LIKE ? ORDER BY compatibility DESC",
                        (f'%"{face_shape}"%',)).fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def get_all_face_shapes():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM face_shapes").fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def get_face_shape_by_name(name: str):
    conn = get_connection()
    row = conn.execute("SELECT * FROM face_shapes WHERE name = ?", (name,)).fetchone()
    conn.close()
    return _row_to_dict(row) if row else None


init_db()

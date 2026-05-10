# OptiMatch — Sistema de Recomendación de Lentes

Sistema web de recomendación de lentes basado en reconocimiento facial.
Frontend en HTML/CSS/JS puro + Backend en FastAPI (Python) + SQLite + OpenCV/MediaPipe.

## Estructura del Proyecto

```
OptiMatch/
├── main.py                    # Backend FastAPI (rutas + API)
├── database.py                # Módulo SQLite (CRUD lentes + formas de rostro)
├── face_analyzer.py           # Análisis facial con OpenCV + MediaPipe (adaptado de v2)
├── OptiMatch.db               # Base de datos SQLite (se genera automáticamente)
├── requirements.txt           # Dependencias Python
├── static/
│   ├── css/
│   │   └── styles.css         # Design system completo
│   ├── js/
│   │   └── app.js             # JavaScript compartido + API helper
│   └── images/                # (Imágenes locales)
└── templates/
    ├── base.html              # Template base (navbar + estructura)
    ├── home.html              # Pantalla 1: Inicio
    ├── face_capture.html      # Pantalla 2: Captura de rostro (cámara/upload)
    ├── processing.html        # Pantalla 3: Procesando análisis
    ├── results.html           # Pantalla 4: Resultados con filtros
    ├── lens_detail.html       # Pantalla 5: Detalle de lente
    ├── virtual_tryon.html     # Pantalla 6: Prueba virtual (MediaPipe Face Mesh JS)
    ├── catalog.html           # Pantalla 7: Catálogo general
    ├── face_guide.html        # Pantalla 8: Guía de formas de rostro
    └── face_analysis.html     # Pantalla 9: Análisis facial completo
```

## Módulos Principales

### 1. database.py — Base de Datos SQLite
- Tabla `glasses`: 12 modelos con nombre, marca, estilo, material, compatibilidad, imágenes y overlay SVG
- Tabla `face_shapes`: 7 formas de rostro con descripción y consejos
- Se inicializa automáticamente al importar (crea tablas + datos semilla)
- Funciones CRUD: `get_all_glasses()`, `get_glass_by_id()`, `get_glasses_by_face_shape()`, etc.

### 2. face_analyzer.py — Análisis Facial (basado en face_recognition_realtime_version_2.py)
- Usa MediaPipe Face Mesh (API legacy, compatible sin archivo .task)
- Extrae 36 landmarks faciales clave
- Calcula 17+ métricas geométricas normalizadas
- Clasificador vectorial con 7 perfiles (Ovalado, Redondo, Cuadrado, Rectángulo, Triángulo, Corazón, Diamante)
- Sistema de scoring ponderado con reglas de desempate
- Entrada: bytes de imagen → Salida: clasificación completa con confianza

### 3. virtual_tryon.html — Prueba Virtual en Tiempo Real
- **MediaPipe Face Mesh** corriendo en el navegador (JavaScript, no requiere servidor)
- Detección de 468 landmarks faciales en tiempo real
- **8 diseños de lentes** renderizados en Canvas:
  - `classic_oval` — Armazón ovalado clásico
  - `modern_square` — Marco cuadrado moderno
  - `cat_eye` — Estilo cat-eye con alas
  - `aviator` — Aviador con doble puente
  - `round_retro` — Circular retro
  - `browline` — Browline con barra superior gruesa
  - `rimless` — Sin montura (ultraligero)
  - `sport_wrap` — Deportivo envolvente
- Posicionamiento automático usando landmarks de ojos, nariz y sienes
- Rotación e inclinación que sigue los movimientos de la cabeza
- Selector lateral para cambiar entre modelos en tiempo real
- Captura de pantalla con lentes superpuestos

## Instalación y Ejecución

```bash
# 1. Crear entorno virtual
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar servidor
python main.py
```

Abre tu navegador en: **http://127.0.0.1:8000**

## Pantallas (9 vistas)

| #  | Ruta                     | Descripción                          |
|----|--------------------------|--------------------------------------|
| 1  | `/`                      | Inicio con hero y features           |
| 2  | `/analisis`              | Captura con cámara o subir foto      |
| 3  | `/procesando`            | Animación de análisis facial         |
| 4  | `/resultados`            | Grid de lentes + filtros + sidebar   |
| 5  | `/detalle/{id}`          | Detalle del lente con galería        |
| 6  | `/prueba-virtual/{id}`   | **Prueba virtual con cámara AR**     |
| 7  | `/catalogo`              | Catálogo general con búsqueda        |
| 8  | `/guia-rostros`          | Guía de 7 formas de rostro           |
| 9  | `/analisis-completo`     | Resultado detallado del análisis     |

## API Endpoints

| Método | Ruta                           | Descripción                          |
|--------|-------------------------------|----------------------------------------|
| POST   | `/api/upload-photo`           | Analiza foto con OpenCV+MediaPipe     |
| GET    | `/api/glasses`                | Todos los lentes (desde SQLite)       |
| GET    | `/api/glasses/{id}`           | Lente por ID                          |
| GET    | `/api/recommendations/{face}` | Recomendaciones por forma facial      |
| GET    | `/api/face-shapes`            | Formas de rostro con tips             |

## Flujo del Usuario

```
Inicio → Análisis Facial → Procesando → Resultados → Detalle → Prueba Virtual
                                              ↓
                                          Catálogo → Detalle → Prueba Virtual
```

## Tecnologías

| Componente          | Tecnología                              |
|---------------------|-----------------------------------------|
| Backend             | FastAPI + Python 3.10+                  |
| Base de datos       | SQLite3                                 |
| Análisis facial     | OpenCV + MediaPipe (Python, servidor)   |
| Prueba virtual      | MediaPipe Face Mesh (JavaScript, navegador) |
| Frontend            | HTML5 + CSS3 + JavaScript vanilla       |
| Templates           | Jinja2                                  |

## Design System

| Token              | Valor        |
|-------------------|--------------|
| Navy (texto)       | `#0F172A`    |
| Azul (acento)      | `#3B82F6`    |
| Azul hover         | `#2563EB`    |
| Azul claro (bg)    | `#EFF6FF`    |
| Gris secundario    | `#94A3B8`    |
| Fondo página       | `#F8FAFC`    |
| Verde (éxito)      | `#22C55E`    |
| Rojo (error)       | `#EF4444`    |
| Tipografía         | Inter        |
| Border radius      | 12px         |

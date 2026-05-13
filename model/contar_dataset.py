"""
contar_dataset.py
Muestra cuántas fotos hay por clase en tu dataset de entrenamiento.
Busca automáticamente las carpetas típicas.
"""

import os
from pathlib import Path

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Carpetas donde puede estar tu dataset (ajusta si es necesario)
CANDIDATOS = [
    "dataset",
    "data",
    "train",
    "training",
    "datos",
    "dataset/train",
    "data/train",
    "dataset/training",
]

def contar_imagenes(carpeta: Path) -> dict:
    conteo = {}
    for clase in sorted(carpeta.iterdir()):
        if not clase.is_dir():
            continue
        fotos = [f for f in clase.iterdir() if f.suffix.lower() in EXTENSIONES]
        conteo[clase.name] = len(fotos)
    return conteo


def main():
    base = Path(__file__).parent

    dataset_path = None
    for candidato in CANDIDATOS:
        ruta = base / candidato
        if ruta.exists() and ruta.is_dir():
            dataset_path = ruta
            break

    if dataset_path is None:
        print("❌ No se encontró carpeta de dataset automáticamente.")
        print("   Carpetas buscadas:", CANDIDATOS)
        print("\n   Ingresa la ruta manualmente:")
        ruta_manual = input("   >> ").strip()
        dataset_path = Path(ruta_manual)
        if not dataset_path.exists():
            print("❌ Esa ruta tampoco existe. Verifica.")
            return

    print(f"\n📁 Dataset encontrado en: {dataset_path}\n")
    conteo = contar_imagenes(dataset_path)

    if not conteo:
        print("⚠️  No se encontraron subcarpetas con imágenes.")
        return

    total = sum(conteo.values())
    max_fotos = max(conteo.values())
    min_fotos = min(conteo.values())

    print(f"{'Clase':<15} {'Fotos':>8}  {'Barra'}")
    print("-" * 50)
    for clase, n in sorted(conteo.items(), key=lambda x: -x[1]):
        barra = "█" * int((n / max_fotos) * 30)
        emoji = "⚠️ " if n < (total / len(conteo)) * 0.7 else "  "
        print(f"{clase:<15} {n:>6}   {barra} {emoji}")

    print("-" * 50)
    print(f"{'TOTAL':<15} {total:>6}")
    print(f"\n📊 Promedio por clase : {total // len(conteo)} fotos")
    print(f"   Clase con más fotos: {max(conteo, key=conteo.get)} ({max_fotos})")
    print(f"   Clase con menos    : {min(conteo, key=conteo.get)} ({min_fotos})")

    desbalance = max_fotos / min_fotos if min_fotos > 0 else float("inf")
    print(f"\n{'⚠️  DESBALANCE ALTO' if desbalance > 2 else '✅ Balance aceptable'}"
          f" — ratio max/min: {desbalance:.1f}x")
    if desbalance > 2:
        print("   Esto puede causar que el modelo favorezca las clases con más fotos.")
        print("   Recomendado: igualar las clases agregando o eliminando fotos.")


if __name__ == "__main__":
    main()

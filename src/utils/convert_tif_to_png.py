from pathlib import Path
from PIL import Image
import os

def convert_tif_to_png(input_dir: Path, output_dir: Path):
    """Convierte todas las imágenes .tif en un directorio a .png"""
    output_dir.mkdir(exist_ok=True)
    for tif_path in input_dir.glob("*.tif"):
        png_path = output_dir / f"{tif_path.stem}.png"
        img = Image.open(tif_path)
        img.save(png_path, "PNG")
        print(f"Convertido: {tif_path.name} -> {png_path.name}")

if __name__ == "__main__":
    # Rutas de ejemplo (ajústalas)
    input_dir = Path("data/raw/leaf3/")  # Directorio con .tif
    output_dir = Path("data/raw/especie_03/")  # Directorio de salida para .png
    convert_tif_to_png(input_dir, output_dir)
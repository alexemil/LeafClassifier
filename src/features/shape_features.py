import cv2
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple
import pandas as pd
import os
import shutil
from datetime import datetime

# Configura logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de rutas
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Ajusta según tu estructura
output_dir = PROJECT_ROOT / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

# Ruta al archivo CSV existente
existing_csv_path = output_dir / "leaf_features.csv"

# Si el archivo ya existe, haz una copia de seguridad
if existing_csv_path.exists():
    backup_path = output_dir / f"leaf_features_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    shutil.copy(existing_csv_path, backup_path)
    logger.info(f"🗂️ Backup realizado en {backup_path}")

def load_image(image_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Carga una imagen y su versión en escala de grises con manejo de errores."""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen en {image_path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image, gray
    except Exception as e:
        logger.error(f"Error al cargar {image_path.name}: {e}")
        return None, None

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Aplica preprocesamiento para mejorar la detección de contornos."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def detect_leaf_contour(thresh_image: np.ndarray) -> Optional[np.ndarray]:
    """Encuentra el contorno más grande (hoja) con validaciones."""
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.warning("No se encontraron contornos.")
        return None
    return max(contours, key=cv2.contourArea)

def calculate_hu_moments(contour: np.ndarray) -> List[float]:
    """Calcula los primeros dos momentos de Hu."""
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments)
    return [abs(np.log(abs(m))) for m in hu_moments.flatten()[:2]]  # Solo los 2 primeros

def extract_shape_features(contour, gray_image=None) -> Dict[str, float]:
    """Calcula características geométricas de la hoja."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    _, _, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w)/h if h != 0 else 0
    rectangularity = area / (w * h) if (w * h) != 0 else 0

    texture_mean = 0
    if gray_image is not None:
        texture_mean = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    
    return {
        "area": area,
        "perimeter": perimeter,
        "solidity": solidity,
        "aspect_ratio": aspect_ratio,
        "rectangularity": rectangularity,
        "texture_mean": texture_mean
    }

def visualize_contour(original: np.ndarray, contour: np.ndarray, save_path: Optional[Path] = None) -> None:
    """Dibuja el contorno y opcionalmente guarda la imagen."""
    result = cv2.drawContours(original.copy(), [contour], -1, (0, 255, 0), 3)
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    if save_path:
        cv2.imwrite(str(save_path), result)

def process_batch_images(input_dir: Path, output_dir: Path, pattern: str, species: str) -> pd.DataFrame:
    """Procesa todas las imágenes que coincidan con el patrón dado."""
    image_paths = sorted(input_dir.glob(pattern))
    if not image_paths:
        logger.warning(f"No se encontraron imágenes con patrón {pattern} en {input_dir}")
        return pd.DataFrame()
    
    features_list = []
    for img_path in image_paths:
        original, gray = load_image(img_path)
        if original is None or gray is None:
            continue

        thresh = preprocess_image(gray)
        contour = detect_leaf_contour(thresh)
        if contour is None:
            continue

        features = extract_shape_features(contour, gray)

        # Nuevas columnas: momentos de Hu
        hu = calculate_hu_moments(contour)
        features["hu_moment1"] = hu[0]
        features["hu_moment2"] = hu[1]

        features["image_name"] = img_path.name
        features["species"] = species
        features_list.append(features)

        # Guarda imagen procesada
        processed_img_path = output_dir / f"contour_{img_path.name}"
        visualize_contour(original, contour, save_path=processed_img_path)
        logger.info(f"Procesada: {img_path.name}")

    return pd.DataFrame(features_list)

if __name__ == "__main__":
    # Configuración de especies y patrones
    species_config = [
        ("Aliso", "l5nr*.png"),
        ("Olmo", "l1nr*.png"), 
        ("Sauce", "l3nr*.png"),    
        ("Anacahuita", "Cordia_boissieri_*.png"),
        ("Encino", "Quercus_virginiana_*.png"),
        ("Huisache", "vachellia_*.png")
    ]
    
    all_features = []
    for species, pattern in species_config:
        input_dir = PROJECT_ROOT / "data" / "raw" / species
        
        if not input_dir.exists():
            logger.error(f"⚠️ Directorio no encontrado: {input_dir}")
            continue
            
        logger.info(f"🔍 Procesando {species} desde {input_dir}...")
        
        df = process_batch_images(input_dir, output_dir, pattern, species)
        
        if not df.empty:
            all_features.append(df)
    
    # Guardar CSV unificado
    if all_features:
        final_df = pd.concat(all_features)
        csv_path = output_dir / "leaf_features.csv"
        final_df.to_csv(csv_path, index=False)
        logger.info(f"✅ CSV guardado en {csv_path}")
        
        # Mostrar resumen
        print("\nResumen de datos procesados:")
        print(final_df['species'].value_counts())
        print("\nColumnas generadas:")
        print(final_df.columns.tolist())
    else:
        logger.error("❌ No se procesaron imágenes válidas.")

import cv2
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Optional
import pandas as pd

# Configura logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_image(image_path: Path) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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

def preprocess_image(gray_image: np.ndarray) -> np.ndarray:
    """Aplica preprocesamiento para mejorar la detección de contornos."""
    blurred = cv2.GaussianBlur(gray_image, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def detect_leaf_contour(thresh_image: np.ndarray) -> Optional[np.ndarray]:
    """Encuentra el contorno más grande (hoja) con validaciones."""
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.warning("No se encontraron contornos.")
        return None
    return max(contours, key=cv2.contourArea)

def extract_shape_features(contour: np.ndarray) -> Dict[str, float]:
    """Calcula características geométricas de la hoja."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    return {"area": area, "perimeter": perimeter, "solidity": solidity}

def visualize_contour(original: np.ndarray, contour: np.ndarray, save_path: Optional[Path] = None) -> None:
    """Dibuja el contorno y opcionalmente guarda la imagen."""
    result = cv2.drawContours(original.copy(), [contour], -1, (0, 255, 0), 3)
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    if save_path:
        cv2.imwrite(str(save_path), result)
    else:
        cv2.imshow("Contorno", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def process_batch_images(input_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Procesa todas las imágenes (l5nr*, l1nr*, l3nr*) y retorna un DataFrame con especies."""
    # Patrones de búsqueda para cada tipo de imagen
    patterns = ["l5nr*.png", "l1nr*.png", "l3nr*.png"]
    species_mapping = {
        "l5nr": "especie_01",
        "l1nr": "especie_02",  # Ejemplo: "roble"
        "l3nr": "especie_03"   # Ejemplo: "arce"
    }
    
    features_list = []
    for pattern in patterns:
        image_paths = sorted(input_dir.glob(pattern))
        if not image_paths:
            logger.warning(f"No se encontraron imágenes con patrón {pattern} en {input_dir}")
            continue

        for img_path in image_paths:
            original, gray = load_image(img_path)
            if original is None or gray is None:
                continue

            thresh = preprocess_image(gray)
            contour = detect_leaf_contour(thresh)
            if contour is None:
                continue

            # Extrae características y añade especie según el prefijo
            features = extract_shape_features(contour)
            features["image_name"] = img_path.name
            
            # Asigna especie automáticamente (ej: l1nr001.png -> especie_02)
            prefix = img_path.name[:3]  # Extrae "l1n", "l3n", etc.
            features["species"] = species_mapping.get(prefix, "unknown")
            
            features_list.append(features)

def process_batch_images(input_dir: Path, output_dir: Path, pattern: str) -> pd.DataFrame:
    """Procesa todas las imágenes que coincidan con el patrón dado."""
    image_paths = sorted(input_dir.glob(pattern))  # Usa el patrón recibido
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

        features = extract_shape_features(contour)
        features["image_name"] = img_path.name
        features_list.append(features)

        processed_img_path = output_dir / f"contour_{img_path.name}"
        visualize_contour(original, contour, save_path=processed_img_path)
        logger.info(f"Procesada: {img_path.name}")

    return pd.DataFrame(features_list)
    
if __name__ == "__main__":
    # Configura rutas
    output_dir = Path("data/processed/")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Procesar CADA especie por separado
    species_folders = [
        ("especie_01", "l5nr*.png"),
        ("especie_02", "l1nr*.png"),
        ("especie_03", "l3nr*.png")
    ]
    
    all_features = []
    for folder, pattern in species_folders:
        input_dir = Path(f"data/raw/{folder}/")
        df = process_batch_images(input_dir, output_dir, pattern)
        if not df.empty:
            df["species"] = folder  # Añade columna de especie
            all_features.append(df)
    
    # Guardar CSV unificado
    if all_features:
        final_df = pd.concat(all_features)
        csv_path = output_dir / "leaf_features.csv"
        final_df.to_csv(csv_path, index=False)
        logger.info(f"✅ CSV guardado en {csv_path}")
    else:
        logger.error("No se procesaron imágenes válidas.")
import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, 
    QFileDialog, QVBoxLayout, QWidget, QTableWidget, 
    QTableWidgetItem, QHBoxLayout
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
import tensorflow as tf
import os
import joblib
from pathlib import Path
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.features.shape_features import preprocess_image, extract_shape_features, detect_leaf_contour, calculate_hu_moments

MODEL_PATH = Path(__file__).resolve().parents[1] / "models/tf_leaf_classifier.h5"
ENCODER_PATH = Path(__file__).resolve().parents[1] / "models/tf_label_encoder.pkl"

class LeafClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Clasificador de Hojas")
        self.setGeometry(200, 200, 1000, 800)  # Aumenté el tamaño para la tabla

        # Carga del modelo y del codificador de etiquetas
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"⚠️ No se encontró el modelo en: {MODEL_PATH}")
        if not os.path.exists(ENCODER_PATH):
            raise FileNotFoundError(f"⚠️ No se encontró el codificador en: {ENCODER_PATH}")

        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.label_encoder = joblib.load(ENCODER_PATH)
        self.class_labels = self.label_encoder.classes_
        self.loaded_image = None

        # Configuración de la interfaz
        self.init_ui()

    def init_ui(self):
        """Configura todos los elementos de la interfaz gráfica"""
        central_widget = QWidget()
        main_layout = QVBoxLayout()

        # Panel de imagen
        self.image_label = QLabel("Cargar o capturar una imagen...", self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(800, 300)

        # Tabla de características
        self.feature_table = QTableWidget()
        self.feature_table.setColumnCount(2)
        self.feature_table.setHorizontalHeaderLabels(["Característica", "Valor"])
        self.feature_table.verticalHeader().setVisible(False)
        self.feature_table.setVisible(False)  # Inicialmente oculta

        # Botones
        button_layout = QHBoxLayout()
        self.load_button = QPushButton("Cargar Imagen", self)
        self.load_button.clicked.connect(self.load_image)
        self.classify_button = QPushButton("Clasificar Hoja", self)
        self.classify_button.clicked.connect(self.classify_leaf)

        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.classify_button)

        # Etiqueta de resultados
        self.result_label = QLabel("", self)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font-size: 14px;")

        # Agregar elementos al layout
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.feature_table)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.result_label)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar imagen", "", 
            "Images (*.png *.jpg *.jpeg)"
        )
        if file_name:
            self.loaded_image = cv2.imread(file_name)
            self.display_image(self.loaded_image)
            self.feature_table.setVisible(False)  # Ocultar tabla al cargar nueva imagen
            self.result_label.setText("")

    def display_image(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_image.shape
        bytes_per_line = channels * width
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image).scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio
        )
        self.image_label.setPixmap(pixmap)

    def validate_leaf_features(self, features):
        """Verifica rangos físicos plausibles para una hoja"""
        checks = [
            (features.get("area", 0) > 1000, "Área demasiado pequeña"),
            (features.get("aspect_ratio", 1) < 10, "Forma no plausible"),
            (0.3 < features.get("solidity", 0.5) < 1.0, "Solidez fuera de rango"),
            (features.get("perimeter", 0) > 100, "Perímetro demasiado pequeño")
        ]
        
        for condition, message in checks:
            if not condition:
                logger.warning(f"Validación fallida: {message}")
                return False
        return True

    def is_outlier(self, feature_values, normalizer, threshold=3.0):
        """Detecta si las características están fuera del rango esperado"""
        for i, (feature, value) in enumerate(zip(normalizer["features_order"], feature_values)):
            z_score = abs(value)  # Ya están normalizados (media=0, std=1)
            if z_score > threshold:
                logger.warning(f"Característica outlier: {feature} (z-score: {z_score:.2f})")
                return True
        return False

    def classify_leaf(self):
        if self.loaded_image is None:
            self.result_label.setText("Por favor, carga o captura una imagen.")
            return

        try:
            # 1. Preprocesamiento de imagen
            original = self.loaded_image.copy()
            gray_image = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            processed = preprocess_image(gray_image)
            contour = detect_leaf_contour(processed)
            
            if contour is None:
                self.result_label.setText("No se detectó una hoja válida.")
                return

            # 2. Extracción de características
            features = extract_shape_features(contour, gray_image)
            hu = calculate_hu_moments(contour)
            features["hu_moment1"] = hu[0]
            features["hu_moment2"] = hu[1]

            # 3. Mostrar características en la tabla
            feature_names = [
                "Área", "Perímetro", "Solidez", "Relación de aspecto",
                "Rectangularidad", "Momento Hu 1", "Momento Hu 2", "Textura (media)"
            ]
            feature_keys = [
                "area", "perimeter", "solidity", "aspect_ratio",
                "rectangularity", "hu_moment1", "hu_moment2", "texture_mean"
            ]

            self.feature_table.setRowCount(len(feature_keys))
            self.feature_table.clearContents()
            
            for i, (key, label) in enumerate(zip(feature_keys, feature_names)):
                value = features.get(key, 0)
                self.feature_table.setItem(i, 0, QTableWidgetItem(label))
                self.feature_table.setItem(i, 1, QTableWidgetItem(f"{value:.4f}"))
            
            self.feature_table.setVisible(True)
            self.feature_table.resizeColumnsToContents()

            # 4. Verificación de características faltantes
            missing_keys = [key for key in feature_keys if key not in features]
            if missing_keys:
                self.result_label.setText(f"Faltan características: {', '.join(missing_keys)}")
                return

            # 5. Carga del normalizador
            normalizer_path = Path(__file__).resolve().parents[1] / "models/tf_normalizer.pkl"
            if not os.path.exists(normalizer_path):
                raise FileNotFoundError(f"No se encontró el normalizador en: {normalizer_path}")
            
            normalizer = joblib.load(normalizer_path)
            features_order = normalizer.get("features_order", feature_keys)
            
            # 6. Normalización de características
            feature_values = []
            for feature in features_order:
                if feature in features:
                    mean = normalizer["mean"].get(feature, 0)
                    std = normalizer["std"].get(feature, 1)
                    normalized_value = (features[feature] - mean) / std if std != 0 else 0
                    feature_values.append(normalized_value)
                else:
                    feature_values.append(0)
            
            # 7. Validación de características físicas
            if not self.validate_leaf_features(features):
                self.result_label.setText("⚠️ Características no válidas para una hoja")
                return

            # 8. Detección de outliers
            if self.is_outlier(feature_values, normalizer):
                self.result_label.setText("⚠️ Patrón no reconocido")
                return
            
            # 9. Predicción
            input_features = np.array([feature_values], dtype=np.float32)
            predictions = self.model.predict(input_features, verbose=0)
            
            # 10. Post-procesamiento de resultados
            max_confidence = np.max(predictions[0])
            confidence_threshold = 0.7
            margin_threshold = 0.3

            if max_confidence < confidence_threshold:
                self.result_label.setText("⚠️ Baja confianza en la predicción")
                return

            sorted_probs = np.sort(predictions[0])[::-1]
            if (sorted_probs[0] - sorted_probs[1]) < margin_threshold:
                self.result_label.setText("⚠️ Múltiples especies posibles")
                return

            predicted_class_idx = np.argmax(predictions[0])
            class_name = self.class_labels[predicted_class_idx]
            self.result_label.setText(
                f"✅ Especie: {class_name}\n"
                f"Confianza: {max_confidence:.2%}"
            )
            
            # 11. Visualización de resultados
            self.visualize_results(original, contour, gray_image, predictions)

        except Exception as e:
            logger.error(f"Error en clasificación: {str(e)}", exc_info=True)
            self.result_label.setText(f"Error durante el análisis: {str(e)}")

    def visualize_results(self, original, contour, gray_image, predictions):
        """Muestra los resultados visuales de la clasificación"""
        # Imagen con contorno y bounding box
        contour_image = original.copy()
        cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(contour_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # Imagen con fondo difuminado
        mask = np.zeros_like(gray_image)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        blurred = cv2.GaussianBlur(original, (25, 25), 0)
        result_image = np.where(mask[..., None] == 255, original, blurred)
        
        # Combinar y mostrar imágenes
        target_size = (400, 300)
        resized_result = cv2.resize(result_image, target_size)
        resized_contour = cv2.resize(contour_image, target_size)
        combined_display = np.hstack((resized_result, resized_contour))
        
        rgb = cv2.cvtColor(combined_display, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb.shape
        bytes_per_line = channel * width
        qimg = QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap)

        # Mostrar gráfico de probabilidades
        plt.figure(figsize=(8, 3))
        plt.bar(self.class_labels, predictions[0], color='seagreen')
        plt.xticks(rotation=45)
        plt.ylabel("Probabilidad")
        plt.title("Distribución de la Predicción")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LeafClassifierApp()
    window.show()
    sys.exit(app.exec())
import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, 
    QVBoxLayout, QWidget, QFileDialog, QHBoxLayout
)
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import Qt
from pathlib import Path
import joblib
import cv2
import numpy as np
import logging
from typing import Optional
from PyQt6.QtCore import QTimer

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LeafClassifierGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Clasificador de Hojas")
        self.setGeometry(100, 100, 1000, 800)
        
        # Variables de estado
        self.current_image: Optional[np.ndarray] = None
        self.current_result: Optional[str] = None
        self.camera_active = False
        self.camera = None
        self.timer = QTimer()
        
        self.setup_ui()
        self.load_ml_models()
        self.setup_styles()
        
        # Configuración de la cámara
        self.timer.timeout.connect(self.update_frame)
        self.camera_btn.clicked.connect(self.toggle_camera)
        self.capture_btn.clicked.connect(self.capture_image)

    def setup_ui(self):
        """Configura los elementos de la interfaz."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Widgets
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(600, 400)
        
        self.result_label = QLabel("Resultado aparecerá aquí")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setWordWrap(True)
        
        # Botones principales
        self.load_btn = QPushButton("Cargar Imagen")
        self.load_btn.clicked.connect(self.load_image)
        
        self.save_btn = QPushButton("Guardar Resultado")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        
        # Botones de cámara
        self.camera_btn = QPushButton("Iniciar Cámara")
        self.capture_btn = QPushButton("Capturar Imagen")
        self.capture_btn.setEnabled(False)
        
        # Layouts
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_btn)
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.camera_btn)
        button_layout.addWidget(self.capture_btn)
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.result_label)
        main_layout.addLayout(button_layout)
        
        self.central_widget.setLayout(main_layout)

    def setup_styles(self):
        """Aplica estilos CSS."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f2f5;
            }
            QLabel {
                font-size: 16px;
                margin: 10px;
                color: #333;
            }
            QLabel#result_label {
                font-size: 14px;
                background-color: #ffffff;
                padding: 15px;
                border-radius: 5px;
                border: 1px solid #ddd;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 12px;
                font-size: 16px;
                border: none;
                border-radius: 5px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.result_label.setObjectName("result_label")
        
        # Fuente personalizada
        font = QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.setFont(font)
        
    def load_ml_models(self):
        """Carga los modelos de machine learning."""
        try:
            models_dir = Path(__file__).parent.parent / "models"
            self.model = joblib.load(models_dir / "leaf_classifier.pkl")
            self.encoder = joblib.load(models_dir / "label_encoder.pkl")
            logger.info("Modelos ML cargados correctamente")
        except Exception as e:
            logger.error(f"Error cargando modelos: {e}")
            self.result_label.setText("Error: No se pudieron cargar los modelos")
            self.load_btn.setEnabled(False)
            
    def load_image(self):
        """Carga y procesa una imagen para clasificación."""
        try:
            filepath, _ = QFileDialog.getOpenFileName(
                self, 
                "Seleccionar Imagen", 
                "", 
                "Imágenes (*.png *.jpg *.jpeg)"
            )
            
            if not filepath:
                return
                
            # Carga y procesamiento de imagen
            self.current_image = cv2.imread(filepath)
            if self.current_image is None:
                raise ValueError("No se pudo cargar la imagen")
                
            # Procesamiento
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                raise ValueError("No se detectaron contornos")
                
            contour = max(contours, key=cv2.contourArea)
            
            # Extracción de características
            features = self.extract_features(contour)
            
            # Predicción
            species, confidence, all_probs = self.predict_species(features)
            
            # Visualización
            self.display_results(filepath, contour, species, confidence, all_probs)
            self.save_btn.setEnabled(True)
            
        except Exception as e:
            logger.error(f"Error procesando imagen: {e}")
            self.result_label.setText(f"Error: {str(e)}")
            self.save_btn.setEnabled(False)
            
    def extract_features(self, contour) -> np.ndarray:
        """Extrae características morfológicas del contorno."""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        hull = cv2.convexHull(contour)
        solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
        
        return np.array([[area, perimeter, solidity]])
        
    def predict_species(self, features):
        """Realiza la predicción usando el modelo ML."""
        probabilities = self.model.predict_proba(features)[0]
        top_pred_idx = np.argmax(probabilities)
        species = self.encoder.classes_[top_pred_idx]
        confidence = probabilities[top_pred_idx] * 100
        
        return species, confidence, probabilities
        
    def display_results(self, image_path: str, contour, species: str, confidence: float, probs: np.ndarray):
        """Muestra los resultados en la interfaz."""
        # Dibuja contorno sobre la imagen
        result_image = self.current_image.copy()
        cv2.drawContours(result_image, [contour], -1, (0, 255, 0), 3)
        
        # Convierte a QPixmap
        height, width, _ = result_image.shape
        bytes_per_line = 3 * width
        q_img = QImage(
            result_image.data, 
            width, 
            height, 
            bytes_per_line, 
            QImage.Format.Format_BGR888
        )
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(
            pixmap.scaled(
                600, 
                400, 
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )
        
        # Formatea texto de resultados
        prob_text = "\n".join([
            f"• {self.encoder.classes_[i]}: {prob*100:.1f}%" 
            for i, prob in enumerate(probs)
        ])
        
        self.current_result = (
            f"🔍 Resultado:\n"
            f"Especie: <b>{species}</b>\n"
            f"Confianza: <b>{confidence:.1f}%</b>\n\n"
            f"📊 Probabilidades:\n{prob_text}"
        )
        self.result_label.setText(self.current_result)
        
    def save_results(self):
        """Guarda la imagen procesada y los resultados."""
        try:
            if not self.current_image or not self.current_result:
                return
                
            filepath, _ = QFileDialog.getSaveFileName(
                self,
                "Guardar Resultados",
                "",
                "PNG Image (*.png);;JPEG Image (*.jpg)"
            )
            
            if filepath:
                # Guarda imagen con contorno
                result_image = self.current_image.copy()
                cv2.putText(
                    result_image,
                    f"Especie: {self.current_result.split('Especie: ')[1].split('\n')[0]}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )
                cv2.imwrite(filepath, result_image)
                
                # Guarda reporte de texto
                txt_path = Path(filepath).with_suffix('.txt')
                with open(txt_path, 'w') as f:
                    f.write(self.current_result)
                
                self.result_label.setText(
                    f"{self.current_result}\n\n"
                    f"✅ Resultados guardados en:\n{filepath}"
                )
                
        except Exception as e:
            logger.error(f"Error guardando resultados: {e}")
            self.result_label.setText(f"Error al guardar: {str(e)}")

    def toggle_camera(self):
        """Activa/desactiva la cámara."""
        if not self.camera_active:
            self.camera = cv2.VideoCapture(0)  # 0 para la cámara predeterminada
            if not self.camera.isOpened():
                self.result_label.setText("Error: No se detectó cámara")
                return
            
            self.camera_active = True
            self.camera_btn.setText("Detener Cámara")
            self.capture_btn.setEnabled(True)
            self.timer.start(30)  # Actualiza cada 30 ms
        else:
            self.timer.stop()
            self.camera.release()
            self.camera_active = False
            self.camera_btn.setText("Iniciar Cámara")
            self.capture_btn.setEnabled(False)
            self.image_label.clear()

    def update_frame(self):
        """Actualiza el frame de la cámara en el QLabel."""
        ret, frame = self.camera.read()
        if ret:
            # Convierte BGR (OpenCV) a RGB (QPixmap)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                600, 400, Qt.AspectRatioMode.KeepAspectRatio
            ))

    def capture_image(self):
        """Captura el frame actual para procesamiento."""
        if self.camera_active:
            ret, frame = self.camera.read()
            if ret:
                self.process_captured_image(frame)

    def process_captured_image(self, image):
        """Procesa la imagen capturada (similar a load_image)."""
        try:
            self.current_image = image.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                raise ValueError("No se detectaron contornos")
                
            contour = max(contours, key=cv2.contourArea)
            
            # Extracción de características
            features = self.extract_features(contour)
            
            # Predicción
            species, confidence, all_probs = self.predict_species(features)
            
            # Visualización
            self.display_results("Captura de cámara", contour, species, confidence, all_probs)
            self.save_btn.setEnabled(True)
            
            # Guarda la imagen temporalmente (opcional)
            cv2.imwrite("data/captured_temp.png", image)
            
        except Exception as e:
            logger.error(f"Error procesando imagen: {e}")
            self.result_label.setText(f"Error: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LeafClassifierGUI()
    window.show()
    sys.exit(app.exec())
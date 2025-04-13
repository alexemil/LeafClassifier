import tensorflow as tf     
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from pathlib import Path
import logging


# Configuración
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data/processed"
MODEL_PATH = PROJECT_ROOT / "models/leaf_cnn.h5"

def build_model(num_classes):
    """Construye modelo basado en MobileNetV2"""
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)

def train_model():
    # Configuración de aumentación de datos
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    # Flujo de datos
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR / "train",
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    
    val_generator = train_datagen.flow_from_directory(
        DATA_DIR / "train",
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    
    # Construir y compilar modelo
    model = build_model(num_classes=len(train_generator.class_indices))
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Entrenamiento
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=30
    )
    
    # Guardar modelo
    model.save(MODEL_PATH)
    logger.info(f"Modelo CNN guardado en {MODEL_PATH}")
    
    # Generar reporte
    y_true = val_generator.classes
    y_pred = np.argmax(model.predict(val_generator), axis=1)
    
    report = classification_report(y_true, y_pred, target_names=list(train_generator.class_indices.keys()))
    logger.info("\nReporte de Clasificación:\n" + report)
    
    # Guardar reporte
    report_df = pd.DataFrame(classification_report(y_true, y_pred, target_names=list(train_generator.class_indices.keys()), output_dict=True)).transpose()
    report_df.to_csv(PROJECT_ROOT / "reports/cnn_classification_report.csv")

if __name__ == "__main__":
    train_model()
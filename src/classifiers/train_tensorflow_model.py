import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score
from pathlib import Path
import matplotlib.pyplot as plt 
import seaborn as sns
import logging
import joblib
import json
from imblearn.over_sampling import SMOTE

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data/processed/leaf_features.csv"
MODEL_PATH = PROJECT_ROOT / "models/tf_leaf_classifier.h5"
ENCODER_PATH = PROJECT_ROOT / "models/tf_label_encoder.pkl"
NORMALIZER_PATH = PROJECT_ROOT / "models/tf_normalizer.pkl"
PLOTS_DIR = PROJECT_ROOT / "reports/figures"
METRICS_PATH = PLOTS_DIR / "tf_metrics.json"

# Asegurar que exista el directorio de figuras
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def load_and_prepare_data(path: Path) -> tuple:
    """Carga y prepara los datos con verificación exhaustiva"""
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    df = pd.read_csv(path)
    logger.info(f"Dataset cargado. Dimensiones: {df.shape}")
    
    # Verificación de características
    required_features = [
        "area", "perimeter", "solidity", "aspect_ratio", 
        "rectangularity", "hu_moment1", "hu_moment2", "texture_mean"
    ]
    
    missing_features = set(required_features) - set(df.columns)
    if missing_features:
        raise ValueError(f"Faltan características requeridas: {missing_features}")
    
    # Verificación de clases
    class_distribution = df["species"].value_counts()
    logger.info("Distribución de clases:\n" + str(class_distribution))
    
    if len(class_distribution) < 2:
        raise ValueError("Se necesitan al menos 2 clases para clasificación")
    
    # Normalización y encoding
    le = LabelEncoder()
    df["species_encoded"] = le.fit_transform(df["species"])
    
    # Guardar estadísticas de normalización
    normalizer = {
        "mean": df[required_features].mean().to_dict(),
        "std": df[required_features].std().to_dict(),
        "features_order": required_features  # Orden fijo
    }
    joblib.dump(normalizer, NORMALIZER_PATH)
    
    # Aplicar normalización
    X = df[required_features].copy()
    for feature in required_features:
        std = normalizer["std"][feature]
        X[feature] = (X[feature] - normalizer["mean"][feature]) / (std if std > 0 else 1.0)
    
    y = df["species_encoded"]
    
    # Balanceo de clases con SMOTE si es necesario
    if class_distribution.min() / class_distribution.max() < 0.5:
        logger.info("Aplicando SMOTE para balancear clases...")
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
    
    return X.values, y, le

def build_model(input_shape: int, num_classes: int) -> tf.keras.Model:
    """Construye un modelo con arquitectura mejorada"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def evaluate_model(model, X_test, y_test, label_encoder):
    """Evaluación detallada del modelo con visualizaciones para reporte"""
    try:
        # Predicciones
        y_pred = np.argmax(model.predict(X_test), axis=1)
        report = classification_report(y_test, y_pred, 
                                    target_names=label_encoder.classes_, 
                                    output_dict=True)
        
        # 1. Matriz de Confusión (Figura 1)
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                   annot_kws={"size": 10}, cbar=False,
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_)
        plt.title("Matriz de Confusión", fontsize=14, pad=20)
        plt.xlabel("Predicción", fontsize=12, labelpad=10)
        plt.ylabel("Verdadero", fontsize=12, labelpad=10)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Gráfico de Barras de Métricas por Clase (Figura 2)
        metrics = ['precision', 'recall', 'f1-score']
        class_names = label_encoder.classes_
        
        plt.figure(figsize=(12, 6))
        width = 0.25
        x = np.arange(len(class_names))
        
        for i, metric in enumerate(metrics):
            values = [report[cls][metric] for cls in class_names]
            plt.bar(x + i*width, values, width, label=metric.capitalize())
        
        plt.xlabel('Especies', fontsize=12, labelpad=10)
        plt.ylabel('Puntuación', fontsize=12, labelpad=10)
        plt.title('Métricas por Especie', fontsize=14, pad=20)
        plt.xticks(x + width, class_names, rotation=45, ha='right', fontsize=10)
        plt.yticks(np.arange(0, 1.1, 0.1), fontsize=10)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "metrics_by_class.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Gráfico de F1-Score por Clase (Figura 3)
        plt.figure(figsize=(10, 6))
        f1_scores = [report[cls]['f1-score'] for cls in class_names]
        
        bars = plt.barh(class_names, f1_scores, color='#4c72b0')
        plt.xlim(0, 1)
        plt.title('F1-Score por Especie', fontsize=14, pad=20)
        plt.xlabel('F1-Score', fontsize=12, labelpad=10)
        plt.ylabel('')
        
        # Añadir valores dentro de las barras
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{width:.2f}',
                   va='center', ha='left', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "f1_scores.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Gráfico de Precisión vs Recall (Figura 4)
        plt.figure(figsize=(8, 8))
        precisions = [report[cls]['precision'] for cls in class_names]
        recalls = [report[cls]['recall'] for cls in class_names]
        
        plt.scatter(precisions, recalls, s=100, alpha=0.7, color='#55a868')
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.3)
        
        # Etiquetar puntos
        for i, cls in enumerate(class_names):
            plt.text(precisions[i]+0.02, recalls[i]-0.02, cls, 
                   fontsize=9, ha='left', va='center')
        
        plt.title('Precisión vs Recall por Especie', fontsize=14, pad=20)
        plt.xlabel('Precisión', fontsize=12, labelpad=10)
        plt.ylabel('Recall', fontsize=12, labelpad=10)
        plt.xlim(0, 1.05)
        plt.ylim(0, 1.05)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "precision_vs_recall.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            "confusion_matrix": str(PLOTS_DIR / "confusion_matrix.png"),
            "metrics_by_class": str(PLOTS_DIR / "metrics_by_class.png"),
            "f1_scores": str(PLOTS_DIR / "f1_scores.png"),
            "precision_vs_recall": str(PLOTS_DIR / "precision_vs_recall.png")
        }
        
    except Exception as e:
        logger.error(f"Error en evaluación: {str(e)}", exc_info=True)
        raise

def plot_metrics_by_class(report, class_names):
    """Genera gráfico de barras con métricas por clase"""
    metrics = ['precision', 'recall', 'f1-score']
    data = {m: [] for m in metrics}
    
    for class_name in class_names:
        for m in metrics:
            data[m].append(report[class_name][m])
    
    x = np.arange(len(class_names))
    width = 0.25
    
    plt.figure(figsize=(12, 6))
    for i, (m, values) in enumerate(data.items()):
        plt.bar(x + i*width, values, width, label=m)
    
    plt.xlabel('Clases')
    plt.ylabel('Puntuación')
    plt.title('Métricas por clase')
    plt.xticks(x + width, class_names, rotation=45, ha='right')
    plt.legend(loc='lower right')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    
    metrics_path = PLOTS_DIR / "tf_class_metrics.png"
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Gráfico de métricas por clase guardado en: {metrics_path}")

def main():
    try:
        # 1. Carga y preparación de datos
        X, y, label_encoder = load_and_prepare_data(FEATURES_PATH)
        logger.info(f"Normalizador guardado en: {NORMALIZER_PATH}")
        
        # 2. División train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # 3. Construcción del modelo
        model = build_model(X.shape[1], len(label_encoder.classes_))
        model.summary(print_fn=lambda x: logger.info(x))
        
        # 4. Entrenamiento con early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1
        )
        
        # 5. Evaluación detallada
        metrics = evaluate_model(model, X_test, y_test, label_encoder)
        
        # 6. Guardado del modelo
        model.save(MODEL_PATH)
        joblib.dump(label_encoder, ENCODER_PATH)
        logger.info(f"✅ Modelo guardado en: {MODEL_PATH}")
        logger.info(f"✅ Encoder guardado en: {ENCODER_PATH}")
        
        return metrics

    except Exception as e:
        logger.error(f"❌ Error durante el entrenamiento: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
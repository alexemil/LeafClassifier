import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rutas
FEATURES_PATH = Path("data/processed/leaf_features.csv")
MODEL_PATH = Path("models/leaf_classifier.pkl")
PLOTS_DIR = Path("reports/figures/")

def load_data(features_path: Path) -> tuple:
    """Carga y prepara los datos."""
    try:
        df = pd.read_csv(features_path)
        
        # Verifica columnas esenciales
        required_cols = ["area", "perimeter", "solidity", "image_name"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"El CSV debe contener las columnas: {required_cols}")
        
        # Codifica etiquetas (si no están numéricas)
        if "species" not in df.columns:
            logger.warning("Columna 'species' no encontrada. Usando 'especie_01' como default.")
            df["species"] = "especie_01"
        
        le = LabelEncoder()
        df["species_encoded"] = le.fit_transform(df["species"])
        
        return df, le
    except Exception as e:
        logger.error(f"Error al cargar datos: {e}")
        raise

def train_model(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """Entrena un modelo con mejor manejo de clases desbalanceadas"""
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced', None]  # Nuevo parámetro
    }
    
    model = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=StratifiedKFold(n_splits=5),  # Validación cruzada estratificada
        scoring='f1_weighted',  # Mejor métrica para clases desbalanceadas
        n_jobs=-1
    )
    model.fit(X, y)
    return model.best_estimator_

def save_artifacts(model, label_encoder: LabelEncoder, plots_dir: Path, y_test, y_pred) -> None:
    """Guarda modelo, encoder y gráficos."""
    try:
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Resto del código (indentado correctamente)
        joblib.dump(model, MODEL_PATH)
        # ... (todo lo demás)
    except Exception as e:
        logger.error(f"Error al guardar artefactos: {e}")
        raise
    
    # Guarda modelo y encoder
    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoder, Path("models/label_encoder.pkl"))
    
    # Gráfico de importancia de características
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=model.feature_importances_,
        y=model.feature_names_in_
    )
    plt.title("Importancia de Características")
    plt.savefig(plots_dir / "feature_importance.png")
    plt.close()

        # Matriz de confusión
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title("Matriz de Confusión")
    plt.savefig(plots_dir / "confusion_matrix.png")
    plt.close()

def main():
    try:
        # Carga y prepara datos
        df, label_encoder = load_data(FEATURES_PATH)
        X = df[["area", "perimeter", "solidity"]]
        y = df["species_encoded"]
        
        # Divide datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Entrena y evalúa
        model = train_model(X_train, y_train)
        y_pred = model.predict(X_test)
        save_artifacts(model, label_encoder, PLOTS_DIR, y_test, y_pred)
        
        # Nuevas métricas
        logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        logger.info(f"F1-score (weighted): {f1_score(y_test, y_pred, average='weighted'):.2f}")
        logger.info("\n" + classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        
        # Guarda artefactos
        logger.info(f"✅ Modelo guardado en {MODEL_PATH}")
        
    except Exception as e:
        logger.error(f"Error en el pipeline: {e}")

if __name__ == "__main__":
    main()

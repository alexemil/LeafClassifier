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

# Agrega esto al inicio del archivo
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Subimos dos niveles desde src/classifiers/
FEATURES_PATH = PROJECT_ROOT / "data/processed/leaf_features.csv"
MODEL_PATH = PROJECT_ROOT / "models/leaf_classifier.pkl"
PLOTS_DIR = PROJECT_ROOT / "reports/figures"

def load_data(features_path: Path) -> tuple:
    """Carga y prepara los datos con validaciones mejoradas.
    
    Args:
        features_path: Ruta al archivo CSV con features
        
    Returns:
        tuple: (DataFrame con features, LabelEncoder ajustado)
        
    Raises:
        ValueError: Si hay problemas con los datos
    """
    try:
        # Cargar datos con verificación de existencia
        if not features_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {features_path}")
            
        df = pd.read_csv(features_path)
        
        # Columnas requeridas (actualizadas con nuevas features)
        required_cols = [
            "area", 
            "perimeter", 
            "solidity",
            "aspect_ratio",
            "rectangularity",
            "hu_moment1",
            "hu_moment2",
            "texture_mean",
            "species"  # Ahora requerimos explícitamente la columna de especies
        ]
        
        # Verificar columnas faltantes
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Faltan columnas esenciales: {missing_cols}\n"
                f"Columnas encontradas: {df.columns.tolist()}"
            )
        
        # Manejo de valores nulos
        if df.isnull().any().any():
            null_counts = df.isnull().sum()
            logger.warning(
                f"Valores nulos encontrados:\n{null_counts[null_counts > 0]}\n"
                "Rellenando con medianas por especie."
            )
            
            # Rellenar nulos con la mediana por especie
            for col in df.select_dtypes(include=np.number).columns:
                df[col] = df.groupby('species')[col].transform(
                    lambda x: x.fillna(x.median())
                )
        
        # Validación de distribución de clases
        species_counts = df['species'].value_counts()
        logger.info(f"Distribución de especies:\n{species_counts}")
        
        if len(species_counts) < 2:
            raise ValueError(
                "Se requiere al menos 2 especies distintas. "
                f"Encontradas: {species_counts.index.tolist()}"
            )
        
        # Codificación de etiquetas
        le = LabelEncoder()
        df["species_encoded"] = le.fit_transform(df["species"])
        
        # Normalización de características (excepto species_encoded)
        numeric_cols = [col for col in df.columns if col not in ['species', 'species_encoded', 'image_name']]
        df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
        
        # Validación final
        if df.empty:
            raise ValueError("El DataFrame resultante está vacío después del procesamiento")
            
        return df, le
        
    except Exception as e:
        logger.error(f"Error crítico al cargar datos: {str(e)}")
        raise ValueError(f"No se pudieron cargar los datos: {str(e)}") from e

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
        
        report_dict = classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            output_dict=True  # <- Esto es lo que lo transforma a diccionario
        )
        
        # Lo conviertes en un DataFrame y lo guardas
        report_df = pd.DataFrame(report_dict).transpose()
        
        # Opcional: redondear los valores
        report_df[["precision", "recall", "f1-score"]] = report_df[["precision", "recall", "f1-score"]].round(2)
        
        # Guardarlo como CSV
        report_df.to_csv(PLOTS_DIR / "classification_report.csv")
        
        # Mostrar en consola
        print("\n=== Clasificación por clase ===")
        print(report_df)
        # Guarda artefactos
        logger.info(f"✅ Modelo guardado en {MODEL_PATH}")
        
    except Exception as e:
        logger.error(f"Error en el pipeline: {e}")

if __name__ == "__main__":
    main()

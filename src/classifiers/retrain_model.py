import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pathlib import Path

def load_data():
    """Cargar datos de features desde archivos"""
    features = []
    labels = []
    
    # Aquí deberías cargar tus datos reales
    # Ejemplo ficticio:
    for especie in ["especie_01", "especie_02", "especie_03"]:
        for i in range(100):  # 100 muestras por especie
            features.append([
                np.random.normal(100, 20),  # area
                np.random.normal(50, 10),   # perimeter
                np.random.normal(0.8, 0.1), # solidity
                np.random.normal(1.5, 0.3), # aspect_ratio
                np.random.normal(0.7, 0.1)  # rectangularity
            ])
            labels.append(especie)
    
    return np.array(features), np.array(labels)

def train_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    
    # Modelo con balance de clases
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluación
    print(classification_report(y_test, model.predict(X_test)))
    
    # Guardar modelo
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model, models_dir / "leaf_classifier.pkl")
    
if __name__ == "__main__":
    train_model()
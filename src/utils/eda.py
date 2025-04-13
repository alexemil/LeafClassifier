import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def run_eda(features_path: Path, plots_dir: Path) -> None:
    """Genera gráficos exploratorios y estadísticas."""
    if not features_path.exists():
        raise FileNotFoundError(f"El archivo {features_path} no existe.")
    df = pd.read_csv(features_path)
    
    # 1. Gráfico de distribución de características
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(["area", "perimeter", "solidity"], 1):
        plt.subplot(2, 2, i)
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribución de {col}")
    plt.tight_layout()
    plt.savefig(plots_dir / "distribuciones.png")
    plt.close()
    
    # 2. Matriz de correlación
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[["area", "perimeter", "solidity"]].corr(), annot=True, cmap="coolwarm")
    plt.title("Matriz de Correlación")
    plt.savefig(plots_dir / "correlacion.png")
    plt.close()

    # 3. Boxplots por especie (si aplica)
    if "species" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="species", y="area")
        plt.title("Área por Especie")
        plt.savefig(plots_dir / "area_por_especie.png")
        plt.close()

if __name__ == "__main__":
    plots_dir = Path("reports/figures")
    plots_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(plots_dir, os.W_OK):
        raise PermissionError(f"No tienes permisos de escritura en {plots_dir}.")
    plots_dir = Path("reports/figures")
    plots_dir.mkdir(exist_ok=True)
    
    features_path = Path("data/features.csv")  # Define the path to the features file
    run_eda(features_path, plots_dir)
    print("✅ Análisis guardado en 'reports/figures/'")

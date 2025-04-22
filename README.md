# 🌿 LeafClassifier - Clasificador de Hojas Nativas

**LeafClassifier** es una herramienta basada en inteligencia artificial para la **clasificación automática de hojas de árboles nativos de Tamaulipas**, desarrollada en Python con una interfaz gráfica en PyQt6 y modelos de aprendizaje automático entrenados con características morfológicas y de textura.

---

## 🚀 Características Principales

- 📸 Carga y visualización de imágenes de hojas.
- ⚙️ Preprocesamiento automático: segmentación, contorno, extracción de características.
- 🧠 Clasificación en una de las especies conocidas mediante un modelo entrenado con TensorFlow.
- 📉 Umbral de confianza para evitar clasificaciones incorrectas.
- 🖥️ Interfaz gráfica intuitiva con resultados en tiempo real.

---

## 📦 Estructura del Proyecto

LeafClassifier/ │ ├── data/ │ ├── raw/ # Imágenes originales │ └── processed/ # Imágenes procesadas por el sistema │ ├── models/ # Modelos entrenados (ej. modelo_tfl.h5) │ ├── src/ │ ├── features/ │ │ └── shape_features.py # Extracción de características │ ├── classifiers/ │ │ └── train_tensorflow_model.py # Entrenamiento del modelo │ └── main.py # Interfaz gráfica PyQt6 │ ├── image_utils.py # Funciones de preprocesamiento ├── requirements.txt # Requisitos del proyecto └── README.md


---

## 💻 Instalación

Requisitos: **Python 3.10 o superior** 

ATENCIÓN! Las imagenes del Dataset que se ocupo para desarrollar esta practica no estarán en el proyecto, para mejor comodidad ingresa al siguiente Enlace:

Enlace: https://drive.google.com/drive/u/0/folders/1OmdU1bjInQ3KxQaxz0eUymSVZuGNjNyp

Especies por nombre cientifico:

    "especie_01": "Aliso blanco",
    "especie_02": "Olmo europeo",
    "especie_03": "Sauce gris",
    "Cordia boissieri": "Anacahuita",
    "Quercus virginiana": "Encino siempre verde",
    "Vachellia farnesiana": "Huizache"

### 1. Clonar el repositorio

```bash
git clone https://github.com/alexemil/LeafClassifier.git
cd LeafClassifier

pip install -r requirements.txt

▶️ ¿Cómo ejecutar el proyecto?

Asegúrate de que las imágenes estén organizadas así:

    data/raw/: Imágenes originales.

    data/processed/: Se llenará automáticamente tras el preprocesamiento.

Paso 1: Extraer características morfológicas

python src/features/shape_features.py

Este script analiza las hojas y guarda sus características en un archivo .csv.
Paso 2: Entrenar el modelo de clasificación

python src/classifiers/train_tensorflow_model.py

Se entrenará un modelo con TensorFlow que se guardará en la carpeta models/.
Paso 3: Ejecutar la interfaz gráfica

python src/main.py

Desde la GUI podrás:

    Cargar una imagen.

    Ver la hoja segmentada y sus contornos.

    Obtener la predicción de la especie.

    Consultar los valores de las características extraídas.

Poner los nombres comunes
Revisar porque el codigo esta tardando tanto
Revisar el modelo

#### Nuevos cambios en el repositorio

-- Cambie los nombres de las plantas ejemplares por los nombres comunes

### Revisar los siguientes archivos

- shape_features.py
- train_tensorflow_model.py
- main.py

### Cambios significativos

1. Se cambio el nombre de las hojas por el nombre cientifico
2. Las metricaz o las imagenes de las graficas se actualizaron de acuerdo a los nombres de las hojas por el nombre común
3. En el main se actualizo la interfaz por una más detallada, se agregaron los valores en una tabla, se hicieron mejoras y ahora se ven los nombres de las por los nombres comunes.
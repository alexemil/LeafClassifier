# LeafClassifier
Enlace a las imagenes que se ocuparon para realizar este proyecto se pueden encontrar en el siguiente enlace:

https://drive.google.com/drive/folders/1OmdU1bjInQ3KxQaxz0eUymSVZuGNjNyp?usp=sharing

Las imagenes originales se tomaron de la sigueinte pagina:
https://www.cvl.isy.liu.se/en/research/datasets/swedish-leaf/

La estructura del proyecto solo tendrá los archivos de codigo fuente y los modelos. 

Se deberá crear la estructura original para la implementación y la funcionalidad del proyecto:

.
├── docs
├── LICENSE
├── models
│   ├── label_encoder.pkl
│   └── leaf_classifier.pkl
├── README.md
├── reports
│   └── figures
│       ├── confusion_matrix.png
│       └── feature_importance.png
├── requirements.txt
└── src
    ├── classifiers
    │   ├── random_forest.py
    │   ├── svm.py
    │   └── train_model.py
    ├── features
    │   ├── color_histogram.py
    │   └── shape_features.py
    ├── main.py
    ├── tests
    │   ├── test_classifier.py
    │   └── test_contours.py
    └── utils
        ├── convert_tif_to_png.py
        ├── eda.py
        ├── image_loader.py
        └── logger.py

Hace falta las carpeta

data:
 - processed
 - raw

Aquí deben de estar las imagenes, en el raw deben de estar las imagenes originales, en el processed ahí se arroja imaganes creadas por un archivo aparte.

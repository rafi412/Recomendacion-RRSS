# TFM - Sistema de Recomendación de Contenido Textual y Visual

Este proyecto desarrolla un sistema de recomendación basado en contenido capaz de sugerir publicaciones e imágenes a partir de los intereses de un usuario.

El trabajo comienza con un enfoque clásico de clasificación textual mediante `TF-IDF + LogisticRegression`, evoluciona hacia recomendaciones semánticas utilizando embeddings y similitud coseno, incorpora un sistema de clasificación visual mediante Transfer Learning y finaliza con una integración de ambos enfoques en un sistema de recomendación completo.

El objetivo principal no es únicamente comparar modelos, sino analizar cómo distintas técnicas de Inteligencia Artificial pueden utilizarse para representar, clasificar y recomendar contenido en escenarios con ambigüedad, ruido y señales mixtas entre categorías.

Enlaces con datos y modelos necesarios:
modelos: https://drive.google.com/drive/folders/1lOIttVFrqVYnPeaoNt7iVGgWQ5zVHX2N?usp=drive_link
datos: https://drive.google.com/drive/folders/1kg7aF3m1hMnaPOwWuZDPcCE2H-KAss4r?usp=drive_link

## Tecnologías utilizadas

* Python
* Jupyter Notebook
* Pandas y NumPy
* Scikit-learn
* Matplotlib
* SentenceTransformers
* PyTorch
* Torchvision
* Joblib

## Estructura general

* `01_clasificacion_textual_supervisada.ipynb`
* `02_recomendacion_embeddings.ipynb`
* `03_comparacion_tfidf_embeddings.ipynb`
* `04_transfer_learning_vision.ipynb`
* `05_sistema_recomendacion_final.ipynb`

Datos y modelos:

* `data/posts.csv`
* `data/imagenes/`
* `pipeline_textual.pkl`
* `models/`
* `artifacts/`

## Qué hace cada notebook

### 1. Clasificación textual supervisada

Construye el baseline del proyecto utilizando TF-IDF y Regresión Logística.

Funciones principales:

* Análisis inicial del dataset.
* Vectorización mediante TF-IDF.
* Entrenamiento del clasificador.
* Evaluación mediante métricas y matriz de confusión.
* Exportación del modelo entrenado.

---

### 2. Recomendación mediante embeddings

Transforma el problema de clasificación en un sistema de recuperación semántica.

Funciones principales:

* Generación de embeddings para cada publicación.
* Construcción de perfiles vectoriales.
* Recuperación mediante similitud coseno.
* Evaluación de recomendaciones.

---

### 3. Comparativa TF-IDF vs Embeddings

Compara ambos enfoques utilizando consultas y escenarios similares.

Aspectos analizados:

* Capacidad de clasificación.
* Recuperación semántica.
* Robustez ante ambigüedad.
* Diferencias entre clasificación y recomendación.
* Visualización e interpretación de resultados.

---

### 4. Sistema visual mediante Transfer Learning

Implementa un clasificador de imágenes utilizando un modelo preentrenado y fine-tuning de la última capa.

Funciones principales:

* Preparación del dataset visual.
* Entrenamiento del modelo base.
* Fine-tuning.
* Comparación entre modelo original y ajustado.
* Evaluación mediante accuracy y matrices de confusión.

---

### 5. Sistema final de recomendación

Integra los componentes desarrollados durante el proyecto.

Funciones principales:

* Generación de perfiles de usuario simulados.
* Recomendación de publicaciones.
* Recomendación de imágenes.
* Evaluación conjunta del sistema.
* Ejemplos prácticos de uso.

## Flujo general del proyecto

1. Construcción y análisis de datasets textuales y visuales.
2. Entrenamiento del modelo supervisado basado en TF-IDF.
3. Implementación del sistema semántico basado en embeddings.
4. Comparación de ambos enfoques.
5. Entrenamiento del modelo visual mediante Transfer Learning.
6. Integración de texto e imagen en un sistema de recomendación final.

## Ejecución recomendada

Los notebooks deben ejecutarse en el siguiente orden:

1. `01_clasificacion_textual_supervisada.ipynb`
2. `02_recomendacion_embeddings.ipynb`
3. `03_comparacion_tfidf_embeddings.ipynb`
4. `04_transfer_learning_vision.ipynb`
5. `05_sistema_recomendacion_final.ipynb`

Instalar previamente las dependencias incluidas en los archivos de requisitos correspondientes.

## Objetivo del proyecto

El objetivo de este TFM es estudiar la evolución desde técnicas tradicionales de clasificación textual hacia sistemas de recomendación más avanzados basados en representaciones semánticas y visión por computador.

El proyecto busca demostrar cómo diferentes técnicas de Inteligencia Artificial pueden combinarse para recomendar contenido textual y visual de forma coherente, utilizando perfiles de usuario simulados y conjuntos de datos específicos para cada modalidad.

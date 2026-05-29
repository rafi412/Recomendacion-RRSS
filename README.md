# TFM - Recomendación de Contenido Textual con TF-IDF y Embeddings

Este proyecto estudia un sistema de recomendación basado en contenido para publicaciones textuales de estilo red social. El trabajo parte de un baseline supervisado con `TF-IDF + LogisticRegression` y evoluciona hacia un enfoque semántico con embeddings y similitud coseno.

La idea principal del TFM no es solo comparar modelos, sino observar cómo cambia el comportamiento del sistema cuando el texto deja de ser limpio y aparece ambigüedad, ruido, vocabulario compartido y señales mixtas entre categorías.

## Tecnologías utilizadas

- `Python`
- `Jupyter Notebook`
- `pandas` y `numpy`
- `scikit-learn`
- `matplotlib`
- `sentence-transformers`
- `joblib`

## Estructura general

- [01_clasificacion_textual_supervisada.ipynb]: construye el baseline supervisado del proyecto.
- [02_recomendacion_embeddings.ipynb]: transforma el problema hacia recomendación semántica con embeddings.
- [03_comparacion_modelos_embeddings.ipynb]: compara ambos enfoques con ejemplos reales y visualizaciones interpretables.
- [data/posts.csv]: dataset principal del proyecto.
- [pipeline_textual.pkl]: pipeline entrenado del modelo TF-IDF.
- `models/`: artefactos del modelo de embeddings multilingüe y comparativas locales.
- `artifacts/`: salidas auxiliares generadas durante los notebooks.

## Qué hace cada notebook

### 1. Clasificación textual supervisada

El notebook `01` entrena un clasificador basado en `TF-IDF + LogisticRegression`. Su función es servir como baseline interpretable y mostrar hasta dónde puede llegar un enfoque léxico clásico cuando el dataset contiene ruido y ambigüedad.

De forma general:

- carga y revisa el dataset,
- prepara el texto,
- entrena el pipeline supervisado,
- evalúa el rendimiento del baseline,
- y guarda el pipeline para reutilizarlo después.

### 2. Recomendación con embeddings

El notebook `02` cambia el enfoque desde clasificación hacia recuperación semántica. En vez de predecir una única etiqueta, representa cada publicación como un vector usando `SentenceTransformers` y recupera contenido parecido mediante similitud coseno. El modelo activo del proyecto es `paraphrase-multilingual-MiniLM-L12-v2`, más adecuado para un corpus y unas consultas en español.

De forma general:

- prepara el texto para embeddings,
- genera vectores semánticos,
- construye rankings de recomendaciones,
- y guarda los artefactos necesarios para reutilizarlos en la comparación final.

### 3. Comparación práctica entre enfoques

El notebook `03` no vuelve a entrenar modelos. Reutiliza los artefactos anteriores para comparar cómo responden `TF-IDF` y embeddings ante consultas reales, especialmente cuando las frases son ambiguas o mezclan varias intenciones.

La comparación se centra en:

- confianza e incertidumbre de `TF-IDF`,
- recuperación semántica de embeddings,
- diferencias prácticas entre clasificación y ranking,
- y visualizaciones simples para interpretar el comportamiento de ambos enfoques.

## Funcionamiento resumido

El flujo del proyecto puede entenderse así:

1. Se entrena un baseline supervisado con `TF-IDF`.
2. Se observa que clasificar una única categoría no siempre captura bien el significado del texto.
3. Se pasa a embeddings para modelar cercanía semántica entre publicaciones.
4. Finalmente se comparan ambos enfoques con ejemplos y métricas visuales más interpretables que un simple acierto binario.

## Ejecución orientativa

El orden recomendado de ejecución es:

1. `01_clasificacion_textual_supervisada.ipynb`
2. `02_recomendacion_embeddings.ipynb`
3. `03_comparacion_modelos_embeddings.ipynb`

Para la parte de embeddings puede ser necesario instalar las dependencias listadas en [requirements_embeddings.txt].

## Objetivo del proyecto

Este TFM busca mostrar una evolución metodológica sencilla pero realista:

- empezar con un baseline claro,
- detectar sus límites,
- incorporar una representación semántica más flexible,
- y comparar ambos enfoques con una lectura práctica y comprensible.

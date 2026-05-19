# Narrativa técnica del TFM

## 1. Problema de partida

El proyecto estudia un sistema de recomendación basado en contenido textual para publicaciones tipo red social. La evolución metodológica parte de un baseline supervisado interpretable y avanza hacia un recomendador semántico dinámico:

1. Clasificación supervisada con `TF-IDF + LogisticRegression`.
2. Representación semántica con `SentenceTransformers`.
3. Recomendación por similitud coseno.
4. Perfil vectorial de usuario construido a partir de likes.
5. Re-ranking con diversidad y feedback negativo.

## 2. Hallazgo metodológico

La primera versión del dataset era demasiado limpia: cada categoría tenía vocabulario muy distintivo, pocos textos ambiguos y patrones de hashtags excesivamente alineados con la etiqueta. Como consecuencia, los modelos parecían mejores de lo que realmente eran.

Ese comportamiento no demostraba robustez del sistema, sino una forma de sobreajuste conceptual al diseño del dato: el problema estaba construido de forma demasiado fácil.

## 3. Regeneración del dataset

El nuevo dataset introduce ruido y solapamiento de forma controlada:

- 11.000 publicaciones sintéticas.
- Textos cortos, medios, largos y outliers.
- Publicaciones ambiguas con categoría secundaria.
- Vocabulario compartido entre categorías.
- Hashtags cruzados y hashtags no relacionados.
- Entidades reales como artistas, marcas, ciudades, videojuegos, equipos y comida.
- Errores leves, slang, emojis, frases cortadas y textos pobres.
- Pequeña proporción de etiquetas ruidosas.

El objetivo no es aumentar datos, sino aumentar la calidad experimental del problema.

## 4. Lectura experimental

Con el nuevo dataset, TF-IDF sigue siendo un baseline útil, pero deja de ser perfecto. Los errores se concentran especialmente en:

- publicaciones ambiguas,
- textos muy cortos,
- etiquetas ruidosas,
- entidades compartidas,
- pares naturalmente solapados como playa-viajes, coches-viajes o videojuegos-música.

Esto justifica el paso a embeddings: no porque TF-IDF sea inútil, sino porque su representación léxica tiene límites claros cuando el significado no coincide exactamente con palabras exclusivas.

## 5. Paso a embeddings

Los embeddings permiten comparar publicaciones por proximidad semántica. Esto encaja mejor con recomendación basada en contenido porque el sistema no necesita predecir una única etiqueta: puede ordenar elementos según su cercanía a un perfil de interés.

Sin embargo, los embeddings introducen nuevos problemas:

- rankings muy concentrados en una misma zona semántica,
- dominancia de categorías muy próximas,
- recuperación de publicaciones ambiguas difíciles de evaluar con una sola etiqueta,
- sensibilidad a perfiles de usuario pobres o demasiado estrechos.

## 6. Sistema dinámico

El perfil de usuario se construye como la media de los embeddings de publicaciones likeadas. Esta estrategia es simple, explicable y coherente con el alcance del TFM, pero tiene trade-offs:

- maximiza similitud, pero puede reducir diversidad;
- mejora afinidad inmediata, pero puede crear redundancia;
- acepta feedback negativo sin reentrenar, pero desplaza el perfil de forma aproximada.

Por eso se introduce un re-ranking con diversidad y una simulación de dislikes. La mejora no se presenta como solución definitiva, sino como una intervención controlada para comparar precisión, diversidad, cobertura y score medio.

## 7. Conclusión narrativa

El TFM debe leerse como un proceso real de refinamiento:

1. Se construye un baseline.
2. Se detecta que el dataset original era demasiado limpio.
3. Se regenera el corpus para introducir dificultad real.
4. TF-IDF muestra limitaciones interpretables.
5. Los embeddings mejoran la recuperación semántica.
6. La recomendación por similitud revela problemas de concentración.
7. La diversidad y el feedback negativo permiten estudiar trade-offs reales.

La aportación principal no es una arquitectura compleja, sino un diseño experimental más honesto: datos más realistas, análisis más profundo y limitaciones visibles.

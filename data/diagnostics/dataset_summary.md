# Diagnostico automatico del dataset

- Registros: 11000
- Categorias: 8
- Posts ambiguos: 42.44%
- Etiquetas ruidosas simuladas: 3.60%
- Descripciones ausentes: 2.83%
- Hashtags ausentes: 17.95%
- Longitud mediana: 17.0 palabras
- Vocabulario TF-IDF de diagnostico: 7480
- Similitud intra-categoria media: 0.0446
- Similitud inter-categoria media: 0.0211
- Brecha intra-inter: 0.0236

## Categorias potencialmente demasiado separadas

| categoria | similitud_intra_media | categoria_inter_mas_cercana | similitud_inter_mas_cercana | brecha | diagnostico |
| --- | --- | --- | --- | --- | --- |
| videojuegos | 0.0568 | musica | 0.0236 | 0.0332 | solapada |
| animales | 0.0498 | playa | 0.0277 | 0.0221 | solapada |
| playa | 0.0475 | animales | 0.0277 | 0.0198 | solapada |
| coches | 0.0449 | viajes | 0.0257 | 0.0192 | solapada |
| cocina | 0.0419 | viajes | 0.0242 | 0.0177 | solapada |
| deporte | 0.0407 | coches | 0.0235 | 0.0172 | solapada |
| musica | 0.0375 | playa | 0.0241 | 0.0134 | solapada |
| viajes | 0.0379 | coches | 0.0257 | 0.0122 | solapada |

## Pares inter-categoria mas cercanos

| categoria_a | categoria_b | similitud_inter_media | similitud_inter_p90 |
| --- | --- | --- | --- |
| animales | playa | 0.0277 | 0.0690 |
| coches | viajes | 0.0257 | 0.0614 |
| cocina | viajes | 0.0242 | 0.0683 |
| musica | playa | 0.0241 | 0.0610 |
| musica | viajes | 0.0236 | 0.0595 |
| musica | videojuegos | 0.0236 | 0.0560 |
| coches | deporte | 0.0235 | 0.0580 |
| cocina | deporte | 0.0235 | 0.0608 |
| coches | playa | 0.0226 | 0.0576 |
| playa | viajes | 0.0225 | 0.0503 |
| cocina | playa | 0.0220 | 0.0515 |
| deporte | musica | 0.0210 | 0.0512 |

## Palabras mas frecuentes

| palabra | frecuencia |
| --- | --- |
| plan | 2129 |
| hoy | 2002 |
| random | 1765 |
| despues | 1688 |
| fondo | 1166 |
| cero | 1089 |
| literalmente | 1075 |
| top | 1073 |
| playlist | 1070 |
| demasiado | 1026 |
| jaja | 1004 |
| raro | 961 |
| musica | 956 |
| update | 887 |
| domingo | 878 |
| cafe | 843 |
| playa | 838 |
| paseo | 828 |
| asi | 826 |
| ruta | 820 |
| bro | 806 |
| decir | 793 |
| modo | 792 |
| renta | 788 |
| luego | 755 |
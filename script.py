import pandas as pd
import numpy as np
import random
import string
from faker import Faker

# Inicializar Faker con localización en español
fake = Faker('es_ES')

# Configuración inicial
NUM_RECORDS = 5000
CATEGORIES =['playa', 'coches', 'deporte', 'cocina', 'viajes', 'musica', 'videojuegos']

# Definición de plantillas por categoría (al menos 15 por cada una con jerga realista)
TEMPLATES = {
    'playa':[
        "Tarde perfecta en el chiringuito de {city} con un buen mojito.",
        "Arena entre los dedos y brisa marina, {name} no necesita más.",
        "Pillando olas increíbles, el mar hoy está de locos.",
        "Vuelta y vuelta en la tumbona como un filete.",
        "Nada como el sonido de las olas al atardecer en {city}.",
        "Protección 50 pero sigo rojo como un guiri.",
        "Día de snorkel, el agua está cristalina.",
        "El agua está helada, pero el primer chapuzón resucita a cualquiera.",
        "Castillos de arena nivel arquitecto.",
        "Paseítos por la orilla que te reinician la vida.",
        "No hay verano sin beso ni playa sin arena en el coche.",
        "Apurando los últimos rayos de sol en la costa.",
        "Bikini, gafas de sol y desconexión total.",
        "Lectura bajo la sombrilla, la auténtica salud.",
        "Amanecer en la playa, un espectáculo que no tiene precio."
    ],
    'coches':[
        "Ese motor ruge como una bestia.",
        "Cambiando las llantas, ahora sí que impone respeto.",
        "300 caballos bajo el capó y se notan en cada recta.",
        "Derrape perfecto en la última curva, estilo rally.",
        "Ruta dominguera de curvas por la sierra de {city}.",
        "Lavado a mano para que brille la pintura mate.",
        "Gasolina por las nubes pero sigo dándole caña.",
        "Restaurando un clásico, pieza a pieza en el garaje.",
        "El sonido del escape me pone los pelos de punta.",
        "Apurando frenadas como si no hubiera un mañana.",
        "Interior de cuero y pantalla táctil, parece una nave espacial.",
        "Cambio manual como Dios manda, los automáticos me aburren.",
        "Pasando la ITV por los pelos, vaya nervios hemos pasado.",
        "Drifting en circuito cerrado, pura adrenalina.",
        "Un depósito lleno y la carretera de {city} abierta ante mí."
    ],
    'deporte':[
        "Menudas agujetas tengo hoy de la paliza del gimnasio.",
        "Marcando un hat-trick espectacular en el partido de hoy.",
        "Preparando la maratón de {city}, ya van 20 kilómetros hoy.",
        "Toca sudar la camiseta, no hay dolor, no hay ganancia.",
        "Récord personal en peso muerto levantando hierro.",
        "El árbitro nos ha robado el partido claramente, menuda estafa.",
        "Tirando triples, hoy entran todas en la cancha.",
        "Día de piernas, mañana no voy a poder ni caminar.",
        "Remontada épica en el último minuto del descuento.",
        "Cardio en ayunas para empezar el día con energía a tope.",
        "Comiendo arroz y pollo, modo volumen activado.",
        "Partidillo de pádel, a ver quién paga las cañas hoy.",
        "Estirando bien para evitar lesiones, que ya no somos unos niños.",
        "Pedaleando por la montaña, menudas cuestas tiene esta ruta.",
        "Una buena rutina y disciplina es el secreto del éxito."
    ],
    'cocina':[
        "Dejando el guiso al chup-chup durante un par de horas.",
        "A punto de emplatar este solomillo nivel estrella Michelin.",
        "Sofreír bien la cebolla es la base de todo plato decente.",
        "Hoy me he currado una paella que ni en {city}.",
        "Probando una receta nueva, a ver si no quemo la cocina.",
        "El toque de cilantro le da una frescura increíble al plato.",
        "Masa madre de tres días, el pan va a quedar espectacular.",
        "Amasando pizza casera con la ayuda de {name}.",
        "Punto de sal perfecto, me ha quedado de rechupete.",
        "Flambear esto me da miedo pero allá voy.",
        "Un buen chorrito de AOVE y solucionado.",
        "Tortilla de patatas, siempre con cebolla, no hay debate posible.",
        "Marinando el pollo para que pille todo el sabor.",
        "Postre casero: tarta de queso que se deshace en la boca.",
        "Fuego lento, paciencia y mucho mimo en los fogones."
    ],
    'viajes':[
        "El jet lag me está matando, pero la ciudad es preciosa.",
        "Escapada de fin de semana en {city} para recargar pilas.",
        "Día intenso de turisteo, me duelen los pies de tanto andar.",
        "Wanderlust puro, ya estoy pensando en el próximo destino.",
        "Perdiéndome por las callejuelas buscando rincones ocultos.",
        "A punto de perder el vuelo por apurar en el Duty Free.",
        "Mochila a la espalda y a recorrer mundo sin rumbo.",
        "El atardecer desde este mirador no necesita filtros.",
        "Comiendo en un puesto callejero, toda una aventura gastronómica.",
        "Haciendo el check-in, por fin empieza la aventura con {name}.",
        "Cinco países en diez días, a tope de ritmo.",
        "Sellando el pasaporte una vez más en la frontera.",
        "Hostal barato y compañeros de cuarto que roncan, un clásico.",
        "Conociendo a gente increíble de todas partes del mundo.",
        "Desconectando en la naturaleza de {country}, ni rastro de cobertura."
    ],
    'musica':[
        "Menudo temazo acaba de sacar {name}, lo tengo en bucle.",
        "En primera fila del concierto, el flow ha sido brutal.",
        "Sacando los acordes con la guitarra acústica nueva.",
        "Pidiendo el bis a gritos, qué energía en el escenario.",
        "Los graves de este altavoz me están haciendo vibrar el pecho.",
        "Descubriendo artistas en la lista de descubrimientos de la semana.",
        "Afinando la batería para el bolo de esta noche en {city}.",
        "Un buen solo de guitarra y me tienen ganado.",
        "Vinilo antiguo sonando en el tocadiscos, magia pura.",
        "Letras que te llegan directas al corazón.",
        "Festival de verano con los colegas, la mejor tradición.",
        "Cambiando las cuerdas al bajo, ya iba siendo hora.",
        "La mezcla del track ha quedado muy limpia, grandes vocales.",
        "El drop de esta canción rompe la pista de baile.",
        "Cantando a pleno pulmón en el coche y desafinando a tope."
    ],
    'videojuegos':[
        "El lag me ha matado justo cuando iba a ganar la partida.",
        "Abriendo loot boxes, a ver si por fin cae la skin legendaria.",
        "Farmear experiencia durante horas para hacer level up.",
        "Han nerfeado mi arma favorita, así no se puede jugar a nada.",
        "Partida clasificatoria tensa, pero hemos carreado bien.",
        "Ese boss de {city} tiene unos patrones de ataque loquísimos.",
        "Construyendo la base perfecta para sobrevivir la noche.",
        "Qué gráficos tiene este AAA, el ray tracing es brutal.",
        "Unos tiritos con los panas por Discord de chill.",
        "Speedrun completado, a punto de hacer nuevo récord mundial.",
        "He ragequiteado de lo lindo, el matchmaking está rotísimo.",
        "Completando las misiones secundarias antes de ir por el final.",
        "Ese combo no hay quien lo esquive, puro hitbox roto.",
        "El modo historia es una obra maestra narrativa espectacular.",
        "Bugeado otra vez, se me ha quedado el personaje atrapado."
    ]
}

# Hashtags por categoría
HASHTAGS = {
    'playa':['#verano', '#sol', '#arena', '#playita', '#relax', '#mar', '#costa'],
    'coches':['#motor', '#racing', '#cars', '#velocidad', '#tuning', '#drifting', '#cargram'],
    'deporte':['#fitness', '#workout', '#gym', '#futbol', '#running', '#maraton', '#entreno'],
    'cocina':['#foodie', '#recetas', '#chef', '#gastronomia', '#instafood', '#casero', '#yummy'],
    'viajes':['#wanderlust', '#travel', '#escapada', '#turismo', '#mochilero', '#viajar', '#jetlag'],
    'musica':['#temazo', '#live', '#concierto', '#music', '#flow', '#festival', '#acustico'],
    'videojuegos':['#gaming', '#gamer', '#esports', '#twitch', '#levelUp', '#streamer', '#setup']
}

def generate_clean_data(num_records):
    """Genera el dataset limpio y equilibrado inicial"""
    data =[]
    for i in range(1, num_records + 1):
        # Seleccionar categoría aleatoriamente
        cat = random.choice(CATEGORIES)
        
        # Generar contenido usando template aleatorio y rellenando variables con Faker
        template = random.choice(TEMPLATES[cat])
        content = template.format(
            city=fake.city(),
            name=fake.first_name(),
            country=fake.country()
        )
        
        # Generar hashtags de su categoría (entre 1 y 4)
        num_tags = random.randint(1, 4)
        tags = " ".join(random.sample(HASHTAGS[cat], num_tags))
        
        data.append({
            'id': i,
            'content': content,
            'hashtags': tags,
            'category': cat
        })
        
    return pd.DataFrame(data)

def apply_data_dirtiness(df):
    """Aplica nulos, outliers y errores de categoría (Data Dirtiness)"""
    n_total = len(df)
    
    # Índices únicos para aplicar las modificaciones en content (para no solapar nulos con outliers)
    all_indices = df.index.tolist()
    random.shuffle(all_indices)
    
    # Cálculos de proporciones
    n_content_nulls = int(n_total * 0.08)       # 8% Content nulo
    n_outliers = int(n_total * 0.02)            # 2% Outliers (basura)
    n_hashtag_nulls = int(n_total * 0.15)       # 15% Hashtags nulos
    
    # 1. Aplicar Outliers en Content (2%)
    outlier_indices = all_indices[:n_outliers]
    for idx in outlier_indices:
        if random.random() < 0.5:
            # Texto sin sentido (tecleo aleatorio)
            df.at[idx, 'content'] = ''.join(random.choices(string.ascii_letters, k=random.randint(15, 60)))
        else:
            # Texto extremadamente largo (2000 palabras)
            df.at[idx, 'content'] = " ".join(fake.words(nb=2000))
            
    # 2. Aplicar Nulos en Content (8%)
    null_content_indices = all_indices[n_outliers:n_outliers + n_content_nulls]
    df.loc[null_content_indices, 'content'] = np.nan
    
    # 3. Aplicar Nulos en Hashtags (15%) - Independiente del content
    hashtag_null_indices = random.sample(df.index.tolist(), n_hashtag_nulls)
    df.loc[hashtag_null_indices, 'hashtags'] = np.nan
    
    # 4. Errores de categoría: 3% de los posts de "deporte" clasificados como "cocina"
    deporte_indices = df[df['category'] == 'deporte'].index.tolist()
    n_errors = int(len(deporte_indices) * 0.03)
    error_indices = random.sample(deporte_indices, n_errors)
    df.loc[error_indices, 'category'] = 'cocina'
    
    return df

if __name__ == "__main__":
    print("Generando datos base realistas...")
    df_clean = generate_clean_data(NUM_RECORDS)
    
    print("Aplicando suciedad a los datos (Nulos, Outliers, Errores de Categoría)...")
    df_dirty = apply_data_dirtiness(df_clean)
    
    # Guardar en CSV
    file_name = 'dataset_posts.csv'
    df_dirty.to_csv(file_name, index=False, encoding='utf-8')
    
    print(f"\n¡Dataset generado con éxito! Guardado en: {file_name}")
    print("-" * 40)
    print("Resumen de las características del CSV:")
    print(f"Total registros: {len(df_dirty)}")
    print(f"Nulos en 'content': {df_dirty['content'].isna().sum()} (8%)")
    print(f"Nulos en 'hashtags': {df_dirty['hashtags'].isna().sum()} (15%)")
    print("\nMuestra de 5 filas aleatorias:")
    print(df_dirty.sample(5))
import pandas as pd
import numpy as np
import random
import string
import unicodedata
from faker import Faker

# Inicializar Faker
fake = Faker('es_ES')

# Configuración
NUM_RECORDS = 5000
CATEGORIES =['playa', 'coches', 'deporte', 'cocina', 'viajes', 'musica', 'videojuegos']

# --- SISTEMA COMBINATORIO PARA MAYOR REALISMO ---
COMPONENTS = {
    'playa': {
        'open':["Madre mía,", "Por fin,", "Qué ganas de esto:", "", "Adivinad quién está en {city}:", "Literalmente", "Modo vacaciones:"],
        'core':["tarde perfecta en el chiringuito con un mojito", "arena entre los dedos y a desconectar de todo", "vuelta y vuelta en la tumbona como un filete", "el agua está helada pero me he bañado igual", "pillando unas olas increíbles", "modo cangrejo activado, me he quemado entero", "postureo máximo en la orilla con {name}", "jugando a las palas nivel profesional"],
        'close':["🌊", "10/10.", "jaja.", "y no necesito más.", "qué paz.", "😎", "a vivir la vida."]
    },
    'coches': {
        'open':["Brum brum,", "Mirad esta joya,", "Increíble,", "", "Hoy toca ruta por {city},", "Enamorado de este bicho:", "Uff,"],
        'core':["el motor V8 rugiendo como una bestia", "pasando la ITV de puro milagro", "cambiando aceite y llantas, ahora sí impone", "derrape guapo en la última rotonda", "la gasolina a precio de oro pero sigo dándole caña", "stage 2 instalada y esto vuela", "interior de cuero espectacular", "lavado a mano para que brille la pintura"],
        'close':["🚗💨", "qué locura.", "me encanta.", "gasss.", "🔥", "puro vicio.", "xd"]
    },
    'deporte': {
        'open':["Día de piernas,", "Hoy toca sufrir,", "Vamos equipo!", "", "Increíble lo de hoy en {city},", "Sin excusas:"],
        'core':["agujetas nivel Dios después del entreno", "marcando un hat-trick para salvar el partido", "nuevo PR en peso muerto, a tope", "el árbitro nos ha robado descaradamente", "corriendo 10km en ayunas, reventado", "comiendo arroz y pollo, modo volumen", "el gymbro me ha fallado hoy", "partidazo de pádel con {name}"],
        'close':["💪", "a tope.", "no pain no gain.", "sudando la gota gorda.", "🥵", "y mañana más.", "victoria!"]
    },
    'cocina': {
        'open':["Marchando,", "Mirad qué pinta,", "Hoy me he coronado:", "", "Experimentando en la cocina:", "Recetón en {city}:"],
        'core':["dejando el guiso al chup-chup horas", "estrenando la airfryer con unas patatas", "paella dominguera espectacular", "un buen chorrito de AOVE y listo", "masa madre fermentando, a ver el pan", "casi quemo la sartén pero ha quedado rico", "tortilla con cebolla siempre", "emplatando nivel estrella Michelin"],
        'close':["👨‍🍳", "brutal.", "ñam ñam.", "espectacular.", "🤤", "a cenar!", "se me hace la boca agua."]
    },
    'viajes': {
        'open':["Próxima parada:", "Escapada express,", "Jet lag activado,", "", "Perdidos por {city},", "Wanderlust:"],
        'core':["turisteando sin parar, me duelen los pies", "perdido en el metro de {country}, ayuda", "pagando un café a precio de turista", "hotelazo con unas vistas que flipas", "mochila a la espalda y a tirar millas", "probando street food de dudosa procedencia", "haciendo el check-in con {name}"],
        'close':["✈️", "wanderlust total.", "me mudo aquí.", "qué maravilla.", "🌍", "no quiero volver.", "😍"]
    },
    'musica': {
        'open':["Temazo,", "En bucle:", "Madre mía el nuevo disco,", "", "En directo desde {city}:", "Pelos de punta,"],
        'core':["en primera fila del concierto, brutal", "los graves de este altavoz me hacen vibrar", "sacando unos acordes con la acústica nueva", "flow espectacular en el escenario", "descubriendo joyitas indies en Spotify", "desafinando en el coche a pleno pulmón", "afinando la batería para el bolo"],
        'close':["🎸", "magia pura.", "🔥", "no puedo parar de escucharlo.", "🎶", "increíble.", "🎵"]
    },
    'videojuegos': {
        'open':["GG,", "Gente,", "Increíble,", "Ragequit inminente:", "", "Literalmente", "Vaya tela:"],
        'core':["el lag me ha arruinado la racha de victorias", "abriendo loot boxes y solo sale humo", "farmeando XP como un tryhard para hacer level up", "han nerfeado a mi main otra vez", "gráficos de locos con el ray tracing", "carrileando a {name} toda la partida", "bugeado en la pared, genial el parche", "ese boss tiene una hitbox rotísima"],
        'close':["🎮", "xd.", "a llorar a la llorería.", "F en el chat.", "🕹️", "menuda estafa.", "10/10 IGN."]
    }
}

# HASHTAGS AMPLIADOS MASIVAMENTE
HASHTAGS = {
    'playa':[
        '#Verano', '#Sol', '#Playa', '#Relax', '#Vacaciones', '#Costa', '#Postu', '#Arena', 
        '#Mar', '#Olas', '#Chiringuito', '#Mojito', '#Atardecer', '#Bikini', '#Bronceado', 
        '#Snorkel', '#Paraiso', '#Desconexion', '#Vistas', '#Calor', '#Isla', '#Brisa'
    ],
    'coches':[
        '#Motor', '#Racing', '#Cars', '#V8', '#Tuning', '#Drifting', '#Gas', '#CochesClasicos', 
        '#MotorSport', '#JDM', '#CarPorn', '#Ruta', '#Velocidad', '#Caballos', '#Llantas', 
        '#Mecanica', '#Turbo', '#V6', '#Exhaust', '#Supercars', '#Drive', '#CarLife'
    ],
    'deporte':[
        '#Fitness', '#Gym', '#Futbol', '#Running', '#PR', '#Crossfit', '#Entreno', '#Workout', 
        '#NoPainNoGain', '#Maraton', '#Padel', '#Baloncesto', '#Fuerza', '#Cardio', '#Salud', 
        '#VidaSana', '#Gimnasio', '#BeastMode', '#Motivacion', '#Atleta', '#Healthy'
    ],
    'cocina':[
        '#Foodie', '#Recetas', '#Chef', '#AOVE', '#Airfryer', '#Casero', '#Yummy', '#Gastronomia', 
        '#InstaFood', '#ComidaSana', '#Postres', '#Gourmet', '#Paella', '#Dulce', '#Salado', 
        '#Cocinando', '#RecetaDelDia', '#Masterchef', '#Sabor', '#Vegano', '#RealFood', '#Tapas'
    ],
    'viajes':[
        '#Wanderlust', '#Travel', '#Turismo', '#Mochileros', '#Trip', '#Viajar', '#Aventura', 
        '#JetLag', '#Escapada', '#Ruta', '#Explorando', '#Viajero', '#Paisaje', '#Naturaleza', 
        '#Destino', '#RoadTrip', '#Vuelos', '#Pasaporte', '#SinFiltros', '#Cultura'
    ],
    'musica':[
        '#Temazo', '#Live', '#Concierto', '#Music', '#Flow', '#Indie', '#Spotify', '#Acustico', 
        '#Festival', '#MusicaEnVivo', '#Guitarra', '#Bateria', '#Cantando', '#Vinilo', '#Rock', 
        '#Pop', '#Urbano', '#DJ', '#CancionDelDia', '#Vibras', '#Ritmo', '#Melodia'
    ],
    'videojuegos':[
        '#Gaming', '#Gamer', '#Esports', '#Twitch', '#LevelUp', '#Tryhard', '#Setup', '#PCGaming', 
        '#PlayStation', '#Xbox', '#Nintendo', '#Streamer', '#Noob', '#ProGamer', '#Loot', 
        '#RPG', '#Shooter', '#GGWP', '#Lag', '#Discord', '#IndieGames', '#RetroGaming'
    ]
}

def remove_accents(input_str):
    """Elimina tildes para simular escritura rápida en internet"""
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

def apply_social_media_noise(text):
    """Degrada el texto para hacerlo parecer escrito por un humano real en RRSS"""
    # 40% de probabilidad de escribir todo en minúsculas
    if random.random() < 0.40:
        text = text.lower()
    
    # 30% de probabilidad de quitar las tildes
    if random.random() < 0.30:
        text = remove_accents(text)
        
    # 15% de probabilidad de añadir vocales repetidas (ej: "holaaa")
    if random.random() < 0.15 and len(text) > 5:
        idx = random.randint(5, len(text)-1)
        if text[idx] in "aeiouAEIOU":
            text = text[:idx] + text[idx] * random.randint(2, 4) + text[idx:]
            
    # 20% de probabilidad de añadir risas o exclamaciones exageradas
    if random.random() < 0.20:
        noise = random.choice([" jajaja", " xd", " lol", " !!!!", "..."])
        text = text.rstrip('.') + noise
        
    return text

def generate_dynamic_content(category):
    """Genera la frase usando el sistema combinatorio"""
    open_ph = random.choice(COMPONENTS[category]['open'])
    core_ph = random.choice(COMPONENTS[category]['core'])
    close_ph = random.choice(COMPONENTS[category]['close'])
    
    # Unir piezas
    raw_text = f"{open_ph} {core_ph} {close_ph}".strip()
    raw_text = " ".join(raw_text.split())
    
    # Rellenar con Faker 
    formatted_text = raw_text.format(
        city=fake.city(),
        country=fake.country(),
        name=fake.first_name()
    )
    
    return apply_social_media_noise(formatted_text)

def generate_realistic_hashtags(category):
    """Genera hashtags con alta variabilidad de formato a partir del pool ampliado"""
    num_tags = random.randint(1, 5) # Subido a 5 posibles hashtags
    tags = random.sample(HASHTAGS[category], num_tags)
    
    # 50% de probabilidad de pasarlos a minúscula (muy habitual)
    if random.random() < 0.5:
        tags = [t.lower() for t in tags]
        
    # 15% de probabilidad de no dejar espacio entre hashtags (#uno#dos)
    separator = "" if random.random() < 0.15 else " "
    
    return separator.join(tags)

def create_base_dataset(num_records):
    data =[]
    for i in range(1, num_records + 1):
        cat = random.choice(CATEGORIES)
        content = generate_dynamic_content(cat)
        tags = generate_realistic_hashtags(cat)
        
        data.append({
            'id': i,
            'content': content,
            'hashtags': tags,
            'category': cat
        })
    return pd.DataFrame(data)

def apply_data_dirtiness(df):
    """Aplica la inyección de defectos: nulos, outliers, errores"""
    n_total = len(df)
    indices = df.index.tolist()
    random.shuffle(indices)
    
    # Cálculos porcentuales
    n_content_nulls = int(n_total * 0.08)       # 8% Content nulo
    n_outliers = int(n_total * 0.02)            # 2% Outliers (basura/anomalías)
    n_hashtag_nulls = int(n_total * 0.15)       # 15% Hashtags nulos
    
    # 1. OUTLIERS (2%)
    outlier_indices = indices[:n_outliers]
    for idx in outlier_indices:
        outlier_type = random.choice(['gibberish', 'extreme_length', 'repeated_word'])
        if outlier_type == 'gibberish':
            df.at[idx, 'content'] = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(20, 100)))
        elif outlier_type == 'extreme_length':
            df.at[idx, 'content'] = " ".join(fake.words(nb=2000))
        else:
            word = fake.word()
            df.at[idx, 'content'] = f"{word} " * 500
            
    # 2. NULOS EN CONTENT (8%)
    null_content_indices = indices[n_outliers:n_outliers + n_content_nulls]
    df.loc[null_content_indices, 'content'] = np.nan
    
    # 3. NULOS EN HASHTAGS (15%)
    hashtag_null_indices = random.sample(df.index.tolist(), n_hashtag_nulls)
    df.loc[hashtag_null_indices, 'hashtags'] = np.nan
    
    # 4. ERRORES DE ETIQUETADO (3% Deporte -> Cocina)
    deporte_indices = df[df['category'] == 'deporte'].index.tolist()
    n_errors = int(len(deporte_indices) * 0.03)
    error_indices = random.sample(deporte_indices, n_errors)
    df.loc[error_indices, 'category'] = 'cocina'
    
    return df

if __name__ == "__main__":
    print("🚀 Generando 5000 registros de datos con combinaciones masivas de hashtags...")
    df_clean = create_base_dataset(NUM_RECORDS)
    
    print("🦠 Inyectando suciedad (Nulos, Outliers, Typos y Errores de Categoría)...")
    df_dirty = apply_data_dirtiness(df_clean)
    
    # Guardado
    file_name = 'posts.csv'
    df_dirty.to_csv(file_name, index=False, encoding='utf-8')
    
    print(f"\n✅ ¡Dataset generado con éxito! Guardado en: {file_name}")
    print("\n🔍 MUESTRA DE DATOS REALISTAS (Observa la variedad de hashtags):")
    # Filtrar solo los que tienen texto y hashtags para la muestra visual
    print(df_dirty.dropna().sample(5)[['content', 'hashtags', 'category']])
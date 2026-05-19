from __future__ import annotations

import argparse
import json
import random
import re
import string
import sys
import unicodedata
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_NUM_RECORDS = 11_000
DEFAULT_OUTPUT_PATH = Path("data/posts.csv")
DEFAULT_REPORT_DIR = Path("data/diagnostics")
DEFAULT_RANDOM_STATE = 42

CATEGORIES = [
    "playa",
    "coches",
    "deporte",
    "cocina",
    "viajes",
    "musica",
    "videojuegos",
    "animales",
]

CATEGORY_WEIGHTS = {
    "playa": 1.08,
    "coches": 0.95,
    "deporte": 1.00,
    "cocina": 1.03,
    "viajes": 1.12,
    "musica": 1.05,
    "videojuegos": 1.00,
    "animales": 0.88,
}

OVERLAP_PAIRS = {
    ("playa", "viajes"): 1.45,
    ("coches", "viajes"): 1.20,
    ("videojuegos", "musica"): 1.10,
    ("cocina", "viajes"): 1.05,
    ("deporte", "musica"): 0.85,
    ("playa", "musica"): 0.85,
    ("cocina", "deporte"): 0.75,
    ("coches", "deporte"): 0.70,
    ("viajes", "musica"): 0.70,
    ("animales", "viajes"): 0.65,
    ("animales", "playa"): 0.50,
}

GENERIC_OPENERS = [
    "",
    "hoy tocaba",
    "no se como explicar esto pero",
    "plan improvisado:",
    "literalmente",
    "me desperte pensando en esto:",
    "domingo raro y",
    "despues de mil vueltas,",
    "nadie me preparo para",
    "si esto no es felicidad entonces nada",
    "mini update:",
    "vibe de hoy:",
]

GENERIC_CONTEXTS = [
    "con gente buena",
    "sin planear demasiado",
    "en modo desconectar",
    "con el movil sin bateria",
    "mientras cae la tarde",
    "con una playlist random de fondo",
    "despues de una semana eterna",
    "entre risas y cero cobertura",
    "con ganas de repetir",
    "sin filtros y con prisa",
    "en plan tranquilo",
    "porque a veces sale bien sin pensarlo",
]

GENERIC_CLOSERS = [
    "",
    "top",
    "god",
    "no pido mas",
    "que locura",
    "muy de hoy",
    "asi si",
    "me renta demasiado",
    "random pero necesario",
    "literalmente obsesionado",
    "ns q decir JAJA",
    "bro esto esta increible",
]

SLANG = [
    "bro",
    "literal",
    "random",
    "full",
    "en bucle",
    "me renta",
    "que fantasía",
    "modo chill",
    "estoy living",
    "se tenia que decir",
    "JAJA",
    "xd",
    "no puedo mas",
]

EMOJIS = [
    "🌊",
    "☀️",
    "🚗",
    "🏁",
    "⚽",
    "🏋️",
    "🍣",
    "🍜",
    "✈️",
    "🧳",
    "🎧",
    "🎸",
    "🎮",
    "🕹️",
    "🐶",
    "🐱",
    "🔥",
    "😂",
    "✨",
    "💀",
]

CITIES = [
    "Madrid",
    "Barcelona",
    "Valencia",
    "Sevilla",
    "Bilbao",
    "Malaga",
    "Cadiz",
    "Ibiza",
    "Tenerife",
    "Gran Canaria",
    "Maspalomas",
    "Tarifa",
    "Roma",
    "Paris",
    "Londres",
    "Lisboa",
    "Berlin",
    "Amsterdam",
    "Tokio",
    "Nueva York",
]

SHARED_HASHTAGS = [
    "#Planazo",
    "#Mood",
    "#Random",
    "#Top",
    "#Domingo",
    "#SinFiltros",
    "#Friends",
    "#Vibes",
    "#Life",
    "#Update",
    "#Hoy",
    "#Chill",
]

UNRELATED_HASHTAGS = [
    "#Lunes",
    "#Oferta",
    "#StudyTok",
    "#Memes",
    "#Trabajo",
    "#Cafe",
    "#GymBro",
    "#Setup",
    "#Aesthetic",
    "#FYP",
]

CATEGORY_DATA = {
    "playa": {
        "entities": ["Ibiza", "Tarifa", "Maspalomas", "Formentera", "Benidorm", "Cádiz", "Tenerife"],
        "topics": [
            "paseo por la costa",
            "baño rapido antes de comer",
            "tarde de chiringuito",
            "calas pequeñas sin cobertura",
            "atardecer en el paseo maritimo",
            "toalla tirada en cualquier sitio",
            "sol flojo pero planazo",
            "brisa salada y cero prisa",
        ],
        "objects": ["mochila", "camara", "sandalias", "protector solar", "altavoz", "helado", "coche", "perro"],
        "hashtags": ["#Playa", "#Costa", "#Verano", "#Sol", "#Calas", "#Mar", "#Relax", "#Atardecer"],
    },
    "coches": {
        "entities": ["BMW", "Audi", "Ferrari", "Tesla", "Seat Leon", "Porsche", "Toyota Supra", "Mercedes"],
        "topics": [
            "ruta corta por carreteras secundarias",
            "lavado rapido antes de salir",
            "parking lleno y aun asi foto",
            "prueba de conduccion con lluvia",
            "gasolina carisima pero habia ganas",
            "interior limpio por primera vez en meses",
            "parada tecnica en mitad del viaje",
            "curvas suaves con musica de fondo",
        ],
        "objects": ["llaves", "playlist", "maleta", "camara", "cafe", "peaje", "gps", "neumaticos"],
        "hashtags": ["#Cars", "#Motor", "#RoadTrip", "#Drive", "#Racing", "#Garaje", "#Ruta", "#CarLife"],
    },
    "deporte": {
        "entities": ["Barça", "Real Madrid", "Betis", "NBA", "Champions", "Decathlon", "Nike", "Adidas"],
        "topics": [
            "partido sufrido hasta el ultimo minuto",
            "entreno corto pero intenso",
            "pachanga despues de clase",
            "paseo largo para despejar",
            "series de carrera con poca fe",
            "dia de pierna y luego sofa",
            "pala nueva para el padel",
            "comida sencilla despues del gimnasio",
        ],
        "objects": ["zapatillas", "botella", "auriculares", "arroz", "pollo", "balon", "reloj", "mochila"],
        "hashtags": ["#Futbol", "#Gym", "#Running", "#Padel", "#Entreno", "#Fitness", "#Partido", "#Workout"],
    },
    "cocina": {
        "entities": ["sushi", "ramen", "tacos", "paella", "tortilla", "Mercadona", "MasterChef", "Dabiz Muñoz"],
        "topics": [
            "cena improvisada con lo que habia",
            "receta vista en TikTok y cero medidas",
            "mesa compartida despues de caminar",
            "airfryer salvando la noche",
            "salsa rara que al final funciona",
            "postre demasiado dulce",
            "taper para mañana si sobrevive",
            "restaurante pequeño encontrado de casualidad",
        ],
        "objects": ["sarten", "cuchillo", "aceite", "arroz", "pan", "tomate", "queso", "cafe"],
        "hashtags": ["#Foodie", "#Receta", "#Cena", "#Cocina", "#Yummy", "#Tapas", "#Casero", "#Sabor"],
    },
    "viajes": {
        "entities": ["Ibiza", "Roma", "Paris", "Lisboa", "Berlin", "Amsterdam", "Tokio", "Ryanair"],
        "topics": [
            "escapada express con maleta pequeña",
            "tren perdido por mirar el movil",
            "hotel sencillo pero bien ubicado",
            "caminata eterna por el centro",
            "mapa abierto y cero orientacion",
            "vuelo temprano con cafe dudoso",
            "road trip sin ruta cerrada",
            "barrio nuevo descubierto por accidente",
        ],
        "objects": ["maleta", "pasaporte", "camara", "billete", "coche", "mochila", "mapa", "auriculares"],
        "hashtags": ["#Travel", "#Viajes", "#Escapada", "#Wanderlust", "#RoadTrip", "#Turismo", "#Aventura", "#Destino"],
    },
    "musica": {
        "entities": ["Taylor Swift", "Bad Bunny", "Rosalia", "Quevedo", "Spotify", "Bizarrap", "Aitana", "Michael Jackson", "The Weeknd"],
        "topics": [
            "tema en bucle desde por la mañana",
            "concierto que empezo tarde pero valio la pena",
            "playlist para conducir sin hablar",
            "guitarra desafinada y aun asi mood",
            "festival con demasiada gente",
            "cancion triste para un plan feliz",
            "altavoz pequeño haciendo milagros",
            "directo que suena mejor de lo esperado",
        ],
        "objects": ["altavoz", "auriculares", "entrada", "micro", "guitarra", "coche", "playlist", "vinilo"],
        "hashtags": ["#Musica", "#Spotify", "#Concierto", "#Temazo", "#Festival", "#Pop", "#Urbano", "#Live"],
    },
    "videojuegos": {
        "entities": ["FIFA", "GTA", "Fortnite", "Minecraft", "Valorant", "PlayStation", "Xbox", "Twitch"],
        "topics": [
            "partida rapida que acabo en tres horas",
            "ranked con lag justo al final",
            "setup medio ordenado para variar",
            "mision secundaria mas interesante que la principal",
            "directo de noche con chat imposible",
            "boss facil hasta que deja de serlo",
            "actualizacion rara pero trae cosas buenas",
            "modo carrera mientras suena musica de fondo",
        ],
        "objects": ["mando", "auriculares", "pantalla", "pizza", "cafe", "playlist", "discord", "silla"],
        "hashtags": ["#Gaming", "#Gamer", "#Twitch", "#PlayStation", "#FIFA", "#GTA", "#Setup", "#GG"],
    },
    "animales": {
        "entities": ["Kiwoko", "veterinario", "Luna", "Toby", "Nala", "Max", "Madrid", "parque"],
        "topics": [
            "paseo largo antes de volver a casa",
            "gato durmiendo encima del teclado",
            "perro mirando la comida con demasiada fe",
            "visita al veterinario sin drama",
            "parque nuevo descubierto de casualidad",
            "siesta compartida despues de comer",
            "ruta pet friendly con parada para cafe",
            "foto borrosa pero momento perfecto",
        ],
        "objects": ["correa", "cama", "pelota", "mochila", "coche", "playa", "cafe", "premio"],
        "hashtags": ["#Mascotas", "#Perros", "#Gatos", "#PetFriendly", "#DogLife", "#CatLife", "#Adopta", "#Paseo"],
    },
}

OVERLAP_TEMPLATES = {
    ("playa", "viajes"): [
        "escapada a {city} con baño rapido, hotel lejos y caminata por la costa",
        "vacaciones con musica, playa y maleta medio rota en {city}",
        "paseo por calas de {city} despues de perder el bus del hotel",
        "viaje de finde para ver atardecer junto al mar en {city}",
    ],
    ("coches", "viajes"): [
        "road trip en {entity_a} por {city} con paradas random y playlist alta",
        "viaje en coche por la costa, gps perdido y cafe de gasolinera",
        "ruta larga con el {entity_a}, maleta atras y cero ganas de llegar",
        "peaje, lluvia y vistas brutales desde el coche camino a {city}",
    ],
    ("videojuegos", "musica"): [
        "jugando al {entity_a} escuchando {entity_b} hasta que se fue el lag",
        "stream de {entity_a} con playlist de {entity_b} y chat imposible",
        "modo carrera en {entity_a} mientras suena {entity_b} de fondo",
        "noche de mando, Discord y tema nuevo de {entity_b}",
    ],
    ("cocina", "viajes"): [
        "probando {entity_a} en {city} despues de caminar sin rumbo",
        "viaje corto y cena de {entity_a} en un sitio que nadie conocia",
        "mercado local, cafe raro y plato de {entity_a} que salvo el dia",
        "ruta por {city} buscando donde comer algo decente",
    ],
    ("deporte", "musica"): [
        "entreno con {entity_b} en los auriculares y cero energia al final",
        "partido del {entity_a} con musica a tope antes de salir",
        "running suave mientras suena {entity_b}, plan raro pero funciona",
        "gym, playlist intensa y camiseta del {entity_a}",
    ],
    ("playa", "musica"): [
        "tarde de playa con {entity_b} sonando en un altavoz pequeño",
        "atardecer en la costa y playlist de {entity_b} en bucle",
        "baño rapido, toalla mojada y temazo de {entity_b}",
        "chiringuito lleno pero la musica estaba increible",
    ],
    ("cocina", "deporte"): [
        "despues del gym cayo {entity_a} porque el cuerpo lo pidio",
        "partido del {entity_b} y cena improvisada en casa",
        "taper de arroz, entreno tarde y cero glamour",
        "proteina, cafe y receta random antes de salir a correr",
    ],
    ("coches", "deporte"): [
        "al partido en {entity_a}, aparcando lejos como siempre",
        "ruta hasta el estadio con camiseta del {entity_b} y lluvia",
        "despues del entreno tocaba lavar el coche",
        "zapatillas en el maletero y plan de ultima hora",
    ],
    ("viajes", "musica"): [
        "vuelo temprano con {entity_b} en bucle y cafe malo",
        "tren a {city}, auriculares puestos y playlist eterna",
        "festival en {city} con maleta pequeña y cero sueño",
        "perdido por {city} siguiendo una cancion de {entity_b}",
    ],
    ("animales", "viajes"): [
        "ruta pet friendly por {city} con parada de cafe",
        "viaje corto con el perro, maleta y demasiadas bolsas",
        "hotel que acepta mascotas y paseo eterno por {city}",
        "escapada improvisada con correa, snacks y cero organizacion",
    ],
    ("animales", "playa"): [
        "paseo por la playa con el perro y media mochila llena de arena",
        "tarde de costa, pelota mojada y foto imposible",
        "baño rapido mientras el perro vigilaba la toalla",
        "atardecer cerca del mar con paseo largo y cero prisa",
    ],
}


def remove_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(character for character in normalized if not unicodedata.combining(character))


def weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    items = list(weights)
    values = [weights[item] for item in items]
    return rng.choices(items, weights=values, k=1)[0]


def weighted_pair_choice(rng: random.Random) -> tuple[str, str]:
    pairs = list(OVERLAP_PAIRS)
    weights = [OVERLAP_PAIRS[pair] for pair in pairs]
    return rng.choices(pairs, weights=weights, k=1)[0]


def choose_length_type(rng: random.Random) -> str:
    return rng.choices(
        ["micro", "short", "medium", "long"],
        weights=[0.16, 0.42, 0.33, 0.09],
        k=1,
    )[0]


def choose_noise_level(rng: random.Random) -> str:
    return rng.choices(
        ["bajo", "medio", "alto"],
        weights=[0.50, 0.34, 0.16],
        k=1,
    )[0]


def pick_category_entity(rng: random.Random, category: str) -> str:
    return rng.choice(CATEGORY_DATA[category]["entities"])


def pick_city(rng: random.Random, fake: Faker) -> str:
    if rng.random() < 0.78:
        return rng.choice(CITIES)
    return fake.city()


def render_category_text(
    rng: random.Random,
    fake: Faker,
    category: str,
    length_type: str,
) -> str:
    data = CATEGORY_DATA[category]
    opener = rng.choice(GENERIC_OPENERS)
    topic = rng.choice(data["topics"])
    context = rng.choice(GENERIC_CONTEXTS)
    entity = pick_category_entity(rng, category)
    city = pick_city(rng, fake)
    object_word = rng.choice(data["objects"])
    closer = rng.choice(GENERIC_CLOSERS)

    if length_type == "micro":
        patterns = [
            "{entity} y {object_word}, {closer}",
            "{topic}, {closer}",
            "{opener} {entity}",
            "{object_word} en {city}. {closer}",
        ]
    elif length_type == "short":
        patterns = [
            "{opener} {topic} {context}",
            "{topic} en {city} con {object_word}",
            "{entity}: {topic} y {closer}",
            "{opener} {topic}, {closer}",
        ]
    elif length_type == "medium":
        patterns = [
            "{opener} {topic} en {city} con {object_word}; {context} y {closer}",
            "{topic} cerca de {city}. {context}, {entity} de fondo y {closer}",
            "{entity} aparecio en el plan: {topic}, {context}, {object_word} incluido",
            "{opener} {topic}; luego {context} y acabamos hablando de {entity}",
        ]
    else:
        patterns = [
            "{opener} {topic} en {city} con {object_word}. Primero parecia un plan normal, luego {context} y acabamos diciendo que {closer}",
            "{topic} en {city}; habia {object_word}, algo de prisa y {entity} saliendo en la conversacion. {context}, bastante real todo",
            "{opener} {topic}. Entre {object_word}, mensajes sin responder y {entity}, el plan quedo raro pero {closer}",
        ]

    text = rng.choice(patterns).format(
        opener=opener,
        topic=topic,
        context=context,
        entity=entity,
        city=city,
        object_word=object_word,
        closer=closer,
    )
    return re.sub(r"\s+", " ", text).strip(" ,;")


def render_overlap_text(
    rng: random.Random,
    fake: Faker,
    primary: str,
    secondary: str,
    length_type: str,
) -> str:
    pair = (primary, secondary)
    if pair not in OVERLAP_TEMPLATES:
        pair = (secondary, primary)
    template = rng.choice(OVERLAP_TEMPLATES[pair])

    entity_a = pick_category_entity(rng, pair[0])
    entity_b = pick_category_entity(rng, pair[1])
    city = pick_city(rng, fake)
    base = template.format(entity_a=entity_a, entity_b=entity_b, city=city)

    if length_type == "micro":
        pieces = base.split(",")
        base = pieces[0]
    elif length_type == "medium":
        base = f"{base}, {rng.choice(GENERIC_CONTEXTS)}"
    elif length_type == "long":
        base = (
            f"{base}. No encaja perfecto en una sola categoria, "
            f"porque mezcla {primary}, {secondary} y un poco de vida real."
        )

    opener = rng.choice(GENERIC_OPENERS)
    closer = rng.choice(GENERIC_CLOSERS)
    text = f"{opener} {base} {closer}".strip()
    return re.sub(r"\s+", " ", text).strip(" ,;")


def inject_typo(rng: random.Random, text: str) -> str:
    words = text.split()
    candidates = [idx for idx, word in enumerate(words) if len(word) >= 5 and word.isalpha()]
    if not candidates:
        return text

    idx = rng.choice(candidates)
    word = words[idx]
    position = rng.randrange(1, len(word) - 1)
    mutation = rng.choice(["drop", "swap", "repeat"])

    if mutation == "drop":
        word = word[:position] + word[position + 1 :]
    elif mutation == "swap" and position < len(word) - 2:
        word = word[:position] + word[position + 1] + word[position] + word[position + 2 :]
    else:
        word = word[:position] + word[position] + word[position:]

    words[idx] = word
    return " ".join(words)


def apply_social_noise(rng: random.Random, text: str, noise_level: str) -> str:
    probabilities = {
        "bajo": {"lower": 0.20, "accent": 0.10, "typo": 0.06, "slang": 0.18, "emoji": 0.22, "cut": 0.03},
        "medio": {"lower": 0.38, "accent": 0.22, "typo": 0.15, "slang": 0.35, "emoji": 0.40, "cut": 0.08},
        "alto": {"lower": 0.55, "accent": 0.35, "typo": 0.28, "slang": 0.55, "emoji": 0.62, "cut": 0.16},
    }[noise_level]

    if rng.random() < probabilities["lower"]:
        text = text.lower()
    if rng.random() < probabilities["accent"]:
        text = remove_accents(text)
    if rng.random() < probabilities["typo"]:
        text = inject_typo(rng, text)
    if rng.random() < probabilities["slang"]:
        insertion = rng.choice(SLANG)
        if rng.random() < 0.5:
            text = f"{insertion} {text}"
        else:
            text = f"{text} {insertion}"
    if rng.random() < probabilities["emoji"]:
        text = f"{text} {rng.choice(EMOJIS)}"
    if rng.random() < probabilities["cut"] and len(text.split()) > 5:
        words = text.split()
        cut_at = rng.randint(3, len(words) - 2)
        text = " ".join(words[:cut_at]) + rng.choice(["...", " y bueno", " no se", " JAJA"])

    if rng.random() < 0.18:
        text = text.rstrip(".") + rng.choice(["!!!", "...", " jajaja", " xd", ""])

    return re.sub(r"\s+", " ", text).strip()


def generate_hashtags(
    rng: random.Random,
    primary: str,
    secondary: str | None,
    noise_level: str,
) -> str | float:
    if rng.random() < 0.17:
        return np.nan

    candidates = []
    candidates.extend(rng.sample(CATEGORY_DATA[primary]["hashtags"], k=rng.randint(1, 2)))

    if secondary and rng.random() < 0.82:
        candidates.extend(rng.sample(CATEGORY_DATA[secondary]["hashtags"], k=rng.randint(1, 2)))

    if rng.random() < 0.55:
        candidates.append(rng.choice(SHARED_HASHTAGS))
    if rng.random() < {"bajo": 0.08, "medio": 0.16, "alto": 0.28}[noise_level]:
        candidates.append(rng.choice(UNRELATED_HASHTAGS))

    candidates = list(dict.fromkeys(candidates))
    rng.shuffle(candidates)
    candidates = candidates[: rng.randint(1, min(5, len(candidates)))]

    if rng.random() < 0.48:
        candidates = [tag.lower() for tag in candidates]

    separator = "" if rng.random() < 0.12 else " "
    return separator.join(candidates)


def detect_entities(text: str, hashtags: str | float) -> str:
    combined = f"{text or ''} {'' if pd.isna(hashtags) else hashtags}".lower()
    found = []
    all_entities = sorted(
        {
            entity
            for category_data in CATEGORY_DATA.values()
            for entity in category_data["entities"]
        },
        key=len,
        reverse=True,
    )
    for entity in all_entities:
        if entity.lower() in combined:
            found.append(entity)
    return " | ".join(dict.fromkeys(found))


def similar_category(rng: random.Random, category: str, secondary: str | None = None) -> str:
    if secondary and rng.random() < 0.70:
        return secondary

    candidates = [
        other
        for pair in OVERLAP_PAIRS
        for other in pair
        if category in pair and other != category
    ]
    if candidates:
        return rng.choice(candidates)
    return rng.choice([candidate for candidate in CATEGORIES if candidate != category])


def generate_outlier_post(rng: random.Random, fake: Faker, post_id: int) -> dict[str, object]:
    primary = weighted_choice(rng, CATEGORY_WEIGHTS)
    outlier_type = rng.choice(["gibberish", "repetition", "spam", "empty_description"])

    if outlier_type == "gibberish":
        description = "".join(rng.choices(string.ascii_letters + string.digits, k=rng.randint(18, 80)))
    elif outlier_type == "repetition":
        token = rng.choice(["top", "viaje", "musica", "playa", "random", "go"])
        description = " ".join([token] * rng.randint(18, 55))
    elif outlier_type == "spam":
        description = f"{fake.sentence(nb_words=6)} link en bio oferta top {rng.choice(SLANG)}"
    else:
        description = np.nan

    hashtags = np.nan if rng.random() < 0.50 else rng.choice(UNRELATED_HASHTAGS)
    return {
        "id": post_id,
        "descripcion": description,
        "hashtags": hashtags,
        "categoria": primary,
        "categoria_generadora": primary,
        "categoria_secundaria": np.nan,
        "es_ambiguo": False,
        "tipo_post": "outlier",
        "nivel_ruido": "alto",
        "etiqueta_ruidosa": False,
        "entidades": detect_entities("" if pd.isna(description) else str(description), hashtags),
    }


def generate_post(rng: random.Random, fake: Faker, post_id: int) -> dict[str, object]:
    if rng.random() < 0.012:
        return generate_outlier_post(rng, fake, post_id)

    length_type = choose_length_type(rng)
    noise_level = choose_noise_level(rng)
    is_ambiguous = rng.random() < 0.36

    if is_ambiguous:
        category_a, category_b = weighted_pair_choice(rng)
        primary, secondary = (category_a, category_b) if rng.random() < 0.55 else (category_b, category_a)
        description = render_overlap_text(rng, fake, primary, secondary, length_type)
    else:
        primary = weighted_choice(rng, CATEGORY_WEIGHTS)
        secondary = None
        description = render_category_text(rng, fake, primary, length_type)

        if rng.random() < 0.11:
            secondary = similar_category(rng, primary)
            extra_topic = rng.choice(CATEGORY_DATA[secondary]["topics"])
            description = f"{description}; tambien {extra_topic}"
            is_ambiguous = True

    description = apply_social_noise(rng, description, noise_level)
    hashtags = generate_hashtags(rng, primary, secondary, noise_level)

    if rng.random() < 0.026:
        description = np.nan
    if pd.isna(description) and rng.random() < 0.22:
        hashtags = np.nan

    assigned_category = primary
    noisy_label = False
    if rng.random() < 0.038:
        assigned_category = similar_category(rng, primary, secondary)
        noisy_label = assigned_category != primary

    return {
        "id": post_id,
        "descripcion": description,
        "hashtags": hashtags,
        "categoria": assigned_category,
        "categoria_generadora": primary,
        "categoria_secundaria": secondary if secondary else np.nan,
        "es_ambiguo": bool(is_ambiguous),
        "tipo_post": length_type,
        "nivel_ruido": noise_level,
        "etiqueta_ruidosa": bool(noisy_label),
        "entidades": detect_entities("" if pd.isna(description) else str(description), hashtags),
    }


def create_dataset(num_records: int, random_state: int) -> pd.DataFrame:
    rng = random.Random(random_state)
    np.random.seed(random_state)

    fake = Faker("es_ES")
    fake.seed_instance(random_state)
    Faker.seed(random_state)

    rows = [generate_post(rng, fake, post_id) for post_id in range(1, num_records + 1)]
    df = pd.DataFrame(rows)
    return df[
        [
            "id",
            "descripcion",
            "hashtags",
            "categoria",
            "categoria_generadora",
            "categoria_secundaria",
            "es_ambiguo",
            "tipo_post",
            "nivel_ruido",
            "etiqueta_ruidosa",
            "entidades",
        ]
    ]


def build_text_column(df: pd.DataFrame) -> pd.Series:
    return (
        df["descripcion"].fillna("").astype(str).str.strip()
        + " "
        + df["hashtags"].fillna("").astype(str).str.strip()
    ).str.replace(r"\s+", " ", regex=True).str.strip()


def normalize_for_analysis(text: str) -> str:
    text = str(text).lower().replace("#", " ")
    text = re.sub(r"[^a-záéíóúüñ0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


SPANISH_STOPWORDS = {
    "a",
    "al",
    "algo",
    "ante",
    "antes",
    "como",
    "con",
    "contra",
    "cual",
    "cuando",
    "de",
    "del",
    "desde",
    "donde",
    "durante",
    "e",
    "el",
    "ella",
    "ellas",
    "ellos",
    "en",
    "entre",
    "era",
    "eran",
    "es",
    "esa",
    "esas",
    "ese",
    "eso",
    "esos",
    "esta",
    "estaba",
    "estaban",
    "estado",
    "estan",
    "estar",
    "estas",
    "este",
    "esto",
    "estos",
    "fue",
    "ha",
    "hay",
    "he",
    "la",
    "las",
    "le",
    "les",
    "lo",
    "los",
    "mas",
    "me",
    "mi",
    "mis",
    "mucho",
    "muy",
    "ni",
    "no",
    "nos",
    "o",
    "para",
    "pero",
    "por",
    "que",
    "se",
    "sin",
    "sobre",
    "su",
    "sus",
    "tambien",
    "tan",
    "te",
    "tiene",
    "todo",
    "tras",
    "tu",
    "un",
    "una",
    "unas",
    "uno",
    "unos",
    "y",
    "ya",
}


def sparse_pair_cosine(matrix, indices_a: np.ndarray, indices_b: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[indices_a].multiply(matrix[indices_b]).sum(axis=1)).ravel()


def sample_similarity_diagnostics(
    df: pd.DataFrame,
    text_clean: pd.Series,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    df_text = df[text_clean.str.len() > 0].copy().reset_index(drop=True)
    text_clean = text_clean[text_clean.str.len() > 0].reset_index(drop=True)

    vectorizer = TfidfVectorizer(
        min_df=3,
        max_df=0.82,
        max_features=12_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm="l2",
    )
    tfidf = vectorizer.fit_transform(text_clean)
    rng = np.random.default_rng(random_state)

    indices_by_category = {
        category: df_text.index[df_text["categoria"].eq(category)].to_numpy()
        for category in sorted(df_text["categoria"].unique())
    }

    intra_rows = []
    for category, indices in indices_by_category.items():
        n_pairs = min(1_800, max(200, len(indices)))
        if len(indices) < 2:
            continue
        left = rng.choice(indices, size=n_pairs, replace=True)
        right = rng.choice(indices, size=n_pairs, replace=True)
        valid = left != right
        sims = sparse_pair_cosine(tfidf, left[valid], right[valid])
        intra_rows.append(
            {
                "categoria": category,
                "n_posts": len(indices),
                "similitud_intra_media": float(np.mean(sims)),
                "similitud_intra_p10": float(np.percentile(sims, 10)),
                "similitud_intra_p90": float(np.percentile(sims, 90)),
            }
        )

    inter_rows = []
    for category_a, category_b in combinations(indices_by_category.keys(), 2):
        indices_a = indices_by_category[category_a]
        indices_b = indices_by_category[category_b]
        n_pairs = min(900, len(indices_a), len(indices_b))
        left = rng.choice(indices_a, size=n_pairs, replace=True)
        right = rng.choice(indices_b, size=n_pairs, replace=True)
        sims = sparse_pair_cosine(tfidf, left, right)
        inter_rows.append(
            {
                "categoria_a": category_a,
                "categoria_b": category_b,
                "similitud_inter_media": float(np.mean(sims)),
                "similitud_inter_p90": float(np.percentile(sims, 90)),
            }
        )

    intra_df = pd.DataFrame(intra_rows).sort_values("similitud_intra_media", ascending=False)
    inter_df = pd.DataFrame(inter_rows).sort_values("similitud_inter_media", ascending=False)
    summary = {
        "tfidf_vocab_size": int(len(vectorizer.get_feature_names_out())),
        "similitud_intra_global": float(intra_df["similitud_intra_media"].mean()),
        "similitud_inter_global": float(inter_df["similitud_inter_media"].mean()),
    }
    summary["separacion_intra_inter"] = summary["similitud_intra_global"] - summary["similitud_inter_global"]
    return intra_df, inter_df, summary


def build_word_frequencies(text_clean: pd.Series, top_n: int = 60) -> pd.DataFrame:
    tokens = []
    for text in text_clean:
        tokens.extend(
            token
            for token in str(text).split()
            if token not in SPANISH_STOPWORDS and len(token) > 2
        )
    return pd.DataFrame(Counter(tokens).most_common(top_n), columns=["palabra", "frecuencia"])


def build_separation_flags(intra_df: pd.DataFrame, inter_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category in intra_df["categoria"]:
        intra = float(intra_df.loc[intra_df["categoria"].eq(category), "similitud_intra_media"].iloc[0])
        related = inter_df[
            inter_df["categoria_a"].eq(category) | inter_df["categoria_b"].eq(category)
        ].copy()
        max_inter = float(related["similitud_inter_media"].max())
        closest = related.sort_values("similitud_inter_media", ascending=False).iloc[0]
        closest_category = (
            closest["categoria_b"] if closest["categoria_a"] == category else closest["categoria_a"]
        )
        gap = intra - max_inter
        if gap > 0.18:
            status = "demasiado_separada"
        elif gap > 0.09:
            status = "separacion_moderada"
        else:
            status = "solapada"
        rows.append(
            {
                "categoria": category,
                "similitud_intra_media": intra,
                "categoria_inter_mas_cercana": closest_category,
                "similitud_inter_mas_cercana": max_inter,
                "brecha": gap,
                "diagnostico": status,
            }
        )
    return pd.DataFrame(rows).sort_values("brecha", ascending=False)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Sin filas._"

    printable = df.copy()
    for column in printable.select_dtypes(include=[float]).columns:
        printable[column] = printable[column].map(lambda value: f"{value:.4f}")

    headers = [str(column) for column in printable.columns]
    rows = printable.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def analyze_dataset(df: pd.DataFrame, report_dir: Path, random_state: int) -> dict[str, object]:
    report_dir.mkdir(parents=True, exist_ok=True)

    df_analysis = df.copy()
    df_analysis["texto_original"] = build_text_column(df_analysis)
    df_analysis["texto_limpio"] = df_analysis["texto_original"].apply(normalize_for_analysis)
    df_analysis["longitud_palabras"] = df_analysis["texto_original"].apply(lambda value: len(value.split()))
    df_analysis["longitud_caracteres"] = df_analysis["texto_original"].str.len()

    category_distribution = (
        df_analysis["categoria"]
        .value_counts()
        .rename_axis("categoria")
        .reset_index(name="n_posts")
        .sort_values("categoria")
    )
    category_distribution["porcentaje"] = (
        100 * category_distribution["n_posts"] / len(df_analysis)
    ).round(2)

    length_by_category = (
        df_analysis.groupby("categoria")["longitud_palabras"]
        .agg(n_posts="size", media="mean", mediana="median", p10=lambda s: s.quantile(0.10), p90=lambda s: s.quantile(0.90))
        .round(2)
        .reset_index()
    )

    word_frequencies = build_word_frequencies(df_analysis["texto_limpio"])
    intra_df, inter_df, similarity_summary = sample_similarity_diagnostics(
        df_analysis,
        df_analysis["texto_limpio"],
        random_state,
    )
    separation_flags = build_separation_flags(intra_df, inter_df)

    random_examples = pd.concat(
        [
            df_analysis.sample(12, random_state=random_state),
            df_analysis[df_analysis["es_ambiguo"]].sample(
                min(12, int(df_analysis["es_ambiguo"].sum())),
                random_state=random_state + 1,
            ),
            df_analysis[df_analysis["nivel_ruido"].eq("alto")].sample(
                min(12, int(df_analysis["nivel_ruido"].eq("alto").sum())),
                random_state=random_state + 2,
            ),
        ],
        ignore_index=True,
    ).drop_duplicates("id")

    category_distribution.to_csv(report_dir / "category_distribution.csv", index=False, encoding="utf-8")
    length_by_category.to_csv(report_dir / "length_by_category.csv", index=False, encoding="utf-8")
    word_frequencies.to_csv(report_dir / "word_frequencies.csv", index=False, encoding="utf-8")
    intra_df.round(4).to_csv(report_dir / "similarity_intra_tfidf.csv", index=False, encoding="utf-8")
    inter_df.round(4).to_csv(report_dir / "similarity_inter_tfidf.csv", index=False, encoding="utf-8")
    separation_flags.round(4).to_csv(report_dir / "separation_flags.csv", index=False, encoding="utf-8")
    random_examples[
        [
            "id",
            "descripcion",
            "hashtags",
            "categoria",
            "categoria_generadora",
            "categoria_secundaria",
            "es_ambiguo",
            "tipo_post",
            "nivel_ruido",
            "etiqueta_ruidosa",
            "entidades",
        ]
    ].to_csv(report_dir / "random_examples.csv", index=False, encoding="utf-8")

    dataset_summary = {
        "n_posts": int(len(df_analysis)),
        "n_categories": int(df_analysis["categoria"].nunique()),
        "ambiguous_rate": float(df_analysis["es_ambiguo"].mean()),
        "noisy_label_rate": float(df_analysis["etiqueta_ruidosa"].mean()),
        "missing_description_rate": float(df_analysis["descripcion"].isna().mean()),
        "missing_hashtags_rate": float(df_analysis["hashtags"].isna().mean()),
        "empty_text_rate": float(df_analysis["texto_original"].str.len().eq(0).mean()),
        "mean_words": float(df_analysis["longitud_palabras"].mean()),
        "median_words": float(df_analysis["longitud_palabras"].median()),
        **similarity_summary,
    }

    with (report_dir / "dataset_summary.json").open("w", encoding="utf-8") as file:
        json.dump(dataset_summary, file, ensure_ascii=False, indent=2)

    markdown = [
        "# Diagnostico automatico del dataset",
        "",
        f"- Registros: {dataset_summary['n_posts']}",
        f"- Categorias: {dataset_summary['n_categories']}",
        f"- Posts ambiguos: {dataset_summary['ambiguous_rate']:.2%}",
        f"- Etiquetas ruidosas simuladas: {dataset_summary['noisy_label_rate']:.2%}",
        f"- Descripciones ausentes: {dataset_summary['missing_description_rate']:.2%}",
        f"- Hashtags ausentes: {dataset_summary['missing_hashtags_rate']:.2%}",
        f"- Longitud mediana: {dataset_summary['median_words']:.1f} palabras",
        f"- Vocabulario TF-IDF de diagnostico: {dataset_summary['tfidf_vocab_size']}",
        f"- Similitud intra-categoria media: {dataset_summary['similitud_intra_global']:.4f}",
        f"- Similitud inter-categoria media: {dataset_summary['similitud_inter_global']:.4f}",
        f"- Brecha intra-inter: {dataset_summary['separacion_intra_inter']:.4f}",
        "",
        "## Categorias potencialmente demasiado separadas",
        "",
        dataframe_to_markdown(separation_flags.round(4)),
        "",
        "## Pares inter-categoria mas cercanos",
        "",
        dataframe_to_markdown(inter_df.head(12).round(4)),
        "",
        "## Palabras mas frecuentes",
        "",
        dataframe_to_markdown(word_frequencies.head(25)),
    ]

    (report_dir / "dataset_summary.md").write_text("\n".join(markdown), encoding="utf-8")
    return dataset_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera un dataset sintetico de publicaciones sociales con ruido, "
            "ambiguedad semantica y diagnosticos automaticos."
        )
    )
    parser.add_argument("--records", type=int, default=DEFAULT_NUM_RECORDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--no-analysis", action="store_true")
    return parser.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generando {args.records} publicaciones sinteticas con seed={args.seed}...")
    df = create_dataset(args.records, random_state=args.seed)
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Dataset guardado en: {args.output}")
    print(f"Columnas: {', '.join(df.columns)}")

    if not args.no_analysis:
        print(f"Calculando diagnostico automatico en: {args.report_dir}")
        summary = analyze_dataset(df, args.report_dir, random_state=args.seed)
        print(
            "Resumen: "
            f"ambiguos={summary['ambiguous_rate']:.2%}, "
            f"etiquetas_ruidosas={summary['noisy_label_rate']:.2%}, "
            f"brecha_intra_inter={summary['separacion_intra_inter']:.4f}"
        )

    sample_columns = ["descripcion", "hashtags", "categoria", "categoria_secundaria", "nivel_ruido"]
    print("\nMuestra aleatoria:")
    print(df.sample(8, random_state=args.seed)[sample_columns].to_string(index=False))


if __name__ == "__main__":
    main()

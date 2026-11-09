# Preguntas para comprobar si el agente usa (o no) la búsqueda web.
# Formato: (pregunta, ¿debería buscar?)
#  - False: conocimiento estable o anterior a la fecha de corte (feb. 2025)
#  - True: posterior a la fecha de corte, cambia con el tiempo o se pide buscar
QUESTIONS = [
    ("¿Dónde está la arteria aorta?", False),
    ("¿Quién escribió el Quijote?", False),
    ("¿Qué es la fotosíntesis?", False),
    ("¿Quién ganó el Mundial de 2022?", False),
    ("¿Quién ganó el Mundial de 2026?", True),
    ("¿Qué tiempo hace hoy en Sevilla?", True),
    ("Noticias de hoy en la Comunidad de Madrid", True),
    ("¿Cuál es la última versión de Python?", True),
    ("Busca en internet qué es LangGraph", True),
]

# Paso 2: búsquedas para comprobar si el snippet contiene la parte relevante
# de la página. Formato: (query, palabras que debería contener una buena respuesta)
SNIPPET_CASES = [
    ("¿Quién ganó el Mundial de 2022?", ["argentina"]),
    ("Honda Rebel 500 potencia CV", ["46"]),
    ("última versión de Python", ["3.15"]),
    # TODO: escribe en la lista el ganador real del Mundial 2026 (p.ej. ["españa"])
    # para comprobar si los snippets lo contienen. Si se deja vacía, solo se
    # analiza de dónde sale cada snippet.
    ("¿Quién ganó el Mundial de 2026?", []),
]

# Paso 3: conversación de varios turnos (para probar el prompt caching).
CONVERSATION = [
    "¿Cuál es la última versión de Python?",
    "¿Qué novedades principales trae esa versión?",
    "¿Y cuándo está previsto que salga la siguiente?",
    "Resúmeme todo lo anterior en tres frases.",
]

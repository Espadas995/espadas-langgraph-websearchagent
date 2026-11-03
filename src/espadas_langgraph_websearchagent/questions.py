# Questions to test if the model is using the web search tool
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
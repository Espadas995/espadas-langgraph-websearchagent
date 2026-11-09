"""
Llamada directa a la API REST de Tavily
=======================================

`langchain_tavily.TavilySearch` por dentro solo hace un POST a
https://api.tavily.com/search. Lo hacemos nosotros mismos con `requests`
porque así:
  - vemos exactamente qué parámetros se envían,
  - podemos usar parámetros que la tool de LangChain no deja tocar
    (por ejemplo `chunks_per_source`),
  - y en el Paso 2 podemos pedir la página completa (`include_raw_content`)
    para compararla con el snippet.

Parámetros más útiles (documentación oficial de Tavily):
  search_depth       "basic" (1 crédito) | "advanced" (2 créditos) | "fast" | "ultra-fast"
  max_results        nº de resultados (0-20)
  chunks_per_source  nº de trozos de texto por página (1-3). Cada trozo tiene
                     ~500 caracteres (basic/fast) u ~800 (advanced)
  topic              "general" | "news" | "finance"
  time_range         "day" | "week" | "month" | "year"
  country            da prioridad a resultados de un país (p.ej. "spain")
  include_answer     respuesta generada por un LLM de Tavily
  include_raw_content  página completa en markdown/texto
  include_usage      devuelve los créditos gastados
"""

import os

import requests

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def tavily_api_search(query: str, **params) -> dict:
    """Hace una búsqueda en Tavily y devuelve el JSON de respuesta como dict.

    Ejemplo:
        tavily_api_search("última versión de Python", max_results=3, chunks_per_source=1)
    """
    headers = {
        "Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}",
        "Content-Type": "application/json",
    }
    # Quitamos los parámetros a None para que Tavily use sus valores por defecto
    body = {"query": query, **{k: v for k, v in params.items() if v is not None}}

    response = requests.post(TAVILY_SEARCH_URL, json=body, headers=headers, timeout=60)
    response.raise_for_status()  # si Tavily devuelve un error, lanzamos una excepción
    return response.json()

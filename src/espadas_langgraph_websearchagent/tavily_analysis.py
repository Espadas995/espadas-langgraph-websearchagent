"""
PASO 2 - Entender Tavily y evaluar su eficacia
===============================================

Preguntas que respondemos en este paso:

1. ¿Qué ve el LLM cuando hacemos bind_tools()?  → get_tool_schema() / print_tool_parameters()
2. ¿Qué devuelve Tavily y con qué estructura?    → describe_response()
3. ¿Cómo se generan los snippets y son la parte
   relevante de la página?                        → analyze_snippets()
4. ¿Cómo se genera el score y concuerda con la
   relevancia real?                               → compare_scores_across_queries(),
                                                    compare_search_depths(),
                                                    score_vs_llm_relevance()
5. ¿El modelo usa las respuestas o alucina?       → check_cited_urls(), check_grounding()

Cómo genera Tavily los snippets (campo `content`), según su documentación:
  - Tavily descarga la página y la parte en trozos ("chunks") de unos 500
    caracteres (basic/fast) u 800 (advanced).
  - Ordena esos trozos según lo relacionados que están con la QUERY
    (no con la pregunta del usuario: con la query que escribió el LLM).
  - Devuelve los mejores `chunks_per_source` trozos (3 por defecto), unidos
    con " [...] ".
  - Con search_depth="ultra-fast" en cambio devuelve un resumen NLP de la página.

Cómo se genera el `score`:
  - Tavily solo dice que "indica la relevancia entre la query y el contenido".
    La fórmula no es pública (es su propio modelo de ranking).
  - Es un número entre 0 y 1, pero NO es una probabilidad: sirve para ordenar
    resultados de UNA búsqueda, no para comparar búsquedas distintas.
  - Tavily recomienda decidir el umbral según el caso de uso (en sus ejemplos
    usan 0,7, pero no lo recomiendan como valor general).
"""

import json
import re

from langchain_anthropic.chat_models import convert_to_anthropic_tool

from espadas_langgraph_websearchagent.judge import check_grounding as llm_check_grounding
from espadas_langgraph_websearchagent.judge import judge_relevance
from espadas_langgraph_websearchagent.searchagent import get_final_answer, get_search_results
from espadas_langgraph_websearchagent.tavily_api import tavily_api_search

# ---------------------------------------------------------------------------
# 1. Lo que ve el LLM: el schema de la tool
# ---------------------------------------------------------------------------


def get_tool_schema(tool) -> dict:
    """Devuelve la tool en el formato JSON exacto que se envía a Anthropic.

    Esto es lo que hace bind_tools() por dentro con cada tool.
    """
    return convert_to_anthropic_tool(tool)


def print_tool_schema(tool) -> None:
    """Imprime el JSON completo de la tool (lo que lee el LLM en CADA llamada)."""
    schema = get_tool_schema(tool)
    text = json.dumps(schema, indent=2, ensure_ascii=False)
    print(text)
    print()
    print(f"Longitud del JSON: {len(text):,} caracteres")


def print_tool_parameters(tool) -> None:
    """Tabla resumen de los parámetros que el LLM puede rellenar al llamar a la tool."""
    schema = get_tool_schema(tool)
    properties = schema["input_schema"].get("properties", {})
    required = schema["input_schema"].get("required", [])

    print(f"Tool: {schema['name']}")
    print(f"Descripción ({len(schema.get('description', ''))} caracteres): {schema.get('description', '')}")
    print()
    print(f"{'parámetro':<18} {'obligatorio':<12} {'por defecto':<14} {'caracteres de descripción':>26}")
    print("-" * 74)
    for name, prop in properties.items():
        description = prop.get("description", "")
        default = str(prop.get("default", "-"))
        print(f"{name:<18} {'sí' if name in required else 'no':<12} {default:<14} {len(description):>26}")


# ---------------------------------------------------------------------------
# 2. Qué devuelve Tavily
# ---------------------------------------------------------------------------


def describe_response(response: dict) -> None:
    """Imprime la estructura de una respuesta de Tavily (claves, tipos y ejemplo)."""
    print("Claves de primer nivel:")
    for key, value in response.items():
        preview = repr(value)[:80]
        print(f"  {key:<22} {type(value).__name__:<8} {preview}")

    results = response.get("results", [])
    print(f"\nNúmero de resultados: {len(results)}")
    if results:
        print("\nClaves de cada resultado:")
        for key, value in results[0].items():
            preview = repr(value)[:80]
            print(f"  {key:<22} {type(value).__name__:<8} {preview}")

    # ¿Cuánto ocupa como texto? (esto es lo que acaba en el ToolMessage)
    as_text = str(response)
    useful = sum(len(r.get("title", "")) + len(r.get("url", "")) + len(r.get("content", "")) for r in results)
    print(f"\nCaracteres totales como texto (str(dict)): {len(as_text):,}")
    print(f"Caracteres útiles (título + url + snippet):  {useful:,}")
    print(f"→ El {100 - useful / max(len(as_text), 1) * 100:.0f}% del texto son claves, comillas, None, score...")


# ---------------------------------------------------------------------------
# 3. ¿Los snippets son la parte relevante de la página?
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Pasa a minúsculas, quita símbolos de markdown y junta los espacios."""
    text = text.lower()
    text = re.sub(r"[#*_`>\[\]|]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_chunks(content: str) -> list[str]:
    """Tavily une los trozos (chunks) del snippet con ' [...] '. Los separamos."""
    return [c.strip() for c in content.split("[...]") if c.strip()]


def _find_chunk(chunk: str, page: str) -> int:
    """Busca un chunk dentro de la página. Devuelve la posición o -1.

    Tavily limpia un poco el texto, así que no buscamos el chunk entero:
    buscamos 3 fragmentos de 40 caracteres (inicio, medio y final). Si
    encontramos alguno, consideramos que el chunk sale de esa página.
    """
    chunk_n, page_n = _normalize(chunk), _normalize(page)
    if not chunk_n:
        return -1
    middle = len(chunk_n) // 2
    for fragment in (chunk_n[:40], chunk_n[middle - 20 : middle + 20], chunk_n[-40:]):
        position = page_n.find(fragment)
        if fragment and position >= 0:
            return position
    return -1


def analyze_snippets(query: str, keywords: list[str] | None = None, **params) -> list[dict]:
    """Compara cada snippet con la página completa de la que sale.

    Hacemos la búsqueda pidiendo también `include_raw_content` (la página entera)
    y para cada resultado calculamos:
      - chunks: en cuántos trozos está partido el snippet
      - chunks_en_pagina: cuántos de esos trozos encontramos en la página
      - posicion_%: dónde está el primer trozo dentro de la página
        (0% = al principio, 100% = al final). Si siempre es ~0% puede que
        Tavily esté cogiendo el principio de la página (menús, cabeceras...)
      - snippet / pagina: tamaño en caracteres (cuánto "resume" Tavily)
      - clave_en_snippet / clave_en_pagina: si alguna palabra clave de la
        respuesta correcta aparece en el snippet y/o en la página.
        Si está en la página pero NO en el snippet → el snippet no ha cogido
        la parte importante.

    Args:
        query: lo que buscamos.
        keywords: palabras que debería contener una buena respuesta
            (p.ej. ["Argentina"] para "¿Quién ganó el Mundial 2022?").
        **params: parámetros extra de Tavily (search_depth, max_results...).
    """
    response = tavily_api_search(query, include_raw_content="markdown", **params)
    keywords = [k.lower() for k in (keywords or [])]

    rows = []
    for r in response.get("results", []):
        content = r.get("content", "") or ""
        page = r.get("raw_content") or ""
        chunks = split_chunks(content)
        positions = [_find_chunk(c, page) for c in chunks]
        found = [p for p in positions if p >= 0]

        rows.append(
            {
                "score": round(r.get("score", 0), 2),
                "url": r.get("url", ""),
                "chunks": len(chunks),
                "chunks_en_pagina": len(found),
                "posicion_%": round(min(found) / max(len(_normalize(page)), 1) * 100) if found else None,
                "snippet": len(content),
                "pagina": len(page),
                "clave_en_snippet": any(k in content.lower() for k in keywords) if keywords else None,
                "clave_en_pagina": any(k in page.lower() for k in keywords) if keywords else None,
                "content": content,
            }
        )
    return rows


def print_snippet_analysis(rows: list[dict]) -> None:
    """Imprime la tabla de analyze_snippets()."""
    print(f"{'score':>5} {'chunks':>6} {'en pág.':>7} {'pos.%':>6} {'snippet':>8} {'página':>8} {'clave snip.':>11} {'clave pág.':>10}  url")
    for row in rows:
        print(
            f"{row['score']:>5} {row['chunks']:>6} {row['chunks_en_pagina']:>7} "
            f"{str(row['posicion_%']):>6} {row['snippet']:>8,} {row['pagina']:>8,} "
            f"{str(row['clave_en_snippet']):>11} {str(row['clave_en_pagina']):>10}  {row['url'][:60]}"
        )


# ---------------------------------------------------------------------------
# 4. ¿Cómo funciona el score?
# ---------------------------------------------------------------------------


def compare_scores_across_queries(queries: list[str], **params) -> dict:
    """Busca varias queries parecidas y muestra el score de cada URL en cada una.

    Si la misma página tiene scores distintos según cómo escribamos la query,
    confirma que el score mide "parecido entre query y contenido" y no la
    calidad de la página en sí.

    Returns:
        {url: {query: score}}
    """
    table: dict[str, dict[str, float]] = {}
    for query in queries:
        response = tavily_api_search(query, **params)
        for r in response.get("results", []):
            table.setdefault(r["url"], {})[query] = round(r.get("score", 0), 2)

    # Imprimimos solo las URLs que aparecen en más de una query (las comparables)
    for url, scores in table.items():
        if len(scores) > 1:
            print(url[:80])
            for query, score in scores.items():
                print(f"    {score:>5}  ← '{query}'")
    return table


def compare_search_depths(query: str, depths=("basic", "advanced"), **params) -> dict:
    """Misma query con distintos search_depth: scores, tamaño del texto y créditos."""
    output = {}
    for depth in depths:
        response = tavily_api_search(query, search_depth=depth, include_usage=True, **params)
        results = response.get("results", [])
        output[depth] = response
        credits = (response.get("usage") or {}).get("credits", "?")
        total_chars = sum(len(r.get("content", "")) for r in results)
        print(f"=== search_depth={depth} | créditos: {credits} | tiempo: {response.get('response_time')} s | caracteres de snippets: {total_chars:,}")
        for r in results:
            print(f"   {r.get('score', 0):.2f}  {r['url'][:80]}")
        print()
    return output


def score_vs_llm_relevance(question: str, results: list[dict]) -> list[dict]:
    """Compara el score de Tavily con la relevancia que da un LLM juez (0-10).

    Si el score de Tavily fuera un buen indicador de relevancia, los
    resultados con score alto deberían tener también nota alta del juez.
    """
    judgements, usage = judge_relevance(question, results)
    by_index = {j.indice: j for j in judgements}

    rows = []
    for i, r in enumerate(results, start=1):
        judgement = by_index.get(i)
        rows.append(
            {
                "score_tavily": round(r.get("score", 0), 2),
                "relevancia_llm": judgement.relevancia / 10 if judgement else None,
                "explicacion": judgement.explicacion if judgement else "",
                "url": r.get("url", ""),
            }
        )
    print(f"(El juez gastó {usage['input_tokens']:,} tokens de entrada y {usage['output_tokens']:,} de salida)")
    return rows


def correlation(xs: list[float], ys: list[float]) -> float | None:
    """Correlación de Pearson hecha a mano (entre -1 y 1).

    1 = cuando sube uno, sube el otro; 0 = no tienen relación.
    Con pocos datos (5-10 puntos) es solo orientativa.
    """
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    n = len(pairs)
    mean_x = sum(p[0] for p in pairs) / n
    mean_y = sum(p[1] for p in pairs) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    var_x = sum((x - mean_x) ** 2 for x, _ in pairs)
    var_y = sum((y - mean_y) ** 2 for _, y in pairs)
    if var_x == 0 or var_y == 0:
        return None
    return cov / (var_x**0.5 * var_y**0.5)


# ---------------------------------------------------------------------------
# 5. ¿El modelo usa las respuestas o alucina?
# ---------------------------------------------------------------------------

URL_PATTERN = re.compile(r"https?://[^\s)\]>\"']+")


def check_cited_urls(result: dict) -> dict:
    """Comprueba si las URLs que cita el agente estaban en los resultados de Tavily.

    Es una comprobación automática y sin coste: si el agente cita una URL que
    Tavily NUNCA devolvió, se la ha inventado (alucinación de fuentes).
    """
    answer = get_final_answer(result)
    # Quitamos signos de puntuación y la "/" final para no dar falsos positivos
    cited = {url.rstrip(".,;:").rstrip("/") for url in URL_PATTERN.findall(answer)}
    returned = {r["url"].rstrip("/") for search in get_search_results(result) for r in search.get("results", [])}

    invented = {url for url in cited if url not in returned}
    return {
        "urls_citadas": sorted(cited),
        "urls_devueltas_por_tavily": sorted(returned),
        "urls_inventadas": sorted(invented),
    }


def check_grounding(result: dict):
    """Pide a un LLM juez que compare la respuesta final con los snippets recibidos."""
    answer = get_final_answer(result)
    results = [r for search in get_search_results(result) for r in search.get("results", [])]
    return llm_check_grounding(answer, results)

"""
PASO 3 - Mejorar el coste y la relevancia del agente
=====================================================

Tenemos dos sitios donde mejorar:

A) A NIVEL DE LA TOOL (parámetros de Tavily)
   - max_results: menos resultados = menos tokens en el ToolMessage.
   - chunks_per_source: menos trozos de texto por página = snippets más cortos.
   - search_depth fijo a "basic": si lo dejamos libre, el LLM a veces elige
     "advanced" (que cuesta 2 créditos de Tavily en vez de 1). Si lo fijamos al
     crear la tool, el valor del LLM se ignora.
   - country="spain": como las preguntas son de España, da prioridad a
     resultados españoles (más relevancia).

B) A NIVEL DEL AGENTE
   1. Mejor system prompt: el agente base no buscó en "¿Quién ganó el Mundial
      de 2026?" ni en "¿Qué tiempo hace hoy en Sevilla?" (7/9 aciertos).
   2. Tool "ligera": la tool de LangChain tiene 8 parámetros con descripciones
      larguísimas que se envían en CADA llamada (aunque no se busque nada).
      Una tool con solo `query` y una descripción corta ahorra muchos tokens fijos.
   3. Formato compacto: en vez de meter str(dict) con todas las claves
      ('follow_up_questions': None, 'images': [], 'request_id'...), metemos solo
      título + URL + snippet.
   4. No inyectar búsquedas infructuosas: si no hay resultados (o todos se
      filtran), metemos un mensaje cortito en vez del resultado completo.
      OJO: no podemos "no enviar nada". La API de Anthropic obliga a que cada
      tool_call tenga su ToolMessage de respuesta; si falta, da error.
   5. Filtrar por score: quitar resultados con score bajo.
   6. LLM juez: un LLM puntúa la relevancia de cada resultado (esquema Pydantic
      con relevancia + explicación) y solo pasamos los relevantes. Cuesta una
      llamada extra; hay que comprobar si compensa.
   7. Prompt caching: ver la explicación en run_conversation().
   8. Sin búsquedas en paralelo (parallel_tool_calls=False): en el ejemplo de la
      Honda Rebel el LLM lanzó 2 búsquedas "advanced" a la vez → 4.741 tokens
      de ToolMessage (casi el doble de lo normal).
"""

from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_tavily import TavilySearch

from espadas_langgraph_websearchagent.judge import judge_relevance
from espadas_langgraph_websearchagent.searchagent import build_agent
from espadas_langgraph_websearchagent.tavily_api import tavily_api_search
from espadas_langgraph_websearchagent.tokens import get_token_usage

# ---------------------------------------------------------------------------
# B1. System prompt mejorado
# ---------------------------------------------------------------------------

# Cambios respecto al original (searchagent.SYSTEM_PROMPT):
#  - Le decimos explícitamente que SÍ tiene acceso a internet con la tool
#    (el modelo a veces responde "no tengo acceso a datos en tiempo real").
#  - Regla clara para fechas: si la pregunta habla de algo posterior a la
#    fecha de corte → buscar SIEMPRE (caso Mundial 2026).
#  - Pedimos una sola búsqueda con una query corta, para ahorrar.
SYSTEM_PROMPT_V2 = """You are a helpful assistant. Today's date is {today}.
Your training knowledge is reliable up to {cutoff}.

You have real-time internet access through the tavily_search tool.
Never say that you cannot access current or real-time information: search instead.

Answer directly, without searching, only for stable facts and concepts
(science, history, definitions) or events clearly before {cutoff}.

You MUST use tavily_search when:
- the question mentions a date, event, season or release after {cutoff}
  (if an event was scheduled after {cutoff}, you do not know its outcome);
- the answer changes over time (news, weather, prices, scores, latest versions);
- the user asks you to search or wants sources.

Make ONE search with a short, specific query. Search again only if the
results were useless. When you use search results, cite the URLs you used."""


# ---------------------------------------------------------------------------
# A. Tool con parámetros ajustados
# ---------------------------------------------------------------------------


def create_tuned_langchain_tool(max_results: int = 3) -> TavilySearch:
    """Misma tool de LangChain, pero con search_depth fijo y menos resultados.

    Fijar search_depth al crear la tool hace que el valor que elija el LLM se
    ignore (lo comprobamos leyendo el código de langchain_tavily: usa
    `self.search_depth if self.search_depth else search_depth`).
    Ojo: el LLM sigue VIENDO todos los parámetros en el schema, así que los
    tokens fijos no bajan. Para eso está la tool ligera.
    """
    return TavilySearch(max_results=max_results, topic="general", search_depth="basic")


# ---------------------------------------------------------------------------
# B2. Tool "ligera": solo el parámetro query
# ---------------------------------------------------------------------------


def create_slim_search_tool(
    max_results: int = 3,
    search_depth: str = "basic",
    chunks_per_source: int = 2,
    country: str | None = "spain",
):
    """Crea una tool de búsqueda mínima que llama a Tavily directamente.

    El LLM solo ve: nombre + una frase de descripción + un parámetro `query`.
    Todo lo demás (max_results, search_depth...) lo decidimos nosotros aquí.
    Se llama igual que la original ("tavily_search") para que el system
    prompt siga siendo válido y la comparación sea justa.
    """

    @tool("tavily_search")
    def tavily_search(query: str) -> dict:
        """Search the web. Returns results with title, URL, relevance score and a text snippet."""
        return tavily_api_search(
            query,
            max_results=max_results,
            search_depth=search_depth,
            chunks_per_source=chunks_per_source,
            country=country,
        )

    return tavily_search


# ---------------------------------------------------------------------------
# B3-B6. Procesar el resultado antes de dárselo al LLM
# ---------------------------------------------------------------------------

# Mensaje corto que metemos cuando la búsqueda no sirve para nada.
NO_RESULTS_MESSAGE = (
    "La búsqueda no encontró resultados relevantes. "
    "No inventes datos: dilo al usuario o responde con lo que sepas avisando de que puede no estar actualizado."
)


def format_compact(results: list[dict]) -> str:
    """Formato compacto: solo lo que el LLM necesita para responder y citar."""
    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(f"[{i}] {r.get('title', '')}\nURL: {r.get('url', '')}\n{r.get('content', '')}")
    return "\n\n".join(blocks)


def make_result_processor(
    compact: bool = True,
    min_score: float | None = None,
    skip_empty: bool = True,
    llm_judge: bool = False,
    min_relevance: int = 5,
):
    """Crea la función que convierte el resultado de Tavily en el texto del ToolMessage.

    Args:
        compact: True → formato compacto (título + URL + snippet).
            False → str(dict) como el agente base.
        min_score: si se indica, quita los resultados con score de Tavily menor.
        skip_empty: si no queda ningún resultado → NO_RESULTS_MESSAGE.
        llm_judge: si es True, un LLM puntúa la relevancia (0-10) y quitamos
            los resultados con nota menor que `min_relevance`.
    """

    def process(observation, question: str) -> tuple[str, dict]:
        # Si la tool devolvió un texto (error o "sin resultados" de LangChain)
        if not isinstance(observation, dict):
            return (NO_RESULTS_MESSAGE if skip_empty else str(observation)), {}

        results = observation.get("results", [])
        extra_usage = {}

        # B5. Filtro por score de Tavily
        if min_score is not None:
            results = [r for r in results if r.get("score", 0) >= min_score]

        # B6. Filtro con LLM juez
        if llm_judge and results:
            judgements, extra_usage = judge_relevance(question, results)
            if judgements:  # si el juez falla, no filtramos nada
                keep = {j.indice for j in judgements if j.relevancia >= min_relevance}
                results = [r for i, r in enumerate(results, start=1) if i in keep]

        # B4. Búsqueda infructuosa → mensaje corto
        if not results and skip_empty:
            return NO_RESULTS_MESSAGE, extra_usage

        # B3. Formato
        if compact:
            return format_compact(results), extra_usage
        return str({**observation, "results": results}), extra_usage

    return process


# ---------------------------------------------------------------------------
# Variantes del agente (cada una añade UNA mejora sobre la anterior)
# ---------------------------------------------------------------------------


def create_agent_variants(min_score: float = 0.5, min_relevance: int = 5) -> dict:
    """Devuelve {nombre: agente}. Las mejoras son acumulativas.

    Así, al comparar una variante con la anterior, vemos el efecto de UNA
    sola mejora.
    """
    return {
        # 0. El agente original
        "0_base": build_agent(),
        # 1. + system prompt mejorado (relevancia: decidir bien cuándo buscar)
        "1_prompt_v2": build_agent(system_prompt_template=SYSTEM_PROMPT_V2),
        # 2. + parámetros de la tool (3 resultados, search_depth fijo, sin paralelo)
        "2_params_tool": build_agent(
            tools=[create_tuned_langchain_tool()],
            system_prompt_template=SYSTEM_PROMPT_V2,
            parallel_tool_calls=False,
        ),
        # 3. + tool ligera (menos tokens fijos, chunks_per_source=2, country=spain)
        "3_tool_ligera": build_agent(
            tools=[create_slim_search_tool()],
            system_prompt_template=SYSTEM_PROMPT_V2,
            parallel_tool_calls=False,
        ),
        # 4. + formato compacto + no inyectar búsquedas vacías
        "4_formato_compacto": build_agent(
            tools=[create_slim_search_tool()],
            system_prompt_template=SYSTEM_PROMPT_V2,
            parallel_tool_calls=False,
            result_processor=make_result_processor(compact=True),
        ),
        # 5. + filtro por score
        "5_filtro_score": build_agent(
            tools=[create_slim_search_tool()],
            system_prompt_template=SYSTEM_PROMPT_V2,
            parallel_tool_calls=False,
            result_processor=make_result_processor(compact=True, min_score=min_score),
        ),
        # 6. + LLM juez (en vez del filtro por score)
        "6_juez_llm": build_agent(
            tools=[create_slim_search_tool()],
            system_prompt_template=SYSTEM_PROMPT_V2,
            parallel_tool_calls=False,
            result_processor=make_result_processor(compact=True, llm_judge=True, min_relevance=min_relevance),
        ),
    }


# ---------------------------------------------------------------------------
# B7. Prompt caching en una conversación de varios turnos
# ---------------------------------------------------------------------------


def run_conversation(agent, questions: list[str]) -> list[dict]:
    """Hace varias preguntas seguidas en la MISMA conversación (como un chatbot).

    ¿Por qué hace falta una conversación para ver el efecto de la caché?
      - La caché funciona por PREFIJO: si el principio de la petición
        (tools → system → mensajes) es idéntico al de una petición anterior
        de hace menos de 5 minutos, esa parte se lee de caché (10% del precio).
        Escribirla la primera vez cuesta un 25% MÁS.
      - En Claude Haiku 4.5 el prefijo tiene que tener al menos 4.096 tokens,
        o NO se cachea (sin error: simplemente cache_creation = 0).
      - Nuestra primera llamada (tools + system + pregunta) tiene ~2.000-2.500
        tokens → NO llega al mínimo. Con una sola pregunta, la caché no ayuda.
      - En un chatbot el historial crece (sobre todo con resultados de
        búsqueda de ~3.000 tokens) y pasa de 4.096 → a partir de ahí, cada
        pregunta nueva reutiliza todo el historial anterior desde la caché.
      - La fecha de hoy está en el system prompt: cambia una vez al día, así
        que solo invalida la caché a medianoche (no es problema).

    Returns:
        Lista con el uso de tokens de cada turno (get_token_usage).
    """
    history = []
    usages = []
    for question in questions:
        result = agent.invoke({"messages": history + [HumanMessage(content=question)]})
        new_messages = result["messages"][len(history) :]  # solo lo de este turno
        usages.append(get_token_usage({"messages": new_messages}))
        history = result["messages"]
    return usages


def print_conversation_usage(name: str, usages: list[dict]) -> None:
    """Tabla turno a turno de tokens y caché."""
    print(f"=== {name} ===")
    print(" turno | input  | output | caché leída | caché escrita | coste $")
    for i, u in enumerate(usages, start=1):
        print(
            f" {i:>5} | {u['input_tokens']:>6,} | {u['output_tokens']:>6,} | "
            f"{u['cache_read_tokens']:>11,} | {u['cache_creation_tokens']:>13,} | {u['cost_usd']:.5f}"
        )
    print(f" TOTAL coste: {sum(u['cost_usd'] for u in usages):.5f} $")
    print()

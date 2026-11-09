"""
PASO 1 - El agente base (LangGraph + Claude Haiku 4.5 + Tavily)
================================================================

Este módulo construye el agente que vamos a evaluar. La idea es muy sencilla:

        ┌────────────┐   ¿el LLM pidió una tool?   ┌────────────┐
START → │  llm_call  │ ──────────── sí ──────────▶ │ tool_node  │
        └────────────┘                             └────────────┘
              │  ▲                                        │
              no └────────────────────────────────────────┘
              ▼
             END

1. `llm_call`: el modelo lee la conversación y decide si responde directamente
   o si pide una búsqueda (devuelve un AIMessage con `tool_calls`).
2. `tool_node`: ejecuta la búsqueda en Tavily y guarda el resultado en un
   `ToolMessage`.
3. Se vuelve a `llm_call` con el resultado y el modelo ya responde.

En vez de dar al agente una tool para saber la fecha, metemos la fecha actual
y la fecha de corte del modelo (knowledge cutoff) directamente en el system
prompt. Es más barato (no hace falta una llamada extra) y así el modelo sabe
qué cosas pueden haber cambiado desde que fue entrenado.

La función `build_agent()` es configurable para poder reutilizarla en el
Paso 3 (variantes del agente con mejoras). `create_agent_with_search_tool()`
es el agente "tal cual", que usamos como referencia (baseline).
"""

import operator
from datetime import date
from typing import Annotated, Callable, Literal

from langchain.chat_models import init_chat_model
from langchain.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

# Las funciones de tokens están en tokens.py. Las importamos aquí también para
# que el notebook antiguo (testing_agent.ipynb) siga funcionando.
from espadas_langgraph_websearchagent.tokens import (  # noqa: F401
    get_token_usage,
    print_token_usage,
)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

# Modelo usado en todo el proyecto (barato y rápido, ideal para un chatbot).
MODEL_NAME = "claude-haiku-4-5-20251001"

# Fecha de corte del conocimiento de Claude Haiku 4.5. Si cambias de modelo,
# cambia también esta fecha.
KNOWLEDGE_CUTOFF_HAIKU = "February 2025"

# System prompt con dos "huecos": {today} y {cutoff}. Se rellenan en cada
# llamada con build_system_prompt().
SYSTEM_PROMPT = """You are a helpful assistant. Today's date is {today}.
Your training knowledge is reliable up to {cutoff}.

Answer directly from your own knowledge when the question is about stable facts,
concepts, or events clearly before your knowledge cutoff.

Use the tavily_search tool ONLY when:
- the question is about events, releases, or data after your knowledge cutoff;
- the information changes frequently (news, prices, weather, scores, software versions);
- the user explicitly asks you to search or wants sources;
- you are not confident your answer is still accurate.

When you answer using search results, cite the URLs you used."""


def build_system_prompt(template: str = SYSTEM_PROMPT) -> str:
    """Rellena el system prompt con la fecha de hoy y la fecha de corte."""
    return template.format(
        today=date.today().isoformat(),
        cutoff=KNOWLEDGE_CUTOFF_HAIKU,
    )


def create_model(temperature: float = 0):
    """Crea el modelo de chat (temperature=0 para que sea lo más repetible posible)."""
    return init_chat_model(MODEL_NAME, temperature=temperature)


def create_default_search_tool() -> TavilySearch:
    """La tool de Tavily tal y como viene en la documentación de LangChain."""
    return TavilySearch(max_results=5, topic="general")


# ---------------------------------------------------------------------------
# Estado del grafo
# ---------------------------------------------------------------------------


class MessagesState(TypedDict):
    # `operator.add` significa: cada nodo devuelve mensajes NUEVOS y LangGraph
    # los añade al final de la lista (no la sustituye).
    messages: Annotated[list[AnyMessage], operator.add]
    # Contador de llamadas al LLM (para saber cuántas vueltas dio el agente).
    llm_calls: int


# Tipo de la función opcional que procesa el resultado de una búsqueda antes
# de dárselo al LLM (se usa en el Paso 3). Recibe:
#   - observation: lo que devolvió la tool (un dict de Tavily o un texto de error)
#   - question: la pregunta original del usuario
# y devuelve:
#   - el texto que se mete en el ToolMessage (lo que "lee" el LLM)
#   - un dict con tokens extra gastados (por ejemplo, si un LLM juez evaluó
#     la relevancia de los resultados). Si no hay gasto extra, dict vacío.
ResultProcessor = Callable[[object, str], tuple[str, dict]]


def default_result_processor(observation, question: str) -> tuple[str, dict]:
    """Comportamiento original: el dict de Tavily se convierte a texto con str().

    Ojo: esto es lo que LangChain hace por debajo cuando le pasas un dict
    como `content` a un ToolMessage. Por eso el LLM recibe algo como
    "{'query': '...', 'follow_up_questions': None, 'answer': None, ...}",
    es decir, TODAS las claves del JSON, aunque no sirvan para nada.
    """
    return str(observation), {}


# ---------------------------------------------------------------------------
# Construcción del agente
# ---------------------------------------------------------------------------


def build_agent(
    tools: list | None = None,
    system_prompt_template: str = SYSTEM_PROMPT,
    result_processor: ResultProcessor = default_result_processor,
    use_prompt_cache: bool = False,
    parallel_tool_calls: bool | None = None,
):
    """Construye el grafo del agente. Todos los parámetros son opcionales.

    Args:
        tools: lista de tools. Por defecto, TavilySearch(max_results=5).
        system_prompt_template: plantilla del system prompt ({today}, {cutoff}).
        result_processor: función que convierte el resultado de la búsqueda
            en el texto del ToolMessage (Paso 3: filtrar, compactar...).
        use_prompt_cache: si es True, activa el prompt caching de Anthropic
            (Paso 3).
        parallel_tool_calls: si es False, el modelo solo puede pedir UNA
            búsqueda por turno (evita búsquedas en paralelo que multiplican
            los tokens). None = comportamiento por defecto (permitidas).
    """
    # Paso 1: modelo + tools
    model = create_model()
    if tools is None:
        tools = [create_default_search_tool()]
    tools_by_name = {tool.name: tool for tool in tools}

    # bind_tools() convierte cada tool a JSON (nombre + descripción + schema
    # de parámetros) y la envía en CADA llamada al modelo. Ese JSON es parte
    # de los ~2000 tokens de entrada "fijos" que vemos incluso cuando el
    # agente no busca nada (lo analizamos en el Paso 2).
    if parallel_tool_calls is None:
        model_with_tools = model.bind_tools(tools)
    else:
        model_with_tools = model.bind_tools(tools, parallel_tool_calls=parallel_tool_calls)

    # Paso 2: nodo del modelo
    def llm_call(state: MessagesState):
        """El LLM decide si responde directamente o si busca en internet."""
        system = SystemMessage(content=build_system_prompt(system_prompt_template))

        if use_prompt_cache:
            # `cache_control` en la llamada = "caché automática": Anthropic pone
            # un punto de caché en el último bloque de la conversación. En la
            # siguiente llamada, todo lo que sea idéntico hasta ese punto
            # (tools + system + mensajes anteriores) se lee de caché y cuesta
            # un 10% del precio normal. Ver Paso 3 para las limitaciones.
            response = model_with_tools.invoke(
                [system] + state["messages"],
                cache_control={"type": "ephemeral"},
            )
        else:
            response = model_with_tools.invoke([system] + state["messages"])

        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    # Paso 3: nodo de las tools
    def tool_node(state: MessagesState):
        """Ejecuta cada búsqueda que pidió el LLM y crea un ToolMessage por cada una."""
        question = get_user_question(state["messages"])

        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = tools_by_name[tool_call["name"]]

            # Ejecutamos la búsqueda. Si falla (sin conexión, sin resultados...)
            # no queremos que el agente se rompa: le pasamos el error como texto.
            try:
                observation = tool.invoke(tool_call["args"])
            except Exception as error:  # noqa: BLE001
                observation = f"Error en la búsqueda: {error}"

            # Convertimos el resultado en el texto que leerá el LLM.
            content, extra_usage = result_processor(observation, question)

            result.append(
                ToolMessage(
                    # `content` = lo que se envía al LLM (¡esto es lo que cuesta tokens!)
                    content=content,
                    # `artifact` = el resultado original completo. NO se envía al
                    # LLM; lo guardamos solo para poder analizarlo nosotros.
                    artifact=observation,
                    tool_call_id=tool_call["id"],
                    # Tokens extra gastados al procesar el resultado (p.ej. el LLM
                    # juez del Paso 3). Tampoco se envía al LLM.
                    response_metadata={"extra_usage": extra_usage},
                )
            )
        return {"messages": result}

    # Paso 4: ¿seguimos o terminamos?
    def should_continue(state: MessagesState) -> Literal["tool_node", END]:
        """Si el último mensaje del LLM pide una tool → tool_node. Si no → fin."""
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tool_node"
        return END

    # Paso 5: montar el grafo
    agent_builder = StateGraph(MessagesState)
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    agent_builder.add_edge("tool_node", "llm_call")

    return agent_builder.compile()


def create_agent_with_search_tool():
    """Agente base (baseline): TavilySearch(max_results=5) y el resultado tal cual."""
    return build_agent()


# ---------------------------------------------------------------------------
# Acceso a los mensajes del resultado (Paso 1: "crear acceso a los ToolMessage")
# ---------------------------------------------------------------------------


def get_user_question(messages: list) -> str:
    """Devuelve el texto del último mensaje del usuario."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.text
    return ""


def get_tool_messages(result: dict) -> list[ToolMessage]:
    """Devuelve todos los ToolMessage (resultados de búsqueda) de una ejecución."""
    return [m for m in result["messages"] if isinstance(m, ToolMessage)]


def get_search_results(result: dict) -> list[dict]:
    """Devuelve los resultados originales de Tavily (los `artifact`) que sean un dict.

    Si una búsqueda falló, su artifact es un texto de error y no se incluye.
    """
    return [m.artifact for m in get_tool_messages(result) if isinstance(m.artifact, dict)]


def get_search_queries(result: dict) -> list[dict]:
    """Devuelve los argumentos con los que el LLM llamó a la tool (query, search_depth...)."""
    queries = []
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                queries.append(tool_call["args"])
    return queries


def get_final_answer(result: dict) -> str:
    """Devuelve el texto de la respuesta final del agente (último AIMessage)."""
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage):
            return message.text
    return ""


def did_search(result: dict) -> bool:
    """¿El agente usó la búsqueda? (= ¿hay algún ToolMessage?)"""
    return len(get_tool_messages(result)) > 0


# ---------------------------------------------------------------------------
# Ejecutar el agente y mostrarlo todo
# ---------------------------------------------------------------------------


def show_graph(agent) -> None:
    """Dibuja el grafo del agente (solo funciona dentro de un notebook)."""
    from IPython.display import Image, display  # import aquí: solo hace falta en notebooks

    display(Image(agent.get_graph(xray=True).draw_mermaid_png()))


def run_agent_with_search_tool(agent, question: str, draw_graph: bool = True) -> dict:
    """Ejecuta el agente con una pregunta, imprime los mensajes y los tokens."""
    if draw_graph:
        show_graph(agent)

    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    for message in result["messages"]:
        message.pretty_print()

    usage = get_token_usage(result)
    print_token_usage(usage)
    return result

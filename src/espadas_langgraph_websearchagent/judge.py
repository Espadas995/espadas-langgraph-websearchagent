"""
LLM como juez (con esquemas Pydantic)
=====================================

Usamos un LLM para evaluar dos cosas:

1. Relevancia de cada resultado de búsqueda (Paso 2 y Paso 3)
   → ¿El snippet ayuda de verdad a responder la pregunta?
   Nos sirve para comparar la nota del LLM con el `score` de Tavily (Paso 2) y
   para filtrar resultados irrelevantes antes de dárselos al agente (Paso 3).

2. Grounding de la respuesta final (Paso 2)
   → ¿El agente usó los resultados o se inventó cosas (alucinó)?

¿Por qué Pydantic? Con `model.with_structured_output(Esquema)` obligamos al
LLM a responder con un objeto que tiene exactamente los campos que queremos
(relevancia, explicación...). Así no tenemos que "parsear" texto libre.

Ojo: el juez también es un LLM, así que puede equivocarse. Además, aquí el
juez es el mismo modelo que el agente (Haiku 4.5), y un modelo puede ser
indulgente consigo mismo. Para un trabajo serio convendría revisar a mano
algunos casos.
"""

from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from espadas_langgraph_websearchagent.searchagent import create_model

# ---------------------------------------------------------------------------
# Esquemas
# ---------------------------------------------------------------------------


class ResultRelevance(BaseModel):
    """Evaluación de UN resultado de búsqueda."""

    indice: int = Field(description="Número del resultado, tal y como aparece entre corchetes [n]")
    relevancia: int = Field(
        description=(
            "De 0 a 10. 0 = no tiene nada que ver con la pregunta; "
            "5 = habla del tema pero no responde; "
            "10 = contiene directamente la respuesta"
        )
    )
    explicacion: str = Field(description="Una frase corta explicando la nota")


class RelevanceReport(BaseModel):
    """Evaluación de todos los resultados de una búsqueda."""

    resultados: list[ResultRelevance]


class GroundingCheck(BaseModel):
    """¿La respuesta final está respaldada por los resultados de búsqueda?"""

    afirmaciones_respaldadas: list[str] = Field(
        description="Datos de la respuesta que SÍ aparecen en los resultados de búsqueda"
    )
    afirmaciones_no_respaldadas: list[str] = Field(
        description="Datos de la respuesta que NO aparecen en los resultados (posibles alucinaciones)"
    )
    usa_los_resultados: bool = Field(description="True si la respuesta se basa principalmente en los resultados")
    explicacion: str = Field(description="Explicación breve del veredicto")


# ---------------------------------------------------------------------------
# Prompts del juez
# ---------------------------------------------------------------------------

RELEVANCE_PROMPT = """You are a strict evaluator of web search results.
For each numbered result, rate from 0 to 10 how useful its text is to answer
the user's question. Judge ONLY the text shown (title + snippet), not what the
page might contain. Answer in Spanish."""

GROUNDING_PROMPT = """You are a strict fact checker.
You receive an answer written by an assistant and the web search results it had.
List which facts of the answer are supported by the search results and which are not.
Facts that are common knowledge and not time-sensitive can be considered supported.
Answer in Spanish."""

_judge_model = None


def _get_judge_model():
    """Creamos el modelo del juez una sola vez (y solo si se usa)."""
    global _judge_model
    if _judge_model is None:
        _judge_model = create_model()
    return _judge_model


def _usage_dict(raw_message) -> dict:
    usage = raw_message.usage_metadata or {}
    return {"input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0)}


def format_results_for_judge(results: list[dict]) -> str:
    """Convierte la lista de resultados de Tavily en texto numerado [1], [2]..."""
    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(f"[{i}] {r.get('title', '')}\n{r.get('content', '')}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------


def judge_relevance(question: str, results: list[dict]) -> tuple[list[ResultRelevance], dict]:
    """Pide al LLM que puntúe la relevancia de cada resultado.

    Hacemos UNA sola llamada para todos los resultados (más barato que una
    llamada por resultado).

    Returns:
        (lista de ResultRelevance, tokens gastados por el juez)
    """
    if not results:
        return [], {"input_tokens": 0, "output_tokens": 0}

    # include_raw=True → además del objeto Pydantic, nos devuelve el AIMessage
    # original, del que sacamos los tokens que ha costado el juez.
    judge = _get_judge_model().with_structured_output(RelevanceReport, include_raw=True)

    output = judge.invoke(
        [
            SystemMessage(content=RELEVANCE_PROMPT),
            HumanMessage(content=f"Pregunta: {question}\n\nResultados:\n\n{format_results_for_judge(results)}"),
        ]
    )
    report: RelevanceReport | None = output["parsed"]
    usage = _usage_dict(output["raw"])

    if report is None:  # el LLM no devolvió un objeto válido
        return [], usage
    return report.resultados, usage


def check_grounding(answer: str, results: list[dict]) -> tuple[GroundingCheck | None, dict]:
    """Pide al LLM que compruebe si la respuesta está respaldada por los resultados."""
    judge = _get_judge_model().with_structured_output(GroundingCheck, include_raw=True)

    output = judge.invoke(
        [
            SystemMessage(content=GROUNDING_PROMPT),
            HumanMessage(
                content=(
                    f"RESPUESTA DEL ASISTENTE:\n{answer}\n\n"
                    f"RESULTADOS DE BÚSQUEDA:\n\n{format_results_for_judge(results)}"
                )
            ),
        ]
    )
    return output["parsed"], _usage_dict(output["raw"])

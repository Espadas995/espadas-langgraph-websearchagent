"""
Evaluación: pasar la batería de preguntas a un agente y resumir los resultados
=============================================================================

Para cada pregunta de questions.QUESTIONS guardamos:
  - si el agente buscó y si debía buscar (¿decisión correcta?)
  - cuántas búsquedas hizo
  - tokens (totales, fijos de la 1ª llamada, de los ToolMessage) y coste
  - si citó URLs que Tavily nunca devolvió (URLs inventadas)
  - opcionalmente, el veredicto del LLM juez sobre si la respuesta se basa en
    los resultados (grounding)
"""

from langchain.messages import HumanMessage

from espadas_langgraph_websearchagent.questions import QUESTIONS
from espadas_langgraph_websearchagent.searchagent import (
    did_search,
    get_final_answer,
    get_search_queries,
)
from espadas_langgraph_websearchagent.tavily_analysis import check_cited_urls, check_grounding
from espadas_langgraph_websearchagent.tokens import estimate_cost, get_token_usage


def evaluate_agent(agent, questions=QUESTIONS, with_grounding: bool = False, verbose: bool = True) -> list[dict]:
    """Pasa todas las preguntas al agente y devuelve una fila (dict) por pregunta."""
    rows = []
    for question, should_search in questions:
        result = agent.invoke({"messages": [HumanMessage(content=question)]})
        usage = get_token_usage(result)
        searched = did_search(result)

        row = {
            "pregunta": question,
            "debia_buscar": should_search,
            "busco": searched,
            "decision_ok": searched == should_search,
            "n_busquedas": len(get_search_queries(result)),
            "queries": get_search_queries(result),
            "tokens_totales": usage["total_tokens"],
            "tokens_fijos_1a_llamada": usage["first_call_input_tokens"],
            "tokens_toolmessage": usage["tool_result_tokens"],
            "coste_usd": usage["cost_usd"],
            "urls_inventadas": len(check_cited_urls(result)["urls_inventadas"]) if searched else 0,
            "respuesta": get_final_answer(result),
            "result": result,  # la ejecución completa, por si queremos mirarla luego
        }

        # Opcional: ¿la respuesta se basa en los resultados? (cuesta 1 llamada extra)
        if with_grounding and searched:
            verdict, judge_usage = check_grounding(result)
            row["afirmaciones_no_respaldadas"] = len(verdict.afirmaciones_no_respaldadas) if verdict else None
            row["usa_los_resultados"] = verdict.usa_los_resultados if verdict else None
            row["coste_juez_usd"] = estimate_cost(judge_usage["input_tokens"], judge_usage["output_tokens"])

        rows.append(row)

        if verbose:
            check = "OK" if row["decision_ok"] else "MAL"
            print(
                f"[{check:>3}] busco={str(searched):<5} tokens={row['tokens_totales']:>6,} "
                f"toolmsg={row['tokens_toolmessage']:>5,} coste={row['coste_usd']:.5f}$  {question}"
            )
    return rows


def summarize(name: str, rows: list[dict]) -> dict:
    """Resume las filas de evaluate_agent() en unas pocas métricas."""
    searched_rows = [r for r in rows if r["busco"]]
    not_searched_rows = [r for r in rows if not r["busco"]]

    def average(values):
        values = list(values)
        return sum(values) / len(values) if values else 0

    return {
        "variante": name,
        "aciertos": f"{sum(r['decision_ok'] for r in rows)}/{len(rows)}",
        "busquedas": sum(r["n_busquedas"] for r in rows),
        "tokens_medios": round(average(r["tokens_totales"] for r in rows)),
        "tokens_fijos_medios": round(average(r["tokens_fijos_1a_llamada"] for r in rows)),
        "toolmsg_medio_si_busca": round(average(r["tokens_toolmessage"] for r in searched_rows)),
        "tokens_medios_sin_buscar": round(average(r["tokens_totales"] for r in not_searched_rows)),
        "coste_total_usd": round(sum(r["coste_usd"] for r in rows), 5),
        "urls_inventadas": sum(r["urls_inventadas"] for r in rows),
    }


def print_summaries(summaries: list[dict]) -> None:
    """Imprime una tabla comparando las variantes."""
    columns = list(summaries[0].keys())
    widths = {c: max(len(c), *(len(str(s[c])) for s in summaries)) for c in columns}
    print(" | ".join(c.ljust(widths[c]) for c in columns))
    print("-+-".join("-" * widths[c] for c in columns))
    for s in summaries:
        print(" | ".join(str(s[c]).ljust(widths[c]) for c in columns))

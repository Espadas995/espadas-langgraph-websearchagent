"""
PASO 1 - Medir tokens y calcular el peso de los ToolMessage
============================================================

¿Cómo sabemos cuántos tokens ocupan los resultados de la búsqueda?
------------------------------------------------------------------
Anthropic NO nos dice directamente "el ToolMessage ocupó X tokens". Solo nos
dice, en cada llamada al modelo, cuántos tokens entraron (input_tokens) y
cuántos salieron (output_tokens). LangChain lo guarda en
`AIMessage.usage_metadata`.

La clave es que la API no tiene memoria: en cada llamada se REENVÍA la
conversación entera. Cuando el agente busca, hace 2 llamadas al LLM:

    Llamada 1 (el LLM decide buscar)
      ENTRA:  [system + tools] + [pregunta]
      SALE:   [AIMessage con tool_call]        ← output_tokens_1 (el LLM escribe la query)

    (Tavily busca y creamos el ToolMessage. Esto no es una llamada al LLM.)

    Llamada 2 (el LLM responde con los resultados)
      ENTRA:  [system + tools] + [pregunta] + [AIMessage con tool_call] + [ToolMessage]
      SALE:   [respuesta final]                 ← output_tokens_2

Si comparamos lo que entra en la llamada 2 con lo que entra en la llamada 1:

    input_2 = input_1 + output_1 + tokens_ToolMessage

    ⇒  tokens_ToolMessage = input_2 - input_1 - output_1

Es decir: lo que "creció" la entrada y que NO viene ni de lo que ya había
(input_1) ni de lo que escribió el propio LLM (output_1) son los resultados
de Tavily.

¿Por qué no son lo mismo los tokens de la tool_call y los del ToolMessage?
--------------------------------------------------------------------------
- tool_call: lo ESCRIBE el LLM (nombre de la tool + argumentos, p.ej. la query).
  Son tokens de SALIDA (output) de la llamada 1. Son pocos (~50-150) pero son
  los más caros (Haiku 4.5: 5 $/millón de tokens de salida).
- ToolMessage: lo DEVUELVE Tavily y nosotros se lo pasamos al LLM. Son tokens
  de ENTRADA (input) de la llamada 2. Son muchos (~2.500-5.000) pero más
  baratos (1 $/millón de tokens de entrada).

Detalles a tener en cuenta (por qué el cálculo es una aproximación muy buena
pero no perfecta):
1. Si el LLM pide varias búsquedas a la vez (llamadas en paralelo), la resta
   da el total de TODOS los ToolMessage de esa vuelta (no uno a uno).
2. El AIMessage con la tool_call se cuenta como output en la llamada 1 y como
   input en la llamada 2. El formato puede cambiar unos pocos tokens.
3. Cada ToolMessage va dentro de un bloque `tool_result` con algún token de
   "envoltorio". Este método los cuenta como parte del ToolMessage.
4. Los tokens del ToolMessage NO se pagan una sola vez: se reenvían en cada
   llamada posterior de la misma conversación. En un chatbot con historial,
   una búsqueda de 3.000 tokens se vuelve a pagar en cada pregunta siguiente.

Para comprobar que la resta es correcta, hay un segundo método "exacto":
la API `count_tokens` de Anthropic (gratuita) cuenta los tokens de una
conversación sin ejecutarla. Contamos la conversación con el ToolMessage y
con el ToolMessage vacío, y restamos. Ver `count_tool_message_tokens_exact()`.
"""

from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

# Precios de Claude Haiku 4.5 en dólares por MILLÓN de tokens.
PRICE_INPUT = 1.00  # tokens de entrada normales
PRICE_OUTPUT = 5.00  # tokens de salida
PRICE_CACHE_WRITE = 1.25  # escribir en la caché (1,25 x entrada, caché de 5 min)
PRICE_CACHE_READ = 0.10  # leer de la caché (0,1 x entrada)


def _usage_of(message: AIMessage) -> dict:
    """Saca los números de uso de un AIMessage en un formato sencillo.

    Ojo: en LangChain, `input_tokens` YA INCLUYE los tokens leídos y escritos
    en caché (LangChain los suma). Por eso los separamos aquí.
    """
    usage = message.usage_metadata or {}
    details = usage.get("input_token_details", {}) or {}
    input_tokens = usage.get("input_tokens", 0)
    cache_read = details.get("cache_read", 0) or 0
    # La escritura en caché puede venir en "cache_creation" o separada por TTL
    cache_creation = (
        (details.get("cache_creation", 0) or 0)
        + (details.get("ephemeral_5m_input_tokens", 0) or 0)
        + (details.get("ephemeral_1h_input_tokens", 0) or 0)
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": usage.get("output_tokens", 0),
        "cache_read": cache_read,
        "cache_creation": cache_creation,
        "uncached_input": input_tokens - cache_read - cache_creation,
        "has_tool_calls": bool(message.tool_calls),
    }


def estimate_cost(uncached_input: int, output: int, cache_read: int = 0, cache_creation: int = 0) -> float:
    """Coste en dólares de unos tokens con los precios de Haiku 4.5."""
    return (
        uncached_input * PRICE_INPUT
        + output * PRICE_OUTPUT
        + cache_read * PRICE_CACHE_READ
        + cache_creation * PRICE_CACHE_WRITE
    ) / 1_000_000


def get_token_usage(result: dict) -> dict:
    """Calcula tokens, coste y peso de los ToolMessage de una ejecución del agente.

    Devuelve un dict con:
        input_tokens / output_tokens / total_tokens: sumas de todas las llamadas al LLM
        tool_call_output_tokens: tokens de salida de las llamadas en las que el LLM pidió buscar
        tool_result_tokens: tokens de los ToolMessage (calculados con la resta explicada arriba)
        first_call_input_tokens: entrada de la 1ª llamada = coste "fijo" (system + tools + pregunta)
        cache_read_tokens / cache_creation_tokens: tokens leídos / escritos en caché
        extra_input_tokens / extra_output_tokens: tokens gastados fuera del agente (LLM juez)
        llm_calls: número de llamadas al LLM
        cost_usd: coste estimado en dólares (incluye caché y juez)
        calls: detalle de cada llamada (lista de dicts)
    """
    calls = []
    previous = None

    for message in result["messages"]:
        # Solo los AIMessage tienen usage_metadata (Human/Tool/System no).
        if not isinstance(message, AIMessage):
            continue

        current = _usage_of(message)
        current["tool_message_tokens"] = 0

        # Si la llamada ANTERIOR pidió una búsqueda, lo que creció la entrada
        # (descontando lo que escribió el LLM) son los ToolMessage.
        if previous is not None and previous["has_tool_calls"]:
            current["tool_message_tokens"] = (
                current["input_tokens"] - previous["input_tokens"] - previous["output_tokens"]
            )

        calls.append(current)
        previous = current

    # Tokens extra (por ejemplo del LLM juez del Paso 3), guardados en los ToolMessage
    extra_input, extra_output = 0, 0
    for message in result["messages"]:
        if isinstance(message, ToolMessage):
            extra = message.response_metadata.get("extra_usage", {}) or {}
            extra_input += extra.get("input_tokens", 0)
            extra_output += extra.get("output_tokens", 0)

    input_tokens = sum(c["input_tokens"] for c in calls)
    output_tokens = sum(c["output_tokens"] for c in calls)

    cost = sum(
        estimate_cost(c["uncached_input"], c["output_tokens"], c["cache_read"], c["cache_creation"])
        for c in calls
    ) + estimate_cost(extra_input, extra_output)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "tool_call_output_tokens": sum(c["output_tokens"] for c in calls if c["has_tool_calls"]),
        "tool_result_tokens": sum(c["tool_message_tokens"] for c in calls),
        "first_call_input_tokens": calls[0]["input_tokens"] if calls else 0,
        "cache_read_tokens": sum(c["cache_read"] for c in calls),
        "cache_creation_tokens": sum(c["cache_creation"] for c in calls),
        "extra_input_tokens": extra_input,
        "extra_output_tokens": extra_output,
        "llm_calls": len(calls),
        "cost_usd": cost,
        "calls": calls,
    }


def print_token_usage(usage: dict) -> None:
    """Imprime el uso de tokens de forma legible."""
    print("===== Uso de tokens =====")
    print(f"Llamadas al LLM:        {usage['llm_calls']:>9}")
    print(f"Tokens de entrada:      {usage['input_tokens']:>9,}")
    print(f"Tokens de salida:       {usage['output_tokens']:>9,}")
    print(f"Tokens totales:         {usage['total_tokens']:>9,}")
    print(f"Coste estimado:         {usage['cost_usd']:>9.5f} $")
    print()

    # Detalle llamada a llamada: así se ve de dónde sale la resta
    print("Detalle por llamada:")
    print("  nº | input  | output | ¿pide tool? | tokens ToolMessage (resta)")
    for i, call in enumerate(usage["calls"], start=1):
        print(
            f"  {i:>2} | {call['input_tokens']:>6,} | {call['output_tokens']:>6,} | "
            f"{'sí' if call['has_tool_calls'] else 'no':^11} | {call['tool_message_tokens']:>6,}"
        )
    print()

    if usage["tool_call_output_tokens"] > 0:
        percentage = usage["tool_result_tokens"] / usage["total_tokens"] * 100
        print("Búsqueda web: sí")
        print(f"  Tokens de la tool_call (salida del LLM):  {usage['tool_call_output_tokens']:>7,}")
        print(f"  Tokens de los ToolMessage (entrada):      {usage['tool_result_tokens']:>7,}  ({percentage:.0f}% del total)")
    else:
        print("Búsqueda web: no")

    if usage["cache_read_tokens"] or usage["cache_creation_tokens"]:
        print(f"  Caché: leídos {usage['cache_read_tokens']:,} | escritos {usage['cache_creation_tokens']:,}")
    if usage["extra_input_tokens"] or usage["extra_output_tokens"]:
        print(f"  LLM juez: entrada {usage['extra_input_tokens']:,} | salida {usage['extra_output_tokens']:,}")


# ---------------------------------------------------------------------------
# Método "exacto" con la API count_tokens de Anthropic (gratuita)
# ---------------------------------------------------------------------------


def count_tool_message_tokens_exact(model, tools: list, system_prompt: str, result: dict) -> int:
    """Cuenta los tokens de los ToolMessage con la API count_tokens.

    Idea: contamos la conversación dos veces
      A) tal cual (con los resultados de Tavily dentro de los ToolMessage)
      B) igual, pero con el contenido de cada ToolMessage sustituido por "-"
    La diferencia A - B son los tokens del contenido de los ToolMessage
    (menos 1 token por el "-" que dejamos para que el mensaje no esté vacío).

    Args:
        model: el ChatAnthropic (por ejemplo searchagent.create_model()).
        tools: las mismas tools que usa el agente (la API exige definirlas si
            la conversación tiene tool_calls).
        system_prompt: el system prompt ya relleno (build_system_prompt()).
        result: lo que devuelve agent.invoke(...).
    """
    messages = list(result["messages"])
    # Quitamos la respuesta final: solo nos interesa hasta los ToolMessage.
    if messages and isinstance(messages[-1], AIMessage) and not messages[-1].tool_calls:
        messages = messages[:-1]

    system = SystemMessage(content=system_prompt)

    with_results = model.get_num_tokens_from_messages([system] + messages, tools=tools)

    emptied = [
        ToolMessage(content="-", tool_call_id=m.tool_call_id) if isinstance(m, ToolMessage) else m
        for m in messages
    ]
    without_results = model.get_num_tokens_from_messages([system] + emptied, tools=tools)

    return with_results - without_results


def count_tool_definition_tokens(model, tools: list) -> int:
    """¿Cuántos tokens cuesta tener las tools "enchufadas" con bind_tools()?

    Contamos el mismo mensaje con y sin tools. La diferencia incluye:
      1) el JSON de cada tool (nombre + descripción + parámetros), y
      2) un system prompt OCULTO que Anthropic añade siempre que hay tools
         (en Claude Haiku 4.5 son 496 tokens con tool_choice "auto",
         según la documentación de Anthropic).
    Este coste se paga en TODAS las llamadas, aunque el agente no busque.
    """
    messages = [HumanMessage(content="hola")]
    with_tools = model.get_num_tokens_from_messages(messages, tools=tools)
    without_tools = model.get_num_tokens_from_messages(messages)
    return with_tools - without_tools


def count_text_tokens(model, text: str) -> int:
    """Tokens aproximados de un texto suelto (p.ej. un snippet o una página).

    Restamos un mensaje mínimo para quitar los pocos tokens de "envoltorio"
    que añade la API a cualquier mensaje.
    """
    with_text = model.get_num_tokens_from_messages([HumanMessage(content=text)])
    minimal = model.get_num_tokens_from_messages([HumanMessage(content=".")])
    return with_text - minimal

# Websearch Agent · Evaluación de Tavily como herramienta de búsqueda

Agente de búsqueda web hecho con **LangGraph**, **Claude Haiku 4.5** y **Tavily**.

## Planteamiento

En la empresa quieren un chatbot que responda consultas usando internet. El objetivo de este proyecto es **evaluar Tavily** como herramienta de búsqueda (*search tool*) de ese chatbot:

1. **Paso 1. Comportamiento y tokens:** ¿el agente busca solo cuando hace falta? ¿Cuántos tokens gasta y cuánto pesan los resultados de búsqueda (`ToolMessage`)?
2. **Paso 2. Entender Tavily:** ¿qué ve el LLM?, ¿qué devuelve Tavily?, ¿de dónde salen los snippets y el score?, ¿son relevantes?, ¿el modelo usa los resultados o alucina?
3. **Paso 3. Mejoras:** reducir el coste y mejorar la relevancia, tanto con los parámetros de la tool como con cambios en el agente.

## Estructura del proyecto

```
├── notebooks/
│   ├── 01_paso1_comportamiento_y_tokens.ipynb   ← Paso 1
│   ├── 02_paso2_analisis_tavily.ipynb           ← Paso 2
│   ├── 03_paso3_mejoras.ipynb                   ← Paso 3
│   └── testing_agent.ipynb                      ← primera exploración (con resultados)
└── src/espadas_langgraph_websearchagent/
    ├── searchagent.py      ← el agente (grafo LangGraph) y funciones para leer los mensajes
    ├── tokens.py           ← cálculo de tokens, peso de los ToolMessage y coste
    ├── tavily_api.py       ← llamada directa a la API REST de Tavily
    ├── tavily_analysis.py  ← Paso 2: schema, snippets, score, alucinaciones
    ├── judge.py            ← LLM juez con esquemas Pydantic (relevancia y grounding)
    ├── improvements.py     ← Paso 3: tool ligera, filtros, prompt v2, caché, variantes
    ├── evaluation.py       ← pasa la batería de preguntas a un agente y resume métricas
    └── questions.py        ← preguntas de prueba
```

## Cómo ejecutarlo

```bash
uv sync                      # instala las dependencias en .venv
cp .env.example .env         # y rellena ANTHROPIC_API_KEY y TAVILY_API_KEY
```

Abre los notebooks con el kernel de `.venv` y ejecútalos en orden (01 → 02 → 03).

> Ejecutar los notebooks gasta créditos de Anthropic y de Tavily. El Paso 3 es el más caro: 7 variantes × 9 preguntas. Con Haiku 4.5 son céntimos de dólar y unos 40 créditos de Tavily.

---

## Paso 1 · Comportamiento y tokens

### El agente

```
START → llm_call ──(¿pide tool?)── sí ──→ tool_node ──┐
            ▲  │                                      │
            │  no → END                               │
            └─────────────────────────────────────────┘
```

La **fecha actual** y la **fecha de corte** del modelo van en el *system prompt*, no en una tool `get_date()`. Una tool supondría una llamada extra al LLM y más tokens fijos en todas las llamadas; en el prompt, la fecha cuesta unos 10 tokens.

### ¿Cómo se calculan los tokens de los ToolMessage?

La API de Anthropic solo nos da `input_tokens` y `output_tokens` por llamada, pero **reenvía la conversación completa en cada llamada**. Cuando el agente busca hay dos llamadas:

```
Llamada 1:  ENTRA  system + tools + pregunta                               → input_1
            SALE   AIMessage con la tool_call                              → output_1
Llamada 2:  ENTRA  system + tools + pregunta + tool_call + ToolMessage     → input_2
```

Por tanto:

```
input_2 = input_1 + output_1 + tokens_ToolMessage
⇒ tokens_ToolMessage = input_2 − input_1 − output_1
```

- La **tool_call** (lo que escribe el LLM: la query) son tokens de **salida** de la llamada 1. Son pocos, pero más caros (5 $/M).
- El **ToolMessage** (lo que devuelve Tavily) son tokens de **entrada** de la llamada 2. Son muchos y más baratos (1 $/M).
- Si hay búsquedas en paralelo, la resta da el total de todas ellas.
- En un chatbot con historial, el ToolMessage **se vuelve a pagar en cada pregunta siguiente**.

Lo comprobamos con la API `count_tokens` de Anthropic (gratuita): contamos la conversación con y sin el contenido del ToolMessage y restamos.

### Resultados (primera ejecución, `testing_agent.ipynb`)

| Pregunta | ¿Debía buscar? | ¿Buscó? | Tokens totales | Tokens ToolMessage |
|---|---|---|---:|---:|
| ¿Dónde está la arteria aorta? | no | no ✅ | 2.733 | 0 |
| ¿Quién escribió el Quijote? | no | no ✅ | 2.434 | 0 |
| ¿Qué es la fotosíntesis? | no | no ✅ | 2.822 | 0 |
| ¿Quién ganó el Mundial de 2022? | no | no ✅ | 2.447 | 0 |
| ¿Quién ganó el Mundial de 2026? | **sí** | no ❌ | 2.476 | 0 |
| ¿Qué tiempo hace hoy en Sevilla? | **sí** | no ❌ | 2.503 | 0 |
| Noticias de hoy en la Comunidad de Madrid | sí | sí ✅ | 8.435 | 3.022 (36%) |
| ¿Cuál es la última versión de Python? | sí | sí ✅ | 8.152 | 3.002 (37%) |
| Busca en internet qué es LangGraph | sí | sí ✅ | 7.873 | 2.487 (32%) |

**7/9 decisiones correctas.** Cuando busca, los ToolMessage son ~**un tercio** del total. En la pregunta de la Honda Rebel, el LLM lanzó 2 búsquedas `advanced` en paralelo: 4.741 tokens de ToolMessage (**45%**).

---

## Paso 2 · Entender Tavily

| Pregunta | Respuesta |
|---|---|
| **¿Qué ve el LLM con `bind_tools()`?** | Un JSON con `name`, `description` e `input_schema`. Tiene **9 parámetros** (`query`, `include_domains`, `exclude_domains`, `search_depth`, `include_images`, `time_range`, `topic`, `start_date`, `end_date`) con descripciones muy largas. A eso se suma un **system prompt oculto de 496 tokens** que Anthropic añade cuando hay tools (Haiku 4.5). Es la mayor parte de los ~2.000 tokens fijos de cada llamada. |
| **¿Qué devuelve Tavily?** | `{query, follow_up_questions, answer, images, results[{title, url, content, score, raw_content}], response_time, request_id}`. El agente base lo mete en el ToolMessage con `str(dict)`, incluidas las claves vacías. |
| **¿Cómo se generan los snippets?** | Tavily parte la página en trozos de ~500 caracteres (`basic`) u ~800 (`advanced`), los ordena por parecido con la **query del LLM** (no con la pregunta del usuario) y devuelve los `chunks_per_source` mejores (3 por defecto) unidos con `[...]`. |
| **¿Son la parte relevante?** | Lo medimos comparando el snippet con la página completa (`include_raw_content`). Buscamos casos en los que la respuesta está en la página pero **no** en el snippet. |
| **¿Cómo se genera el score?** | Tavily solo dice que mide la *"relevancia entre query y contenido"*. La fórmula no es pública. Cambia con la redacción de la query y no es una probabilidad: sirve para ordenar dentro de una búsqueda. |
| **¿Score = relevancia real?** | Lo comparamos con la nota (0-10) de un LLM juez con esquema Pydantic (`relevancia`, `explicacion`) y calculamos la correlación. |
| **¿Usa los resultados o alucina?** | (1) Comprobación automática de **URLs inventadas** (citadas pero no devueltas por Tavily). (2) **LLM juez de grounding**: afirmaciones respaldadas y no respaldadas. |

Los resultados numéricos están en el notebook `02_paso2_analisis_tavily.ipynb`.

---

## Paso 3 · Mejoras

### A nivel de la tool

| Parámetro | Efecto |
|---|---|
| `max_results` 5 → 3 | menos tokens en el ToolMessage |
| `chunks_per_source` 3 → 2 | snippets más cortos (con 1 puede perderse la respuesta) |
| `search_depth="basic"` fijo | el LLM no puede elegir `advanced` (2 créditos) |
| `country="spain"` | prioriza resultados de España |
| `include_answer` | respuesta resumida de Tavily, útil para preguntas simples |

### A nivel del agente (variantes acumulativas)

| Variante | Qué añade |
|---|---|
| `0_base` | agente original |
| `1_prompt_v2` | prompt que deja claro que **sí** tiene internet y que debe buscar cualquier evento posterior a la fecha de corte |
| `2_params_tool` | 3 resultados, `search_depth` fijo, sin búsquedas en paralelo |
| `3_tool_ligera` | tool propia con **solo `query`** y una frase de descripción, para gastar menos tokens fijos en cada llamada |
| `4_formato_compacto` | título + URL + snippet en vez de `str(dict)`, y mensaje corto si la búsqueda no da resultados |
| `5_filtro_score` | quita resultados con score < 0,5 |
| `6_juez_llm` | un LLM juez (Pydantic: `relevancia`, `explicacion`) filtra los irrelevantes |

Además se prueba el **prompt caching** en una conversación de 4 turnos. En Claude Haiku 4.5 el prefijo mínimo cacheable es de **4.096 tokens**. Una pregunta suelta (~2.500 tokens) no llega, así que la caché solo ayuda en conversaciones largas (chatbot).

**No inyectar búsquedas infructuosas:** la API de Anthropic exige un `ToolMessage` por cada `tool_call`, así que no se puede "no enviar nada". Se envía un mensaje corto en vez del resultado vacío.

### Resultados

> Rellenar con la tabla `print_summaries(...)` del notebook 03.

| variante | aciertos | tokens medios | tokens fijos | ToolMessage medio | coste total |
|---|---|---:|---:|---:|---:|
| 0_base | 7/9 | | | | |
| 1_prompt_v2 | | | | | |
| 2_params_tool | | | | | |
| 3_tool_ligera | | | | | |
| 4_formato_compacto | | | | | |
| 5_filtro_score | | | | | |
| 6_juez_llm | | | | | |

---

## Conclusiones

1. **El mayor coste no son solo los resultados, sino el coste fijo.** La definición de la tool de Tavily más el prompt oculto de Anthropic suponen ~2.000 tokens en **cada** llamada, y se pagan dos veces cuando el agente busca. Una tool ligera con un solo parámetro es probablemente la mejora más rentable para un chatbot.
2. **Los ToolMessage pesan ~1/3 del total** cuando hay búsqueda, y en un chatbot se vuelven a pagar en cada turno. Formato compacto, menos resultados y menos trozos por página reducen ese peso.
3. **El snippet depende de la query que escribe el LLM, y el score mide parecido con esa query**, no si responde a la pregunta del usuario. Filtrar por score ayuda, pero hay que elegir el umbral con datos (comparándolo con un juez).
4. **El prompt importa para la relevancia:** el agente base falló justo en los casos en los que debía buscar (Mundial 2026, el tiempo). Cuando no busca, es cuando más riesgo hay de que se invente la respuesta.
5. **El prompt caching no ayuda con preguntas sueltas en Haiku 4.5** (mínimo de 4.096 tokens), pero sí en conversaciones largas.
6. **El LLM juez mejora la relevancia, pero añade una llamada.** Solo compensa si ahorra más tokens de los que gasta.

### Siguientes pasos

- Ampliar la batería de preguntas y revisar a mano las respuestas, porque el juez es el mismo modelo que el agente.
- Probar modelos más nuevos de Anthropic (más baratos y con un mínimo de caché más bajo). Habría que actualizar la fecha de corte del system prompt y repetir la evaluación.
- Limitar el número de vueltas del agente (máximo de búsquedas por pregunta).

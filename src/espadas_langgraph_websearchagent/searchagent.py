import operator
from datetime import date
from typing import Annotated, Literal

from IPython.display import Image, display
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

KNOWLEDGE_CUTOFF_HAIKU = "July 2025"
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

# Definition of the agent
def create_agent_with_search_tool():
    # Step 1: Define tools and model
    model = init_chat_model(
        "claude-haiku-4-5-20251001",
        temperature=0
    )


    # Define tools
    search_tool = TavilySearch(
        max_results=5,
        topic="general",
        # include_answer=False,
        # include_raw_content=False,
        # include_images=False,
        # include_image_descriptions=False,
        # search_depth="basic",
        # time_range="day",
        # include_domains=None,
        # exclude_domains=None
    )


    # Augment the LLM with tools
    tools = [search_tool]
    tools_by_name = {tool.name: tool for tool in tools}
    model_with_tools = model.bind_tools(tools)

    
    # Step 2: Define state
    class MessagesState(TypedDict):
        messages: Annotated[list[AnyMessage], operator.add]
        llm_calls: int

    # Step 3: Define model node
    def llm_call(state: MessagesState):
        """LLM decides whether to answer directly or search the web"""

        system = SystemMessage(
            content=SYSTEM_PROMPT.format(
                today=date.today().isoformat(), cutoff=KNOWLEDGE_CUTOFF_HAIKU
            )
        )

        return {
            "messages": [model_with_tools.invoke([system] + state["messages"])],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    # Step 4: Define tool node
    def tool_node(state: MessagesState):
        """Performs the tool call"""

        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        return {"messages": result}

    # Step 5: Define logic to determine whether to end
    # Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
    def should_continue(state: MessagesState) -> Literal["tool_node", END]:
        """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

        messages = state["messages"]
        last_message = messages[-1]

        # If the LLM makes a tool call, then perform an action
        if last_message.tool_calls:
            return "tool_node"

        # Otherwise, we stop (reply to the user)
        return END

    # Step 6: Build agent
    # Build workflow
    agent_builder = StateGraph(MessagesState)

    # Add nodes
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)

    # Add edges to connect nodes
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        ["tool_node", END]
    )
    agent_builder.add_edge("tool_node", "llm_call")

    # Compile the agent
    agent = agent_builder.compile()
    return agent


# Show token usage by the agent
def get_token_usage(result):
    # Counter
    input_tokens, output_tokens, tool_call_tokens, tool_result_tokens = 0, 0, 0, 0

    # Previous message from Haiku
    previous_message = None

    for message in result["messages"]:
        # Only for AIMessage, not Human nor Tool
        if not isinstance(message, AIMessage):
            continue

        # Haiku tokens
        tokens_in = message.usage_metadata["input_tokens"]
        tokens_out = message.usage_metadata["output_tokens"]

        # Sum tokens
        input_tokens += tokens_in
        output_tokens += tokens_out

        # Tool call tokens
        if message.tool_calls:
            tool_call_tokens += tokens_out

        # Tavily result tokens (input growth after a tool call)
        if previous_message is not None and previous_message.tool_calls:
            previous_in = previous_message.usage_metadata["input_tokens"]
            previous_out = previous_message.usage_metadata["output_tokens"]
            tavily_tokens = tokens_in - previous_in - previous_out
            tool_result_tokens += tavily_tokens

        previous_message = message

    total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "tool_call_tokens": tool_call_tokens,
        "tool_result_tokens": tool_result_tokens,
    }


# Print token usage in a readable format
def print_token_usage(usage):
    print("===== Token usage =====")
    print(f"Input tokens:       {usage['input_tokens']:>9,}")
    print(f"Output tokens:      {usage['output_tokens']:>9,}")
    print(f"Total tokens:       {usage['total_tokens']:>9,}")
    print()

    if usage["tool_call_tokens"] > 0:
        percentage = usage["tool_result_tokens"] / usage["total_tokens"] * 100

        print("Web search: yes")
        print(f"  Tool call tokens:   {usage['tool_call_tokens']:>7,}")
        print(f"  Tool result tokens: {usage['tool_result_tokens']:>7,}  ({percentage:.0f}% of total)")
    else:
        print("Web search: no")


# Run the agent
def run_agent_with_search_tool(agent, question):
    # Show the agent
    display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

    # Invoke
    messages = [HumanMessage(content=question)]
    messages = agent.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()

    # Token usage
    usage = get_token_usage(messages)
    print_token_usage(usage)
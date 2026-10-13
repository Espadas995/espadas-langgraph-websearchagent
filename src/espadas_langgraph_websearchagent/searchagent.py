import operator
from typing import Annotated, Literal

from IPython.display import Image, display
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


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
        """LLM decides whether to call a tool or not"""

        return {
            "messages": [
                model_with_tools.invoke(
                    [
                        SystemMessage(
                            content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                        )
                    ]
                    + state["messages"]
                )
            ],
            "llm_calls": state.get('llm_calls', 0) + 1
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


def run_agent_with_search_tool(agent, question):
    # Show the agent
    display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

    # Invoke
    messages = [HumanMessage(content=question)]
    messages = agent.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()
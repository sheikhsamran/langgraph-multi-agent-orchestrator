from typing import TypedDict, Annotated
import os
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import traceable

checkpointer = InMemorySaver()


# ---------- State ----------
class State(TypedDict):
    messages: Annotated[list, add_messages]
    next: str  # which agent the orchestrator picked


# ---------- Tools ----------
@tool
def get_stock_price(symbol: str) -> float:
    '''Return the current price of a stock given the stock symbol
    :param symbol: stock symbol
    :return: current price of the stock
    '''
    return {
        "MSFT": 200.3,
        "AAPL": 100.4,
        "AMZN": 150.0,
        "RIL": 87.6,
    }.get(symbol, 0.0)


stock_tools = [get_stock_price]

# ---------- Model ----------
llm = init_chat_model(
    "openai/gpt-oss-120b",
    model_provider="openai",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
stock_llm = llm.bind_tools(stock_tools)

AGENTS = ["stock_agent", "general_agent"]

ROUTER_PROMPT = SystemMessage(
    content=(
        "You are an orchestrator. Read the user's latest message and pick the "
        "best agent to handle it.\n"
        "- stock_agent: questions about stock prices or shares.\n"
        "- general_agent: everything else.\n"
        "Reply with ONLY one word: stock_agent or general_agent."
    )
)


# ---------- Nodes ----------
def orchestrator(state: State) -> dict:
    """Manager node: only decides who works next, does not answer."""
    last_user_message = state["messages"][-1]
    decision = llm.invoke([ROUTER_PROMPT, last_user_message]).content.strip().lower()
    next_agent = "stock_agent" if "stock_agent" in decision else "general_agent"
    return {"next": next_agent}


def route_from_orchestrator(state: State) -> str:
    return state["next"]


def stock_agent(state: State) -> dict:
    """Specialist: can call the stock price tool."""
    return {"messages": [stock_llm.invoke(state["messages"])]}


def general_agent(state: State) -> dict:
    """Specialist: plain chat, no tools."""
    return {"messages": [llm.invoke(state["messages"])]}


# ---------- Graph ----------
builder = StateGraph(State)

builder.add_node("orchestrator", orchestrator)
builder.add_node("stock_agent", stock_agent)
builder.add_node("stock_tools", ToolNode(stock_tools))
builder.add_node("general_agent", general_agent)

# Start -> orchestrator
builder.add_edge(START, "orchestrator")

# Orchestrator -> chosen specialist
builder.add_conditional_edges(
    "orchestrator",
    route_from_orchestrator,
    {name: name for name in AGENTS},
)

# Stock agent loop: agent -> tools -> agent, until no more tool calls
builder.add_conditional_edges(
    "stock_agent",
    tools_condition,
    {"tools": "stock_tools", END: END},
)
builder.add_edge("stock_tools", "stock_agent")

# General agent finishes directly
builder.add_edge("general_agent", END)

graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}


@traceable
def call_graph(query: str):
    result = graph.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config,
    )
    return result["messages"][-1].content


while True:
    in_message = input("You: ")
    if in_message.lower() in {"quit", "exit"}:
        break
    print("Bot:", call_graph(in_message))
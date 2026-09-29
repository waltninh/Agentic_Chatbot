from langgraph.graph import START, END, StateGraph
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv
import sqlite3 # mở kết nối tới database đọc ghi dữ liệu bằng SQL 
from langgraph.checkpoint.sqlite import SqliteSaver # biến checkpoint thành data để lưu vào SQLite và đọc ngược lại 

from langgraph.prebuilt import ToolNode, tools_condition #1 run tool then give back result,look at AI if want to run tool
from langchain_tavily import TavilySearch
from langchain_core.tools import tool #Decorator @tool 

import requests #http API
import math
import os 
from typing import Any # chú thích kiểu dữ liệu 


load_dotenv()

llm = ChatOpenRouter(
    model = "deepseek/deepseek-v4.1-flash",
    temperature = 0.7
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced",
)

@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations.
    Input should be a valid math expression.
    Example: 2 + 2, math.sqrt(16), 10 * 5
    """

    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }

        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL.
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    r = requests.get(url)
    return r.json()

@tool
def get_current_weather(city: str) -> dict:
    """
    Get the current weather for a given city (e.g. 'Hanoi', 'London', 'Tokyo')
    using Weatherstack. Returns temperature (°C), weather description,
    humidity, wind speed and feels-like temperature.
    """
    api_key = os.getenv("WEATHERSTACK_API_KEY")
    url = f"http://api.weatherstack.com/current?access_key={api_key}&query={city}"
    r = requests.get(url)
    return r.json()

tools = [search_tool, calculator, get_stock_price, get_current_weather]

llm_with_tools = llm.bind_tools(tools)


tool_node = ToolNode(tools) 

def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}



conn = sqlite3.connect(database='chatbot.db',
                       check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

# If the LLM asked for a tool, go to ToolNode; else finish
graph.add_conditional_edges("chat_node", tools_condition)

graph.add_edge("tools", "chat_node")

 
chatbot = graph.compile(checkpointer=checkpointer)

from langgraph.graph import START, END, StateGraph
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatOpenRouter(
    model = "deepseek/deepseek-v4.1-flash",
    temperature = 0.7,
    max_tokens = 10000
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState) -> dict:
    messages = state["messages"]           # ca lich su, khong chi tin cuoi
    response = llm.invoke(messages)
    return {"messages": [response]}


checkpointer = MemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
 
chatbot = graph.compile(checkpointer=checkpointer)

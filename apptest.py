from chat_bot_v1 import chatbot
from langchain_core.messages import BaseMessage, HumanMessage

config = {"configurable": {"thread_id": "2004"}}

for chunk, metadata in chatbot.stream(
    {"messages": [HumanMessage(content=user_message)]},
    config=config,
    stream_mode="messages",
):
    if chunk.content:
        print(chunk.content, end="", flush=True)
print()

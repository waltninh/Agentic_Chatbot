import uuid

from chat_bot_v1 import chatbot
from langchain_core.messages import AIMessageChunk, HumanMessage
import streamlit as st

st.set_page_config(page_title="Agentic chatbot", page_icon=":material/smart_toy:")

# moi phien trinh duyet co 1 thread_id rieng
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

with st.sidebar:
    if st.button("Cuộc trò chuyện mới", icon=":material/add:", width="stretch"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

config = {"configurable": {"thread_id": st.session_state.thread_id}}

st.title("Agentic chatbot")

# hien lai lich su da luu trong checkpointer (MemorySaver)
for message in chatbot.get_state(config).values.get("messages", []):
    role = "user" if message.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)


def stream_reply(user_input):
    # tra ve tung doan chu cua AI de hien dan ra nhu ChatGPT
    for chunk, _ in chatbot.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessageChunk) and chunk.content:
            yield chunk.content


if user_input := st.chat_input("Nhập tin nhắn...", submit_mode="disable"):
    # hien tin nhan cua nguoi dung
    with st.chat_message("user"):
        st.markdown(user_input)

    # goi chatbot va hien cau tra loi
    with st.chat_message("assistant"):
        st.write_stream(stream_reply(user_input))

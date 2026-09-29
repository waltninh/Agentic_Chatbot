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

def show_tool_call(name, args, result):
    # 1 buoc "da dung tool" co the bam vao de xem dau vao / ket qua
    with st.status(f"Đã dùng tool **{name}**", type="step", state="complete"):
        st.markdown("**Đầu vào**")
        st.json(args)
        st.markdown("**Kết quả**")
        st.code(str(result)[:2000])


# hien lai lich su da luu trong checkpointer (SqliteSaver)
history = chatbot.get_state(config).values.get("messages", [])
tool_results = {m.tool_call_id: m.content for m in history if m.type == "tool"}
bubble = None
for message in history:
    if message.type == "human":
        bubble = None
        with st.chat_message("user"):
            st.markdown(message.content)
    elif message.type == "ai":
        # gom cac buoc tool + cau tra loi vao chung 1 khung assistant
        if bubble is None:
            bubble = st.chat_message("assistant")
        with bubble:
            for call in message.tool_calls:
                show_tool_call(call["name"], call["args"], tool_results.get(call["id"], ""))
            if message.content:
                st.markdown(message.content)


if user_input := st.chat_input("Nhập tin nhắn...", submit_mode="disable"):
    # hien tin nhan cua nguoi dung
    with st.chat_message("user"):
        st.markdown(user_input)

    # goi chatbot: "messages" de hien tung chu, "updates" de biet AI goi tool nao
    with st.chat_message("assistant"):
        text, text_box, running = "", None, {}
        for mode, data in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                chunk, _ = data
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    if text_box is None:
                        text_box = st.empty()
                    text += chunk.content
                    text_box.markdown(text)
                continue

            for node, update in data.items():
                for m in (update or {}).get("messages", []):
                    if m.type == "ai" and m.tool_calls:
                        # AI vua yeu cau goi tool -> hien "dang dung tool"
                        for call in m.tool_calls:
                            running[call["id"]] = st.status(
                                f":shimmer[Đang dùng tool **{call['name']}**]", type="step"
                            )
                            with running[call["id"]]:
                                st.markdown("**Đầu vào**")
                                st.json(call["args"])
                        text, text_box = "", None
                    elif m.type == "tool":
                        # tool chay xong -> them ket qua, doi sang "da dung"
                        status = running.pop(m.tool_call_id)
                        with status:
                            st.markdown("**Kết quả**")
                            st.code(str(m.content)[:2000])
                        status.update(label=f"Đã dùng tool **{m.name}**", state="complete")


def list_threads():
    # lay tat ca thread_id da luu trong MemorySaver, moi nhat len dau
    last_update = {}
    for cp in chatbot.checkpointer.list(None):
        tid = cp.config["configurable"]["thread_id"]
        last_update[tid] = max(last_update.get(tid, ""), cp.checkpoint["ts"])
    return sorted(last_update, key=last_update.get, reverse=True)


def switch_thread(tid):
    st.session_state.thread_id = tid


# danh sach cuoc hoi thoai cu (dat cuoi file de thay luon chat vua gui)
with st.sidebar:
    st.subheader("Lịch sử chat")
    for tid in list_threads():
        messages = chatbot.get_state({"configurable": {"thread_id": tid}}).values.get("messages", [])
        if not messages:
            continue
        # lay cau hoi dau tien lam tieu de
        title = messages[0].content[:40]
        st.button(
            title,
            key=tid,
            on_click=switch_thread,
            args=(tid,),
            type="primary" if tid == st.session_state.thread_id else "tertiary",
            width="stretch",
        )

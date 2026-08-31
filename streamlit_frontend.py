import streamlit as st
from langgraph_backend import chatbot, retrieve_all_threads, get_conversation_state, iter_async
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def extract_text(content):
    """Gemini returns content as a list of blocks (text + extras/signature).
    Pull out only the human-readable text, dropping metadata like
    thought signatures that shouldn't be shown to the user."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""

def is_tool_call_chunk(message_chunk):
    """True if this AIMessage chunk is the model deciding to call a tool
    (not the final answer) — these often carry throwaway/narration text."""
    return bool(
        getattr(message_chunk, "tool_calls", None)
        or getattr(message_chunk, "tool_call_chunks", None)
    )

def load_conversation(thread_id):
    # ✅ FIX: use the async-safe wrapper instead of chatbot.get_state (sync)
    state = get_conversation_state(thread_id)
    messages = state.values.get("messages", [])
    # ✅ FIX: Filter out ToolMessages
    return [msg for msg in messages if not isinstance(msg, ToolMessage)]

# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])

# ============================ Sidebar ============================
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")
for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            # ✅ FIX: Skip ToolMessages
            if isinstance(msg, ToolMessage):
                continue
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": extract_text(msg.content)})
        st.session_state["message_history"] = temp_messages

# ============================ Main UI ============================

# Render history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    # Show user's message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # Assistant streaming block
    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            # chatbot.astream(...) is an async generator (required now that
            # MCP tools + the checkpointer are async-only). iter_async drives
            # it on the backend's dedicated event-loop thread and yields
            # items synchronously so st.write_stream can consume it as usual.
            async def _astream():
                async for message_chunk, metadata in chatbot.astream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="messages",
                ):
                    yield message_chunk, metadata

            for message_chunk, metadata in iter_async(_astream()):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )
                    continue
                if isinstance(message_chunk, AIMessage):
                    if is_tool_call_chunk(message_chunk):
                        continue

                    text = extract_text(message_chunk.content)
                    if text:
                        yield text

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
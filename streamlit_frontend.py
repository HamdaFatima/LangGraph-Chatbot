import streamlit as st
from langgraph_backend import (
    chatbot,
    retrieve_all_threads,
    get_conversation_state,
    iter_async,
    ingest_pdf,
    thread_document_metadata,
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid


# =========================== Utilities ===========================
def generate_thread_id():
    # Kept as a string so it matches what the backend/checkpointer returns
    # from retrieve_all_threads and what ingest_pdf expects.
    return str(uuid.uuid4())


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
    state = get_conversation_state(thread_id)
    messages = state.values.get("messages", [])
    # Filter out ToolMessages
    return [msg for msg in messages if not isinstance(msg, ToolMessage)]


def get_thread_title(thread_id):
    """Short human-readable label for a thread, based on its first user
    message. Cached in session_state so we don't re-fetch from the backend
    on every rerun — but only once we have a REAL title, so a thread that's
    briefly empty doesn't get stuck showing 'New chat' forever."""
    if thread_id in st.session_state["thread_titles"]:
        return st.session_state["thread_titles"][thread_id]

    for msg in load_conversation(thread_id):
        if isinstance(msg, HumanMessage):
            text = extract_text(msg.content).strip()
            if text:
                title = text[:40] + ("…" if len(text) > 40 else "")
                st.session_state["thread_titles"][thread_id] = title
                return title
            break

    # No real message found yet — don't cache, so we try again next rerun.
    return "New chat"


# ======================= Session Initialization ===================
# A per-browser-session id so each user only sees their own threads in the
# sidebar. Stored in the URL's query params (not just session_state) so it
# survives page reloads and server restarts — session_state alone resets on
# every reload, which was wiping out access to previously saved threads.
if "user_id" not in st.session_state:
    if "uid" in st.query_params:
        st.session_state["user_id"] = st.query_params["uid"]
    else:
        new_id = str(uuid.uuid4())
        st.session_state["user_id"] = new_id
        st.query_params["uid"] = new_id

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = [
        str(t) for t in retrieve_all_threads(st.session_state["user_id"])
    ]

if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {}

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}  # {thread_key: {filename: summary}}

# ============================ Sidebar ============================
st.sidebar.title("LangGraph Chatbot")

# Variables the PDF panel needs
thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})

# Repopulate from the backend after a page reload / thread switch.
# Adjust this if thread_document_metadata returns a different shape.
if not thread_docs:
    try:
        meta = thread_document_metadata(thread_key)
        if isinstance(meta, dict) and meta:
            if "filename" in meta:
                thread_docs[meta["filename"]] = meta
            else:
                # Assume {filename: summary} shape
                thread_docs.update(meta)
    except Exception:
        pass

if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()  # refresh so the PDF panel matches the new thread

# ---------- PDF upload ----------
st.sidebar.header("Document")

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF for this chat",
    type=["pdf"],
    key=f"pdf_uploader_{thread_key}",  # per-thread key so files don't leak across chats
)
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            try:
                summary = ingest_pdf(
                    uploaded_pdf.getvalue(),
                    thread_id=thread_key,
                    filename=uploaded_pdf.name,
                )
            except Exception as e:
                status_box.update(label="❌ Indexing failed", state="error", expanded=True)
                st.error(f"Could not index PDF: {e}")
                st.stop()
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)
        st.rerun()  # show the green "Using ..." box immediately

# ---------- Conversations ----------
st.sidebar.header("My Conversations")
for thread_id in st.session_state["chat_threads"][::-1]:
    label = get_thread_title(thread_id)
    if st.sidebar.button(label, key=f"thread_btn_{thread_id}"):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                continue
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": extract_text(msg.content)})
        st.session_state["message_history"] = temp_messages
        st.rerun()

# ============================ Main UI ============================

# Render history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    # Only now does this thread earn a spot in the sidebar — and a title
    # derived from the actual first message, instead of showing up empty.
    current_thread_id = st.session_state["thread_id"]
    add_thread(current_thread_id)
    if current_thread_id not in st.session_state["thread_titles"]:
        title = user_input.strip()
        st.session_state["thread_titles"][current_thread_id] = (
            title[:40] + ("…" if len(title) > 40 else "")
        )

    # Show user's message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {
            "thread_id": thread_key,  # same string used for ingest_pdf
            "user_id": st.session_state["user_id"],
        },
        "metadata": {"thread_id": thread_key},
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
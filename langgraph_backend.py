from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Any, Optional, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool, BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import os
import aiosqlite
import requests
import asyncio
import threading
import tempfile

load_dotenv()

# =================== Dedicated async loop for backend tasks ===================
# Runs in its own background thread so Streamlit (sync, and possibly serving
# multiple sessions concurrently) can safely submit async work without each
# session fighting over a shared loop via run_until_complete.
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    """Submit a coroutine to the backend loop and block until it's done."""
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop without blocking."""
    return _submit_async(coro)


def iter_async(async_gen):
    """Bridge an async generator to a sync generator, driven by the backend's
    dedicated event-loop thread — needed because chatbot.astream() must run
    on the same loop the checkpointer's aiosqlite connection is bound to."""
    while True:
        try:
            item = submit_async_task(async_gen.__anext__()).result()
        except StopAsyncIteration:
            break
        yield item


# -------------------
# 1. LLM + embeddings
# -------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", max_retries=3)

# Gemini embeddings (uses the same GOOGLE_API_KEY as the chat model).
# Free tier allows ~100 embedding requests/min, which is fine for small PDFs.
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Alternative: local embeddings with no quota or API key
#   pip install langchain-huggingface sentence-transformers
# from langchain_huggingface import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a helpful assistant with access to tools: a web search tool, "
    "a stock price tool, a document retrieval tool (rag_tool), and any "
    "additional tools provided via MCP servers. "
    "For questions about current events, today's news, live prices, or "
    "anything time-sensitive, you MUST call the search tool rather than "
    "answering from memory or telling the user to look it up themselves. "
    "After the tool returns results, read them and give the user a direct, "
    "specific answer (e.g. actual headlines, actual numbers) synthesized "
    "from what the tool found — never just point them to a website instead "
    "of answering. "
    "If the user asks about an uploaded PDF or document (e.g. 'what is this "
    "pdf about', 'summarize the document', 'what skills are listed'), call "
    "rag_tool with a relevant search query and answer from the returned "
    "context. Do not ask the user to upload the file again unless rag_tool "
    "reports that no document is indexed. "
    "For general knowledge questions you can answer confidently and "
    "accurately on your own (e.g. capitals, historical facts, definitions), "
    "answer directly without calling any tool."
))

# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id is not None and str(thread_id) in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[str(thread_id)]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        summary = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = summary

        return summary
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 3. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    r = requests.get(url)
    return r.json()


@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """
    Retrieve relevant information from the PDF uploaded to the current chat.
    Use this for any question about the uploaded document (its content,
    summary, or details). Only pass the search query.
    """
    # thread_id comes from the run config (injected by LangGraph, hidden from
    # the model), so it always matches the id used in ingest_pdf.
    thread_id = (config or {}).get("configurable", {}).get("thread_id")
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    return {
        "query": query,
        "context": [doc.page_content for doc in result],
        "metadata": [doc.metadata for doc in result],
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


client = MultiServerMCPClient(
    {
        "arith": {
            "transport": "stdio",
            "command": "python3",
            "args": ["/Users/nitish/Desktop/mcp-math-server/main.py"],
        },
        "expense": {
            "transport": "streamable_http",  # if this fails, try "sse"
            "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
        }
    }
)


def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception:
        return []


mcp_tools = load_mcp_tools()
tools = [search_tool, get_stock_price, *mcp_tools, rag_tool]
llm_with_tools = llm.bind_tools(tools) if tools else llm


# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. Nodes
# -------------------
async def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SYSTEM_PROMPT] + messages

    try:
        response = await llm_with_tools.ainvoke(messages)
    except Exception as e:
        import traceback
        print("chat_node error:", repr(e))
        traceback.print_exc()
        response = AIMessage(
            content="Sorry, the model is temporarily unavailable. Please try again in a moment."
        )
    return {"messages": [response]}


tool_node = ToolNode(tools) if tools else None


# -------------------
# 6. Checkpointer
# -------------------
async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_init_checkpointer())

# -------------------
# 7. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


# -------------------
# 8. Helpers
# -------------------
def retrieve_all_threads(user_id: str):
    """Return thread_ids that belong to this user_id, plus any legacy
    threads created before user_id tagging existed (their config has no
    'user_id' key at all — treat those as visible to everyone rather than
    permanently hiding conversations that predate this feature).
    """
    async def _alist_threads():
        all_threads = set()
        async for checkpoint in checkpointer.alist(None):
            cfg = checkpoint.config["configurable"]
            owner = cfg.get("user_id")
            if owner is None or owner == user_id:
                all_threads.add(cfg["thread_id"])
        return list(all_threads)

    return run_async(_alist_threads())


def get_conversation_state(thread_id):
    """Sync-friendly wrapper around chatbot.aget_state for the frontend."""
    config = {"configurable": {"thread_id": thread_id}}
    return run_async(chatbot.aget_state(config=config))


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})
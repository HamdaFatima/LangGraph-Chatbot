# LangGraph Multi-Tool Chatbot

A conversational AI agent built with **LangGraph** and **Streamlit**, capable of using external tools (web search, stock prices, calculator) to answer questions, with persistent multi-thread conversation history backed by SQLite.

## Features

- **Tool-using agent**: Automatically decides when to call a calculator, a live stock price lookup (Alpha Vantage), or a web search (DuckDuckGo) based on the user's question
- **Streaming responses**: Answers stream token-by-token in the UI, with live status indicators showing which tool is currently running
- **Persistent conversations**: Chat history is checkpointed to SQLite via LangGraph's `SqliteSaver`, so conversations survive app restarts
- **Multi-thread support**: Start new conversations or revisit past ones from the sidebar, each with its own isolated message history
- **Gemini-powered**: Uses `gemini-2.5-flash` via `langchain-google-genai`

## Architecture
```
User input
│
▼
┌─────────────┐ tool call needed? ┌───────────┐
│ chat_node │ ───────────────────────────▶│ tools │
│ (Gemini LLM) │◀─────────────────────────── │ (ToolNode) │
└─────────────┘ tool result └───────────┘
│
▼ (no tool call needed)
Final answer
```

Built as a `StateGraph` with conditional routing (`tools_condition`) that loops between the LLM node and the tool-execution node until the model produces a final answer with no further tool calls.

**Tools available to the agent:**
| Tool | Purpose |
|---|---|
| `calculator` | Basic arithmetic (add, subtract, multiply, divide) |
| `get_stock_price` | Live stock quote lookup via Alpha Vantage API |
| `search_tool` | Web search via DuckDuckGo |

## Tech Stack

- [LangGraph](https://langchain-ai.github.io/langgraph/) — agent orchestration and state management
- [LangChain](https://python.langchain.com/) — tool binding and message handling
- [Streamlit](https://streamlit.io/) — chat UI
- [Google Gemini](https://ai.google.dev/) (`gemini-2.5-flash`) — the underlying LLM
- SQLite — conversation checkpointing

## Setup
```bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

Create a `.env` file with:
```
GEMINI_API_KEY=your_key_here
ALPHA_VANTAGE_API_KEY=your_key_here
```

Run:
```bash
streamlit run streamlit_frontend.py
```

## Usage
Ask anything — general questions are answered directly, while questions needing live data ("what's the price of AAPL?", "search for the latest AI news", "multiply 45 by 12") trigger the relevant tool automatically. A status indicator shows which tool is running in real time.

## Project Structure

```
.
├── streamlit_frontend.py   # Streamlit UI, streaming, thread management
├── langgraph_backend.py    # LangGraph agent definition, tools, graph compilation
├── requirements.txt
└── README.md
```
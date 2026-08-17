from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import os
import sqlite3
import requests

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=api_key)

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a helpful assistant with access to tools: calculator, "
    "get_stock_price, and a web search tool. "
    "Only use a tool when the question genuinely requires it — "
    "for example, real-time stock prices, current events, or arithmetic "
    "on numbers you don't already know. "
    "For general knowledge questions you can answer confidently and "
    "accurately on your own (e.g. capitals, historical facts, definitions), "
    "answer directly without calling any tool."
))

# Tools
search_tool = DuckDuckGoSearchRun(region='us-en')

@tool
def calculator(first_num: float, second_num: float, operation: str) -> float:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, subtract, multiply, divide
    """
    try:
        if operation == 'add':
            return first_num + second_num
        elif operation == 'subtract':
            return first_num - second_num
        elif operation == 'multiply':
            return first_num * second_num
        elif operation == 'divide':
            if second_num != 0:
                return first_num / second_num
            else:
                raise ValueError("Cannot divide by zero.")
        else:
            raise ValueError("Invalid operation. Please use 'add', 'subtract', 'multiply', or 'divide'.")
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a given symbol (e.g. 'AAPL','TSLA')
    using alpha Vantage API key in the URL"""
    # Fixed the f-string - removed nested quotes
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    r = requests.get(url)
    return r.json()

# Make tools list
tools = [get_stock_price, calculator, search_tool]

tool_node = ToolNode(tools)

# ✅ Use llm_with_tools (model with tools bound)
llm_with_tools = model.bind_tools(tools)

class chatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: chatState):
    messages = state['messages']
    
    # Prepend system prompt if not already present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SYSTEM_PROMPT] + messages

    response = llm_with_tools.invoke(messages)
    
    return {'messages': [response]}

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpoint = SqliteSaver(conn=conn)

graph = StateGraph(chatState)

graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chat_node')

# ✅ FIXED: Add the conditional edge mapping
graph.add_conditional_edges(
    'chat_node',
    tools_condition,
    {
        'tools': 'tools',    # If tools needed, go to 'tools' node
        '__end__': END       # If no tools needed, end
    }
)

graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpoint)

def retrieve_all_threads():
    all_threads = set()
    for item in checkpoint.list(None):
        all_threads.add(item.config['configurable']['thread_id'])
    return list(all_threads)
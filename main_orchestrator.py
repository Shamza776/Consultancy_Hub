import os
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
#from langchain_core.messages import SystemMessage
from src.specialist_tools import (
    legal_specialist_tool,
    hr_specialist_tool,
    it_specialist_tool,
    customer_success_tool
)

# 1. INITIALIZE THE LLM (THE BRAIN)
# Temperature=0 makes the AI factual and consistent, which is vital for Corporate RAG.
llm = ChatOllama(model="llama3.1", temperature=0)

# 2. DEFINE THE TOOLBOX
# These are the specialists the Manager can call.
tools = [
    legal_specialist_tool,
    hr_specialist_tool,
    it_specialist_tool,
    customer_success_tool
]

# 3. THE SYSTEM INSTRUCTIONS (THE MANAGER'S RULES)
# This is how the AI knows how to handle private data and when to use tools.
system_message = (
    "You are the 'Corporate Consultancy Hub Manager'. "
    "Your goal is to answer employee questions using your specialized departmental tools. "
    "\n\nRULES:\n"
    "1. If a question is about Legal, NDAs, or Compliance, use the 'legal_specialist_tool'.\n"
    "2. If a question is about HR or IT, use their respective mock tools for now.\n"
    "3. If the retrieved information does not answer the question, do not make up an answer. "
    "State that the internal database does not have that information.\n"
    "4. Always maintain a professional, corporate tone."
)

# 4. CREATE THE AGENT (THE ORCHESTRATOR)
# LangGraph handles the 'Reasoning' loop (deciding which tool to use).
agent_executor = create_react_agent(llm, tools, prompt=system_message)

def run_hub(user_query: str):
    """
    The main entry point for the UI.
    """
    print(f"--- Processing Query: {user_query} ---")
    
    # We pass the query as a list of messages
    inputs = {"messages": [("user", user_query)]}
    
    # The agent runs and returns the full conversation history
    response = agent_executor.invoke(inputs)
    
    # We only care about the very last message (the AI's final answer)
    return response["messages"][-1].content

if __name__ == "__main__":
    # Terminal Test
    test_query = "What is the policy for reporting a data breach?"
    print("\nAI RESPONSE:\n", run_hub(test_query))

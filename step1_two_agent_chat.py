# IMPORTS
import os
from typing import TypedDict, List, Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages

# 1. LOAD ENVIRONMENT VARIABLES
# Load API key from .env file I will create later
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set your OPENAI_API_KEY in a .env file")

# 2. DEFINE THE STATE
# The "whiteboard" for our conversation.
# Just a list of messages. LangGraph will pass this between nodes.
class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], add_messages]

# 3. INITIALIZE THE LLM
# We'll use a cheap and fast model. "gpt-3.5-turbo" is perfect for learning.
# If you prefer a local model, you can swap this out for Ollama later.
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# 4. DEFINE NODE FUNCTIONS
# Each node is a function that takes the current state and returns an update.

# Agent 1: The "Questioner"
def agent_1_node(state:AgentState):
    """Agent 1's job is to start the conversation or ask a question."""
    print("---Agent 1 is thinking...---")
    # We call the LLM. It will see the entire message history in the state.
    response = model.invoke(state["messages"])
    # We return an update to the state. We're adding the new AI messages to the list.
    # LangGraph's default reducer will append this to the existing 'messages' list.
    return {"messages": [response]}

# Agent 2: The "Responder"
def agent_2_node(state:AgentState):
    """Agent 2's job is to respond to Agent 1."""
    print("---Agent 2 is thinking...---")
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# 5. BUILD THE GRAPH
# This is where we define the workflow.

# Create a graph builder that uses our AgentState
workflow = StateGraph(AgentState)

# Add the nodes. We give each node a name and the function it should run.
workflow.add_node("agent_1", agent_1_node)
workflow.add_node("agent_2", agent_2_node)

# Define the edges: the path through the graph.
workflow.add_edge(START, "agent_1") # Start -> Agent 1
workflow.add_edge("agent_1", "agent_2") # Agent 1 -> Agent 2
workflow.add_edge("agent_2", END) # Agent 2 -> End

# Compile the graph into a runnable application.
app = workflow.compile()

# 6. RUN THE CONVERSATION
print("Starting the Multi-Agent Conversation ===\n")

# Initial input: a message from the user to kick things off.
initial_state = {
    "messages": [HumanMessage(content="Introduce yourself and say hello to the other agent.")]
}

# Run the graph
final_state = app.invoke(initial_state)

# Print the final conversation
print("\n=== Conversation Log ===")
for message in final_state["messages"]:
    if isinstance(message, HumanMessage):
        print(f"Human: {message.content}")
    elif isinstance(message, AIMessage):
        # This message could be from agent 1 or agent 2, but we'll just label it as AI for now.
        print(f"AI: {message.content}")

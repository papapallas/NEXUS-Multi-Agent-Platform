import os
from typing import List, Literal, Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from typing import TypedDict

# Setup
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Debate State
class DebateState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], add_messages]
    turn_count: int

# Agent 1 : Pro
def pro_agent(state: DebateState):
    """Pro argues in favor of the proposition."""
    print("--- Pro Agent is thinking ---")
    response = model.invoke([HumanMessage(content="You are the Pro side in a formal debate. " \
    "Argue strongly for the proposition. Do not act as moderator. " \
    "Directly rebut the last argument made by the Con side. Present a new strong argument or counterargument.")] + state["messages"])
    return {"messages": [response], "turn_count": state["turn_count"] + 1}

# Agent 2 : Con
def con_agent(state: DebateState):
    """Con argues against the proposition."""
    print("--- Con Agent is thinking. ---")
    response = model.invoke([HumanMessage(content="You are the CON side in a formal debate. " \
    "Argue strongly against the proposition. Do not act as moderator. " \
    "Directly rebut the last argument made by the Pro side. Present a new strong argument or counterargument. ")] + state["messages"])
    return {"messages": [response], "turn_count": state["turn_count"] + 1}

# Agent 3: Moderator
def moderator_node(state: DebateState):
    """Decides who speaks next or ends the debate."""
    print("---- Moderator is considering -----")
    return {}

# Routing function - this decides the next node based on turn_count
def route_after_moderator(state: DebateState):
    if state["turn_count"] >= 5:
        return END
    return "pro_agent" if state["turn_count"] % 2 == 0 else "con_agent"

# Build the graph
workflow = StateGraph(DebateState)

workflow.add_node("pro_agent", pro_agent)
workflow.add_node("con_agent", con_agent)
workflow.add_node("moderator", moderator_node)

workflow.add_edge(START, "moderator")                                   # start with moderator
workflow.add_conditional_edges("moderator", route_after_moderator)      # after moderator node, use the routing function to decide where to go
workflow.add_edge("pro_agent", "moderator")                             # after Pro, go back to moderator
workflow.add_edge("con_agent", "moderator")                             # after Con, go back to moderator

app = workflow.compile()

# Initial State
proposition = "Artificial intelligence will eventually replace most human jobs."
initial_state = {
    "messages": [HumanMessage(content=f"We are debating the proposition: '{proposition}'. Pro, please begin")],
    "turn_count": 0
}

# Run the debate
final_state = app.invoke(initial_state)

# Print the transcript
print("\n=== DEBATE TRANSCRIPT ===\n")
for msg in final_state["messages"]:
    speaker = "Human" if isinstance(msg, HumanMessage) else "AI"
    print("-" * 60)
    print(f"{speaker}:\n")
    print(msg.content)
    print()
import os
import json
from typing import List, Literal, Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from typing import TypedDict
from dataclasses import dataclass

# PERSONA DEFINITION
@dataclass
class Persona:
    name: str
    openness: float    #0-1
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float

    def describe(self) -> str:
        desc = f"Your name is {self.name}. "
        # Openness
        if self.openness > 0.7:
            desc += "You are highly open, curious, and imaginative. "
        elif self.openness < 0.3:
            desc += "You are conventional and prefer routine. "
        else:
            desc += "You are moderately open to new experiences. "
        # Conscientiousness
        if self.conscientiousness > 0.7:
            desc += "You are very organised, disciplined, and detail-oriented. "
        elif self.conscientiousness < 0.3:
            desc += "You are spontaneous and sometimes disorganised. "
        else:
            desc += "You have a balanced approach to order. "
        # Extraversion
        if self.extraversion > 0.7:
            desc += "You are outgoing, talkative, and energetic. "
        elif self.extraversion < 0.3:
            desc += "You are reserved and prefer solitude. "
        else:
            desc += "You are moderately sociable. "
        # Agreeableness
        if self.agreeableness > 0.7:
            desc += "You are cooperative, compassionate, and trusting. "
        elif self.agreeableness < 0.3:
            desc += "You are competitive and sceptical of others. "
        else:
            desc += "You tend to be cooperative but can be competitive. "
        # Neuroticism
        if self.neuroticism > 0.7:
            desc += "You are prone to stress, anxiety, and mood swings. "
        elif self.neuroticism < 0.3:
            desc += "You are emotionally stable and calm. "
        else:
            desc += "You have moderate emotional reactivity. "
        return desc

#SETUP
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Load personas from JSON
with open("personas.json", "r") as f:
    personas_data = json.load(f)
personas = [Persona(**p) for p in personas_data]

# Assign personas to Pro and Con
pro_persona = personas[0]
con_persona = personas[1]

# DEBATE STATE
class DebateState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], add_messages]
    turn_count: int

# AGENT NODES
# Agent 1 : Pro
def pro_agent(state: DebateState):
    """Pro argues in favor of the proposition."""
    print("--- Pro Agent is thinking ---")
    system_content = (
        f"You are the PRO side in a debate. {pro_persona.describe()}"
        "Argue strongly FOR the proposition. "
        "Directly rebut the last argument made by the Con side. "
        "Present a new strong argument or counterargument."
    )
    # Combine system prompt with conversation history
    messages = [SystemMessage(content=system_content)] + state["messages"]
    response = model.invoke(messages)
    return {"messages": [response], "turn_count": state["turn_count"] + 1}

# Agent 2 : Con
def con_agent(state: DebateState):
    """Con argues against the proposition."""
    system_content = (
        f"You are the CON side in a debate. {con_persona.describe()}"
        "Argue strongly AGAINST the proposition. "
        "Directly rebut the last argument made by the Pro side. "
        "Present a new strong argument or counterargument. "
    )
    # Combine system prompt with conversation history
    messages = [SystemMessage(content=system_content)] + state["messages"]
    response = model.invoke(messages)
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
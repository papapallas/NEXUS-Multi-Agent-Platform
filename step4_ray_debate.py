import ray, os, json, time
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dataclasses import dataclass

load_dotenv()

# Persona
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
    
# Ray Actors
@ray.remote
class ProAgent:
    def __init__(self, persona: Persona, model_name: str = "gpt-3.5-turbo"):
        self.persona = persona
        self.model = ChatOpenAI(
            model=model_name, 
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
            )
    
    def argue(self, conversation_history: List[Dict], proposition: str) -> str:
        """Take the conversation so far and return a new argument."""
        system_content = (
            f"You are the pro side in a debate. {self.persona.describe()} "
            f"The proposition is: '{proposition}'. "
            "Argue strongly FOR the proposition. "
            "Directly rebut the last argument made by the Con side. "
            "Present a new strong argument or counterargument. "
        )
        # Convert history to LangChain message objects
        messages = [SystemMessage(content=system_content)]
        for msg in conversation_history:
            if msg["role"] == "human":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        response = self.model.invoke(messages)
        return response.content

@ray.remote
class ConAgent:
    def __init__(self, persona: Persona, model_name: str = "gpt-3.5-turbo"):
        self.persona = persona
        self.model = ChatOpenAI(
            model=model_name, 
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
            )

    def argue(self, conversation_history: List[Dict], proposition: str) -> str:
        system_content = (
            f"You are the CON side in a debate. {self.persona.describe()} "
            f"The proposition is: '{proposition}'. "
            "Argue strongly AGAINST the proposition. "
            "Directly rebut the last argument made by the Pro side. "
            "Present a new strong argument or counterargument. "
        )
        messages = [SystemMessage(content=system_content)]
        for msg in conversation_history:
            if msg["role"] == "human":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        response = self.model.invoke(messages)
        return response.content
    
@ray.remote
class DebateSupervisor:
    def __init__(self):
        #You could store active debate handles here, but for simplicity we'll just run one debate per call
        pass

    def run_debate(self, proposition: str, pro_persona: Persona, con_persona: Persona, max_turns: int = 5):
        """Orchestrate a single debate using Pro and Con actors."""
        # Create actor instances for this debate
        pro = ProAgent.remote(pro_persona)
        con = ConAgent.remote(con_persona)

        # Initial Conversation
        history = [
            {"role": "human", "content": f"We are debating the proposition: '{proposition}'. Pro, please begin."}
        ]
        turn_count = 0

        while turn_count < max_turns:
            # Decide who speaks (alternating, starting with Pro)
            if turn_count % 2 == 0:
                # Call pro.argue remotely and wait for result
                response = ray.get(pro.argue.remote(history, proposition))
                role = "pro"
            else:
                response = ray.get(con.argue.remote(history, proposition))
                role = "con"

            history.append({"role": "ai", "content": response})
            turn_count += 1
            print(f"[Debate {proposition[:20]}... Turn {turn_count}] {role.capitalize()}: {response[:60]}...")
        return history
    
# Main
if __name__ == "__main__":
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    load_dotenv()

    # Load personas (same JSON as before)
    with open("personas.json", "r") as f:
        personas_data = json.load(f)
    personas = [Persona(**p) for p in personas_data]

    # Create a supervisor actor (we'll use one supervisor to run multiple debates)
    supervisor = DebateSupervisor.remote()

    # Define a list of propositions to debate
    propositions = [
        "Artificial intelligence will eventually replace most human jobs.",
        "Social media does more harm than good.",
        "Universal basic income should be implemented worldwide.",
        "Space exploration is a waste of resources.",
    ]

    # Launch debates in parallel (use a list of object references)
    futures = []
    for i, prop in enumerate(propositions):
        pro_idx = i % len(personas)
        con_idx = (i + 1) % len(personas)
        future = supervisor.run_debate.remote(prop, personas[pro_idx], personas[con_idx], max_turns=3)
        futures.append(future)
        
    # Wait for all debates to complete and collect transcripts
    results = ray.get(futures)
    # Print full transcripts (or save to files)
    for i, hist in enumerate(results):
        print(f"\n========== Debate {i+1}: {propositions[i]} ===========\n")
        for msg in hist:
            print(f"{msg['role'].upper()}: {msg['content']}\n")

    # Shutdown ray
    ray.shutdown()
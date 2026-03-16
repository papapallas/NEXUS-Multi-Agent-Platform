import ray
import os
import json
import time
import uuid
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dataclasses import dataclass
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

# ---------- Persona (unchanged) ----------
@dataclass
class Persona:
    name: str
    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float

    def describe(self) -> str:
        desc = f"Your name is {self.name}. "
        if self.openness > 0.7:
            desc += "You are highly open, curious, and imaginative. "
        elif self.openness < 0.3:
            desc += "You are conventional and prefer routine. "
        else:
            desc += "You are moderately open to new experiences. "
        if self.conscientiousness > 0.7:
            desc += "You are very organised, disciplined, and detail-oriented. "
        elif self.conscientiousness < 0.3:
            desc += "You are spontaneous and sometimes disorganised. "
        else:
            desc += "You have a balanced approach to order. "
        if self.extraversion > 0.7:
            desc += "You are outgoing, talkative, and energetic. "
        elif self.extraversion < 0.3:
            desc += "You are reserved and prefer solitude. "
        else:
            desc += "You are moderately sociable. "
        if self.agreeableness > 0.7:
            desc += "You are cooperative, compassionate, and trusting. "
        elif self.agreeableness < 0.3:
            desc += "You are competitive and sceptical of others. "
        else:
            desc += "You tend to be cooperative but can be competitive. "
        if self.neuroticism > 0.7:
            desc += "You are prone to stress, anxiety, and mood swings. "
        elif self.neuroticism < 0.3:
            desc += "You are emotionally stable and calm. "
        else:
            desc += "You have moderate emotional reactivity. "
        return desc

# ---------- Ray Actors with Memory ----------
@ray.remote
class ProAgent:
    def __init__(self, persona: Persona, role: str, chroma_path: str = "./chroma_db"):
        self.persona = persona
        self.role = role  # "pro"
        self.model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, api_key=os.getenv("OPENAI_API_KEY"))
        # Initialize Chroma client (each actor gets its own client, but all point to same persistent DB)
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="agent_memories",
            embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-ada-002"
            )
        )

    def store_memory(self, statement: str, proposition: str, debate_id: str, turn: int, debate_number: int):
        """Store a statement in Chroma with metadata."""
        self.collection.add(
            documents=[statement],
            metadatas=[{
                "debate_id": debate_id,
                "debate_number": debate_number,
                "turn": turn,
                "role": self.role,
                "proposition": proposition,
                "timestamp": time.time()
            }],
            ids=[str(uuid.uuid4())]
        )

    def retrieve_memories(self, proposition: str, current_debate_number: int, n_results: int = 3) -> List[str]:
        """Retrieve relevant past statements from last 3 debates."""
        results = self.collection.query(
            query_texts=[proposition],
            n_results=n_results,
            where={
                "$and": [
                    {"role": {"$eq": self.role}},
                    {"debate_number": {"$gte": current_debate_number - 3}},
                    {"debate_number": {"$lt": current_debate_number}}
                ]
            }
        )
        if results['documents']:
            return results['documents'][0]
        return []

    def argue(self, conversation_history: List[Dict], proposition: str, debate_id: str, turn: int, debate_number: int) -> str:
        """Generate an argument, augmented with retrieved memories."""
        # Retrieve past memories
        past_statements = self.retrieve_memories(proposition, debate_number)
        memory_text = ""
        if past_statements:
            memory_text = "Here are some relevant things you said in past debates:\n" + "\n".join(past_statements) + "\n\n"

        system_content = (
            f"You are the PRO side in a debate. {self.persona.describe()} "
            f"The proposition is: '{proposition}'. "
            f"{memory_text}"
            "Argue strongly FOR the proposition. "
            "Directly rebut the last argument made by the Con side. "
            "Present a new strong argument or counterargument."
        )
        messages = [SystemMessage(content=system_content)]
        for msg in conversation_history:
            if msg["role"] == "human":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        response = self.model.invoke(messages)
        # Store this statement in memory
        self.store_memory(response.content, proposition, debate_id, turn, debate_number)
        return response.content

@ray.remote
class ConAgent:
    def __init__(self, persona: Persona, role: str, chroma_path: str = "./chroma_db"):
        self.persona = persona
        self.role = role  # "con"
        self.model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, api_key=os.getenv("OPENAI_API_KEY"))
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="agent_memories",
            embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-ada-002"
            )
        )

    def store_memory(self, statement: str, proposition: str, debate_id: str, turn: int, debate_number: int):
        self.collection.add(
            documents=[statement],
            metadatas=[{
                "debate_id": debate_id,
                "debate_number": debate_number,
                "turn": turn,
                "role": self.role,
                "proposition": proposition,
                "timestamp": time.time()
            }],
            ids=[str(uuid.uuid4())]
        )

    def retrieve_memories(self, proposition: str, current_debate_number: int, n_results: int = 3) -> List[str]:
        results = self.collection.query(
            query_texts=[proposition],
            n_results=n_results,
            where={
                "$and": [
                    {"role": {"$eq": self.role}},
                    {"debate_number": {"$gte": current_debate_number - 3}},
                    {"debate_number": {"$lt": current_debate_number}}
                ]
            }
        )
        if results['documents']:
            return results['documents'][0]
        return []

    def argue(self, conversation_history: List[Dict], proposition: str, debate_id: str, turn: int, debate_number: int) -> str:
        past_statements = self.retrieve_memories(proposition, debate_number)
        memory_text = ""
        if past_statements:
            memory_text = "Here are some relevant things you said in past debates:\n" + "\n".join(past_statements) + "\n\n"

        system_content = (
            f"You are the CON side in a debate. {self.persona.describe()} "
            f"The proposition is: '{proposition}'. "
            f"{memory_text}"
            "Argue strongly AGAINST the proposition. "
            "Directly rebut the last argument made by the Pro side. "
            "Present a new strong argument or counterargument."
        )
        messages = [SystemMessage(content=system_content)]
        for msg in conversation_history:
            if msg["role"] == "human":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        response = self.model.invoke(messages)
        self.store_memory(response.content, proposition, debate_id, turn, debate_number)
        return response.content

@ray.remote
class DebateSupervisor:
    def __init__(self):
        self.debate_counter = 0

    def run_debate(self, proposition: str, pro_persona: Persona, con_persona: Persona, max_turns: int = 5):
        self.debate_counter += 1
        debate_id = f"debate_{self.debate_counter}"
        debate_number = self.debate_counter

        pro = ProAgent.remote(pro_persona, role="pro")
        con = ConAgent.remote(con_persona, role="con")

        history = [
            {"role": "human", "content": f"We are debating the proposition: '{proposition}'. Pro, please begin."}
        ]
        turn_count = 0

        while turn_count < max_turns:
            if turn_count % 2 == 0:
                response = ray.get(pro.argue.remote(history, proposition, debate_id, turn_count, debate_number))
                role = "pro"
            else:
                response = ray.get(con.argue.remote(history, proposition, debate_id, turn_count, debate_number))
                role = "con"

            history.append({"role": "ai", "content": response})
            turn_count += 1
            print(f"[Debate {proposition[:20]}... Turn {turn_count}] {role.capitalize()}: {response[:60]}...")
        return history

# ---------- Main ----------
if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    load_dotenv()

    # Load personas
    with open("personas.json", "r") as f:
        personas_data = json.load(f)
    personas = [Persona(**p) for p in personas_data]

    # Create a supervisor
    supervisor = DebateSupervisor.remote()

    # Propositions for two debates (we'll run them sequentially to test memory)
    propositions = [
        "Artificial intelligence will eventually replace most human jobs.",
        "Social media does more harm than good.",
    ]

    # Run debates sequentially to demonstrate memory recall
    for i, prop in enumerate(propositions):
        print(f"\n=== Starting Debate {i+1}: {prop} ===\n")
        pro_idx = i % len(personas)
        con_idx = (i + 1) % len(personas)
        future = supervisor.run_debate.remote(prop, personas[pro_idx], personas[con_idx], max_turns=3)
        result = ray.get(future)  # wait for each debate to finish
        # Optionally print full transcript
        print(f"\n===== Debate {i+1} Transcript =====\n")
        for msg in result:
            print(f"{msg['role'].upper()}: {msg['content']}\n")

    # Now run a third debate to see if agents remember from previous debates
    print("\n=== Starting Third Debate (with memory recall) ===\n")
    prop3 = "Universal basic income should be implemented worldwide."
    pro_idx = 0
    con_idx = 1
    future = supervisor.run_debate.remote(prop3, personas[pro_idx], personas[con_idx], max_turns=3)
    result = ray.get(future)
    print("\n===== Third Debate Transcript =====\n")
    for msg in result:
        print(f"{msg['role'].upper()}: {msg['content']}\n")

    ray.shutdown()
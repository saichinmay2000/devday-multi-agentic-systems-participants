from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class EnergyState(TypedDict):
    input: str
    energy_level: str
    response: str

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=0
)

def detect_energy_level(state: EnergyState):
    prompt = f"""You are an energy-level classifier for a weekend activity recommender.

Read the user's message and decide how much physical and mental energy they currently have. Classify it into exactly one of these three labels:

- high   → enthusiastic, energetic, motivated, excited, restless, eager to move (e.g. "I'm pumped!", "ready to crush a hike", "feeling amazing")
- medium → steady, okay, neutral, mildly positive or mildly tired but functional (e.g. "I'm fine", "decent day", "a bit slow but okay")
- low    → tired, drained, sad, lazy, burnt out, unwell, unmotivated (e.g. "exhausted", "wiped out", "just want to lie down")

Rules:
- Output ONLY one word: high, medium, or low.
- No punctuation, no explanation, no quotes, no extra text.
- If the message is ambiguous or neutral, default to medium.

Message: "{state['input']}"

Answer:"""
    
    result = llm.invoke(prompt).content.strip().lower()
            
    state["energy_level"] = result
    return state

def low_energy_node(state: EnergyState):
    prompt = f"""The user is feeling LOW energy — tired, drained, or unmotivated.

Their message: "{state['input']}"

Suggest ONE restorative weekend activity that requires minimal effort and helps them recharge (e.g. cozy movie marathon, reading a book with tea, a gentle nap, light journaling, a warm bath, listening to calm music).

Respond in 2–3 friendly sentences. Acknowledge how they feel briefly, then give the recommendation. No lists, no headers."""
    state["response"] = llm.invoke(prompt).content.strip()
    return state

def medium_energy_node(state: EnergyState):
    prompt = f"""The user is feeling MEDIUM energy — steady and functional, neither drained nor pumped.

Their message: "{state['input']}"

Suggest ONE balanced weekend activity that's engaging but not exhausting (e.g. a casual walk in the park, brunch with a friend, visiting a museum, light cooking project, a bike ride, browsing a bookstore).

Respond in 2–3 friendly sentences. Acknowledge their vibe briefly, then give the recommendation. No lists, no headers."""
    state["response"] = llm.invoke(prompt).content.strip()
    return state

def high_energy_node(state: EnergyState):
    prompt = f"""The user is feeling HIGH energy — enthusiastic, motivated, and ready to move.

Their message: "{state['input']}"

Suggest ONE active, adventurous weekend activity that channels their energy (e.g. a hike, a HIIT workout, rock climbing, a long bike ride, a dance class, exploring a new neighborhood on foot).

Respond in 2–3 friendly sentences. Match their excitement briefly, then give the recommendation. No lists, no headers."""
    state["response"] = llm.invoke(prompt).content.strip()
    return state

def route_energy(state: EnergyState):
    return state["energy_level"]

# Build the graph
builder = StateGraph(EnergyState)

# Add nodes
builder.add_node("detect_energy_level", detect_energy_level)
builder.add_node("low", low_energy_node)
builder.add_node("medium", medium_energy_node)
builder.add_node("high", high_energy_node)

# Set entry point
builder.set_entry_point("detect_energy_level")

builder.add_conditional_edges(
    "detect_energy_level",
    route_energy,
    {"low": "low", "medium": "medium", "high": "high"},
)

# Add edges to END
builder.add_edge("low", END)
builder.add_edge("medium", END)
builder.add_edge("high", END)

# Compile graph
graph = builder.compile()

if __name__ == "__main__":
    print("-" * 50)
    print("Welcome to the Weekend Activity Recommender!")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nHey how are you feeling today? (or type 'quit' to exit): ")
            if user_input.strip().lower() in ['quit', 'exit', 'q']:
                break
                
            if not user_input.strip():
                continue
                
            print("\nProcessing...")
            result = graph.invoke({"input": user_input})
            
            print(f"Detected Energy Level: [{result['energy_level'].upper()}]")
            print(f"Recommendation: {result['response']}")
            
        except Exception as e:
            print(f"An error occurred: {e}")
            break

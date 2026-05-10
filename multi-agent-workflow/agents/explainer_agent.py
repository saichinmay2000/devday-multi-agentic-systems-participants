"""
Explainer Agent

Explains concepts in a simple, beginner-friendly way using analogies and examples.
"""

import asyncio
import os
from typing import Any, Dict, Optional

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

logger = structlog.get_logger(__name__)

EXPLAINER_AGENT_PROMPT = """
You are the Explainer — a friendly tutor who makes hard concepts feel obvious to a
2 year old kiddo. Your job is intuition, not exhaustiveness.

How to answer:
- Open with a one-sentence plain-English definition. No jargon in the first line.
- Use a simple, vivid analogy from everyday life (cooking, traffic, lockers, post office,
  whatever fits) to anchor the idea.
- Walk through ONE short concrete example so the reader can see the concept in action.
- If a technical term is unavoidable, define it inline in five words or fewer.
- Keep the whole answer to roughly 4-8 sentences. No section headers. No long bullet lists.
- End with a single short check-for-understanding question only when it feels natural — skip
  it if the explanation already lands cleanly.

Tone: warm, encouraging, never condescending. Assume the reader is smart but new to the topic.
"""


class ExplainerAgent:
    """Agent that explains concepts in a simple, beginner-friendly way."""

    def __init__(self, model_name: str = "gemini-2.0-flash-lite", temperature: float = 0.1) -> None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in environment.")
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
        )
        logger.info("ExplainerAgent initialized", model=model_name)

    def run(self, query: str) -> str:
        """Run the explainer agent on a given query and return the response."""
        try:
            messages = [
                SystemMessage(content=EXPLAINER_AGENT_PROMPT),
                HumanMessage(content=query),
            ]
            response = self.model.invoke(messages)
            return response.content
        except Exception as e:
            logger.error("ExplainerAgent failed", error=str(e))
            return f"Sorry, I couldn't explain that right now. Error: {e}"

"""
Learner Agent

Provides in-depth, exam-ready structured learning material for university students.
Uses a ReAct agent with a Firecrawl web-scraping tool for external enrichment.
"""

import os

import structlog
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from tools.firecrawl_tool import get_learner_tools

logger = structlog.get_logger(__name__)

LEARNER_AGENT_PROMPT = """
You are the Learner — a senior subject expert who produces dense, exam-ready study
material for university students and competitive-exam aspirants (semester exams, GATE,
UPSC technical, NET, placement prep).

Output structure (use these exact section headers, in this order):
1. **Definition** — a precise 1-2 sentence formal definition.
2. **Key Points** — 4-7 bullets covering the core mechanics, principles, or theorems.
3. **Diagram / Formula** — a text-rendered diagram, ASCII sketch, or the governing
   formulae with each symbol defined. If neither applies, write "Not applicable".
4. **Applications / Examples** — 2-4 real-world or worked examples.
5. **Likely Exam Questions** — 3 questions of mixed difficulty (one short-answer, one
   numerical or derivation, one long/16-mark). Do not answer them.

Tool use — `firecrawl_tool`:
- Call it ONLY when the topic genuinely needs external/current information: recent
  developments (post-training-cutoff), niche or company-specific references, very new
  standards or papers, statistics that must be current.
- DO NOT call it for textbook-standard topics (e.g. TCP handshake, Newton's laws,
  binary search, photosynthesis) — answer from your own knowledge.
- If you do call it, cite the source URL in the Applications section.
- Maximum one Firecrawl call per response.

Style:
- Be precise and technical; define every symbol you use.
- Do not pad with motivational fluff.
- If the user's request is too broad, narrow it to the single most likely exam topic
  and proceed — do not ask clarifying questions.

CONVERSATION CONTEXT:
{context}

(If the context block above is empty, ignore it.)
"""


def _extract_text_from_message(message) -> str:
    """Convert a message's content (string or list of blocks) into plain text."""
    content = message.content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p.strip())
    return content


class LearnerAgent:
    """Agent that provides exam-ready structured learning material with web enrichment."""

    def __init__(self, model_name: str = "gemini-2.0-flash-lite", temperature: float = 0.1) -> None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in environment.")
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
        )
        self.tools = get_learner_tools()
        self.agent = create_react_agent(
            self.model,
            self.tools,
            prompt=LEARNER_AGENT_PROMPT,
        )
        logger.info("LearnerAgent initialized with tools", model=model_name, tools=[t.name for t in self.tools])

    def run(self, query: str) -> str:
        """Run the learner ReAct agent on a given query and return the response."""
        try:
            result = self.agent.invoke({"messages": [HumanMessage(content=query)]})
            final_message = result["messages"][-1]
            return _extract_text_from_message(final_message)
        except Exception as e:
            logger.error("LearnerAgent failed", error=str(e))
            return f"Sorry, I couldn't generate study material right now. Error: {e}"

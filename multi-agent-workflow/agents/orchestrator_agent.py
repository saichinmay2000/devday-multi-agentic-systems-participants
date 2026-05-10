"""
Orchestrator Agent

Routes conversations to the appropriate sub-agent (ExplainerAgent or LearnerAgent)
based on user intent. The two agents are registered as LangChain tools.
"""

import os
from typing import Optional

import structlog
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from agents.explainer_agent import ExplainerAgent
from agents.learner_agent import LearnerAgent
from tools.save_to_notion import save_notes_to_notion
logger = structlog.get_logger(__name__)

class AgentToolInput(BaseModel):
    """Input schema shared by both agent tools."""
    message: str = Field(description="The user's query to respond to")

def _build_agent_tools():
    """Instantiate agents and wrap them as LangChain StructuredTools."""
    explainer = ExplainerAgent()
    learner = LearnerAgent()

    def _call_explainer(message: str) -> str:
        logger.info("Calling agent: explainer")
        return explainer.run(message)

    def _call_learner(message: str) -> str:
        logger.info("Calling agent: learner")
        return learner.run(message)

    explainer_tool = StructuredTool.from_function(
        func=_call_explainer,
        name="explainer",
        description=(
            "Use this tool when the user wants a concept explained simply — "
            "they say things like 'explain', 'what is', 'I don't understand', "
            "'teach me', or 'explain like I'm 5'."
        ),
        args_schema=AgentToolInput,
    )

    learner_tool = StructuredTool.from_function(
        func=_call_learner,
        name="learner",
        description=(
            "Use this tool when the user wants in-depth study material, "
            "exam-ready notes, 16-mark answers, revision material, or asks about "
            "university/competitive exam preparation."
        ),
        args_schema=AgentToolInput,
    )
    return [explainer_tool, learner_tool, save_notes_to_notion]

ORCHESTRATOR_PROMPT = """
You are the Exam Helper orchestrator — a router that helps university and competitive-exam
students by delegating their request to the right specialist tool. You do NOT answer the
student's academic question yourself; you decide which tool to call and return its output.

You have three tools:

1. `explainer`
   - Use when the user wants a concept explained simply or wants to build intuition.
   - Trigger phrases: "explain", "what is", "what does ... mean", "I don't understand",
     "ELI5", "explain like I'm 5", "teach me", "help me understand", "simple explanation".
   - Pass the user's full question as `message`.

2. `learner`
   - Use when the user wants exam-ready study material or in-depth coverage.
   - Trigger phrases: "exam notes", "study material", "16-mark answer", "long answer",
     "revision notes", "syllabus topic", "important questions", "structured notes",
     "prepare me for ...", "GATE/UPSC/semester exam".
   - Pass the user's full question as `message`.

3. `save_notes_to_notion`
   - Use ONLY when the user explicitly asks to save / store / write / push notes to Notion.
   - Trigger phrases: "save to Notion", "store this in Notion", "add this page to Notion".
   - Required args: a short `title` and the `content` to save.
   - Typical flow: if the user says "make notes on X and save them", first call `learner`
     to produce the notes, then call `save_notes_to_notion` with the learner's output as
     `content` and a concise topic-based `title`.

Routing rules:
- Pick exactly ONE routing tool per turn (`explainer` OR `learner`), unless the user has
  combined a study request with an explicit save request, in which case call `learner`
  first and then `save_notes_to_notion`.
- If the request is ambiguous (e.g. a bare topic like "photosynthesis"), prefer `explainer`
  for short questions and `learner` if the user mentions exams, depth, or structured notes.
- Never invent answers, never call a tool more than once for the same purpose, and never
  call `save_notes_to_notion` unless the user explicitly asked to save.

Return the tool's output verbatim as your final answer.

CURRENT STATE:
- Intent: {intent}
"""


class OrchestratorAgent:
    """
    Single orchestrator node that holds the two sub-agents as tools and uses
    a ReAct agent loop to route user queries to the right tool.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash-lite", temperature: float = 0.7) -> None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in environment.")

        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
        )
        self.tools = _build_agent_tools()
        self._intent = "unknown"
        logger.info("OrchestratorAgent initialized with tools", tools=[t.name for t in self.tools])

    def run(self, messages: list, intent: str = "unknown") -> str:
        """
        Run the orchestrator ReAct agent over the full message history.

        Args:
            messages: List of LangChain BaseMessage objects (conversation history).
            intent: Detected user intent ("explain", "learn", or "unknown").

        Returns:
            Final response string from the orchestrator / routed tool.
        """
        try:
            from langchain_core.messages import AIMessage, ToolMessage

            prompt = ORCHESTRATOR_PROMPT.format(intent=intent)
            agent = create_react_agent(self.model, self.tools, prompt=prompt)

            result = agent.invoke({"messages": messages})
            all_messages = result.get("messages", [])

            orchestrator_response = ""
            ai_fallback = ""

            for msg in reversed(all_messages):
                if isinstance(msg, ToolMessage) and msg.content:
                    orchestrator_response = msg.content
                    break
                if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                    ai_fallback = self._extract_text(msg.content)

            return orchestrator_response or ai_fallback or "I'm here to help! What would you like to learn?"

        except Exception as e:
            logger.error("OrchestratorAgent.run failed", error=str(e))
            return f"Something went wrong: {e}"

    @staticmethod
    def _extract_text(content) -> str:
        """Extract plain text from string or list-of-blocks content."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block["text"])
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts)
        return str(content)

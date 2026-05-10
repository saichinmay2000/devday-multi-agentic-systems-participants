"""
Multi-Agent Workflow — Exam Helper System

Architecture:
    User Message
         |
         v
    [orchestrator node]  ←─ OrchestratorAgent (ReAct loop)
         |                        |              |
         |                [explainer tool]  [learner tool]
         v
    Final Response

The entire routing logic lives inside OrchestratorAgent.
LangGraph is used to define the graph: START → orchestrator → END.
"""

import os
from typing import Annotated, Any, Dict, List, Optional

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agents.orchestrator_agent import OrchestratorAgent

logger = structlog.get_logger(__name__)



class ExamHelperState(TypedDict):
    """State that flows through every node in the graph."""
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    orchestrator_result: Optional[str]
    error: List[str]


def get_initial_state() -> ExamHelperState:
    return ExamHelperState(
        messages=[],
        user_query="",
        orchestrator_result=None,
        error=[],
    )



def make_orchestrator_node(agent: OrchestratorAgent):
    """
    Returns a LangGraph-compatible node function that wraps OrchestratorAgent.
    """
    def orchestrator_node(state: ExamHelperState) -> Dict[str, Any]:
        try:
            response = agent.run(state.get("messages", []))
            return {
                "orchestrator_result": response,
                "messages": [AIMessage(content=response)],
                "error": [],
            }
        except Exception as e:
            error_msg = f"Orchestrator node failed: {e}"
            logger.error(error_msg)
            return {
                "orchestrator_result": None,
                "error": [error_msg],
            }

    return orchestrator_node



class MultiAgentWorkflow:
    """
    LangGraph workflow with a single orchestrator node.
    The orchestrator internally routes to explainer or learner tools.
    """

    def __init__(self, conversation_id: Optional[str] = None) -> None:
        self.orchestrator = OrchestratorAgent()
        self.memory = MemorySaver()
        self.conversation_id = conversation_id or f"session_{hash(str(os.urandom(8)))}"
        self.config = {"configurable": {"thread_id": self.conversation_id}}
        self._state: Optional[ExamHelperState] = None

        self._graph = self._build_graph()
        logger.info("MultiAgentWorkflow ready", conversation_id=self.conversation_id)

    def _build_graph(self):
        """Build and compile the LangGraph state graph."""
        graph = StateGraph(ExamHelperState)

        graph.add_node("orchestrator", make_orchestrator_node(self.orchestrator))
        graph.add_edge(START, "orchestrator")
        graph.add_edge("orchestrator", END)

        return graph.compile(checkpointer=self.memory)

    def _get_state(self) -> ExamHelperState:
        if self._state is None:
            self._state = get_initial_state()
        return self._state

    def chat(self, user_message: str) -> str:
        """
        Send a message through the workflow and return the assistant's response.
        Maintains full conversation history across turns.
        """
        state = self._get_state()

        state["messages"] = list(state.get("messages", [])) + [
            HumanMessage(content=user_message)
        ]
        state["user_query"] = user_message

        final_state = self._graph.invoke(state, self.config)
        self._state = dict(final_state)

        return final_state.get("orchestrator_result") or "I'm here to help! What would you like to learn?"

    def reset(self) -> None:
        """Start a new conversation."""
        self._state = None
        self.conversation_id = f"session_{hash(str(os.urandom(8)))}"
        self.config = {"configurable": {"thread_id": self.conversation_id}}
        logger.info("Workflow reset", new_id=self.conversation_id)

    def get_greeting(self) -> str:
        """Return a welcoming greeting from the orchestrator model."""
        try:
            response = self.orchestrator.model.invoke(
                "You are the Exam Helper assistant. Write a single warm 1-2 sentence greeting "
                "for a student who just opened the app. Briefly mention you can either explain "
                "concepts simply or generate exam-ready study notes, and invite them to ask "
                "their first question. No emojis, no lists, plain text only."
            )
            return response.content if response and response.content else "Hi! What would you like to learn today?"
        except Exception:
            return "Hi! What would you like to learn today?"

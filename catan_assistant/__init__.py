"""LLM-Powered Catan Assistant - Main Package"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__description__ = "LLM-powered Settlers of Catan game analysis and strategy assistant"

from .core.game_state import GameState, GameAction, PlayerState
from .reasoning.dspy_module import CatanDecisionModule
from .orchestration.langgraph_pipeline import CatanOrchestrator
from .orchestration.langchain_tools import CatanToolkit

__all__ = [
    "GameState",
    "GameAction", 
    "PlayerState",
    "CatanDecisionModule",
    "CatanOrchestrator",
    "CatanToolkit"
]
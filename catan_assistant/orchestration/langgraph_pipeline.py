"""
LangGraph orchestration pipeline for Catan Assistant.
Manages the workflow from screen capture to decision output.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Dict, Any, List, TypedDict
import asyncio
from ..perception.screen_capture import ScreenCapture
from ..perception.state_extraction import StateExtractor
from ..reasoning.dspy_module import CatanDecisionModule
from ..core.game_state import GameState
import json

class CatanWorkflowState(TypedDict):
    """State structure for the Catan workflow."""
    screenshot: bytes
    raw_board_data: Dict[str, Any]
    game_state: GameState
    decision_result: Dict[str, Any]
    error_message: str
    workflow_step: str

class CatanOrchestrator:
    """
    Main orchestrator using LangGraph for the Catan Assistant pipeline.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.screen_capture = ScreenCapture(config.get("capture", {}))
        self.state_extractor = StateExtractor(config.get("extraction", {}))
        self.decision_module = CatanDecisionModule()
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Define the state graph
        workflow = StateGraph(CatanWorkflowState)
        
        # Add nodes
        workflow.add_node("capture", self._capture_node)
        workflow.add_node("extract_state", self._extract_state_node)
        workflow.add_node("make_decision", self._decision_node)
        workflow.add_node("format_output", self._output_node)
        workflow.add_node("handle_error", self._error_node)
        
        # Set entry point
        workflow.set_entry_point("capture")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "capture",
            self._should_continue_after_capture,
            {
                "continue": "extract_state",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "extract_state",
            self._should_continue_after_extraction,
            {
                "continue": "make_decision",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("make_decision", "format_output")
        workflow.add_edge("format_output", END)
        workflow.add_edge("handle_error", END)
        
        # Compile with memory
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    async def _capture_node(self, state: CatanWorkflowState) -> CatanWorkflowState:
        """Node for screen capture."""
        try:
            state["workflow_step"] = "capture"
            screenshot = await self.screen_capture.capture_screen()
            state["screenshot"] = screenshot
            return state
        except Exception as e:
            state["error_message"] = f"Capture failed: {str(e)}"
            return state
    
    async def _extract_state_node(self, state: CatanWorkflowState) -> CatanWorkflowState:
        """Node for game state extraction."""
        try:
            state["workflow_step"] = "extract_state"
            raw_data = await self.state_extractor.extract_from_screenshot(
                state["screenshot"]
            )
            game_state = GameState.from_json(raw_data)
            
            state["raw_board_data"] = raw_data
            state["game_state"] = game_state
            return state
        except Exception as e:
            state["error_message"] = f"State extraction failed: {str(e)}"
            return state
    
    async def _decision_node(self, state: CatanWorkflowState) -> CatanWorkflowState:
        """Node for AI decision making."""
        try:
            state["workflow_step"] = "make_decision"
            decision_result = self.decision_module.forward(state["game_state"])
            state["decision_result"] = decision_result
            return state
        except Exception as e:
            state["error_message"] = f"Decision making failed: {str(e)}"
            return state
    
    async def _output_node(self, state: CatanWorkflowState) -> CatanWorkflowState:
        """Node for formatting output."""
        state["workflow_step"] = "format_output"
        
        # Format the final recommendation
        decision = state["decision_result"]
        formatted_output = {
            "recommendation": {
                "action": decision["best_action"],
                "rationale": decision["rationale"],
                "confidence": decision["confidence"]
            },
            "analysis": {
                "position": decision["position_analysis"],
                "priorities": decision["strategic_priorities"],
                "threats": decision["opponent_threats"]
            },
            "game_context": {
                "turn": state["game_state"].turn,
                "player": state["game_state"].current_player,
                "victory_points": state["game_state"].get_current_player_state().victory_points
            }
        }
        
        state["decision_result"] = formatted_output
        return state
    
    async def _error_node(self, state: CatanWorkflowState) -> CatanWorkflowState:
        """Node for error handling."""
        state["workflow_step"] = "error"
        # Log error and provide fallback response
        return state
    
    def _should_continue_after_capture(self, state: CatanWorkflowState) -> str:
        """Determine next step after capture."""
        return "error" if state.get("error_message") else "continue"
    
    def _should_continue_after_extraction(self, state: CatanWorkflowState) -> str:
        """Determine next step after state extraction."""
        return "error" if state.get("error_message") else "continue"
    
    async def process_game(self, thread_id: str = "default") -> Dict[str, Any]:
        """
        Process a complete game analysis cycle.
        
        Args:
            thread_id: Unique identifier for this analysis session
            
        Returns:
            Complete analysis results
        """
        initial_state = CatanWorkflowState(
            screenshot=b"",
            raw_board_data={},
            game_state=None,
            decision_result={},
            error_message="",
            workflow_step="init"
        )
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Run the workflow
        final_state = await self.workflow.ainvoke(initial_state, config)
        
        if final_state.get("error_message"):
            return {
                "success": False,
                "error": final_state["error_message"],
                "step": final_state["workflow_step"]
            }
        
        return {
            "success": True,
            "result": final_state["decision_result"],
            "workflow_steps": ["capture", "extract_state", "make_decision", "format_output"]
        }

class StreamingOrchestrator(CatanOrchestrator):
    """
    Streaming version of the orchestrator for real-time updates.
    """
    
    async def stream_analysis(self, thread_id: str = "stream"):
        """
        Stream the analysis process step by step.
        
        Yields:
            Dict with step updates as they complete
        """
        initial_state = CatanWorkflowState(
            screenshot=b"",
            raw_board_data={},
            game_state=None,
            decision_result={},
            error_message="",
            workflow_step="init"
        )
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Stream the workflow execution
        async for step_output in self.workflow.astream(initial_state, config):
            # Extract the current step and its output
            for node_name, node_output in step_output.items():
                yield {
                    "step": node_name,
                    "status": "completed",
                    "data": {
                        "workflow_step": node_output.get("workflow_step"),
                        "error": node_output.get("error_message"),
                        "partial_result": node_output.get("decision_result", {})
                    }
                }
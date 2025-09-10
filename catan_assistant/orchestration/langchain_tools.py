"""
LangChain tools for Catan Assistant.
Provides tool-based interfaces for various game operations.
"""

from langchain_core.tools import tool, BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from typing import Dict, Any, List, Optional, Type
from pydantic import BaseModel, Field
import json
from ..core.game_state import GameState, GameAction, ActionType
from ..perception.screen_capture import ScreenCapture
from ..perception.state_extraction import StateExtractor
from ..reasoning.dspy_module import CatanDecisionModule

class CaptureInput(BaseModel):
    """Input schema for screen capture tool."""
    region: Optional[Dict[str, int]] = Field(
        default=None, 
        description="Screen region to capture {x, y, width, height}"
    )

class StateExtractionInput(BaseModel):
    """Input schema for state extraction tool."""
    image_data: bytes = Field(description="Screenshot image data")
    extraction_mode: str = Field(
        default="full", 
        description="Extraction mode: 'full', 'board_only', 'resources_only'"
    )

class DecisionInput(BaseModel):
    """Input schema for decision making tool."""
    game_state_json: str = Field(description="Game state in JSON format")
    player_perspective: str = Field(description="Player to make decision for")

class CatanScreenCaptureTool(BaseTool):
    """Tool for capturing Catan game screen."""
    
    name = "catan_screen_capture"
    description = "Captures screenshot of Catan game board for analysis"
    args_schema: Type[BaseModel] = CaptureInput
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.screen_capture = ScreenCapture(config)
    
    def _run(
        self, 
        region: Optional[Dict[str, int]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Capture screen and return base64 encoded image."""
        try:
            screenshot = self.screen_capture.capture_screen(region)
            import base64
            encoded_image = base64.b64encode(screenshot).decode()
            return f"Screenshot captured successfully. Size: {len(screenshot)} bytes"
        except Exception as e:
            return f"Screen capture failed: {str(e)}"

class CatanStateExtractionTool(BaseTool):
    """Tool for extracting game state from screenshots."""
    
    name = "catan_state_extraction"
    description = "Extracts Catan game state from screenshot image"
    args_schema: Type[BaseModel] = StateExtractionInput
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.state_extractor = StateExtractor(config)
    
    def _run(
        self,
        image_data: bytes,
        extraction_mode: str = "full",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Extract game state from image data."""
        try:
            raw_state = self.state_extractor.extract_from_screenshot(
                image_data, mode=extraction_mode
            )
            return json.dumps(raw_state, indent=2)
        except Exception as e:
            return f"State extraction failed: {str(e)}"

class CatanDecisionTool(BaseTool):
    """Tool for making strategic decisions in Catan."""
    
    name = "catan_decision_maker"
    description = "Makes strategic decisions for Catan gameplay using AI"
    args_schema: Type[BaseModel] = DecisionInput
    
    def __init__(self):
        super().__init__()
        self.decision_module = CatanDecisionModule()
    
    def _run(
        self,
        game_state_json: str,
        player_perspective: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Make strategic decision based on game state."""
        try:
            state_data = json.loads(game_state_json)
            game_state = GameState.from_json(state_data)
            
            # Ensure we're making decision for correct player
            game_state.current_player = player_perspective
            
            decision_result = self.decision_module.forward(game_state)
            
            return json.dumps({
                "decision": decision_result,
                "player": player_perspective,
                "success": True
            }, indent=2)
        except Exception as e:
            return f"Decision making failed: {str(e)}"

# Functional tool definitions using @tool decorator
@tool
def analyze_board_position(game_state_json: str, player_name: str) -> str:
    """
    Analyze the current board position for strategic opportunities.
    
    Args:
        game_state_json: Current game state in JSON format
        player_name: Name of the player to analyze for
        
    Returns:
        Strategic analysis of the board position
    """
    try:
        state_data = json.loads(game_state_json)
        game_state = GameState.from_json(state_data)
        player_state = game_state.player_states[player_name]
        
        analysis = {
            "victory_points": player_state.victory_points,
            "resource_situation": _analyze_resources(player_state.resources),
            "building_capacity": _analyze_building_potential(player_state),
            "strategic_position": _analyze_strategic_position(game_state, player_name)
        }
        
        return json.dumps(analysis, indent=2)
    except Exception as e:
        return f"Board analysis failed: {str(e)}"

@tool
def calculate_victory_probability(game_state_json: str, player_name: str) -> str:
    """
    Calculate probability of victory for a given player.
    
    Args:
        game_state_json: Current game state in JSON format
        player_name: Name of the player to calculate for
        
    Returns:
        Victory probability analysis
    """
    try:
        state_data = json.loads(game_state_json)
        game_state = GameState.from_json(state_data)
        
        player_vp = game_state.player_states[player_name].victory_points
        max_opponent_vp = max(
            state.victory_points 
            for name, state in game_state.player_states.items() 
            if name != player_name
        )
        
        # Simple heuristic-based probability calculation
        vp_advantage = player_vp - max_opponent_vp
        base_probability = 0.25  # Equal chance with 4 players
        
        if vp_advantage > 0:
            probability = min(0.8, base_probability + (vp_advantage * 0.15))
        else:
            probability = max(0.05, base_probability + (vp_advantage * 0.1))
        
        return json.dumps({
            "player": player_name,
            "victory_probability": probability,
            "current_vp": player_vp,
            "vp_advantage": vp_advantage,
            "analysis": f"Player has {probability:.1%} chance of victory"
        }, indent=2)
    except Exception as e:
        return f"Victory probability calculation failed: {str(e)}"

@tool
def suggest_trade_opportunities(game_state_json: str, player_name: str) -> str:
    """
    Suggest potential trade opportunities with other players.
    
    Args:
        game_state_json: Current game state in JSON format
        player_name: Name of the player to suggest trades for
        
    Returns:
        List of potential trade opportunities
    """
    try:
        state_data = json.loads(game_state_json)
        game_state = GameState.from_json(state_data)
        player_resources = game_state.player_states[player_name].resources
        
        trade_opportunities = []
        
        for other_player, other_state in game_state.player_states.items():
            if other_player == player_name:
                continue
            
            # Find potential beneficial trades
            opportunities = _find_trade_opportunities(
                player_resources, 
                other_state.resources,
                other_player
            )
            trade_opportunities.extend(opportunities)
        
        return json.dumps({
            "player": player_name,
            "trade_opportunities": trade_opportunities,
            "total_opportunities": len(trade_opportunities)
        }, indent=2)
    except Exception as e:
        return f"Trade analysis failed: {str(e)}"

def _analyze_resources(resources: Dict[str, int]) -> Dict[str, Any]:
    """Analyze resource situation."""
    total_resources = sum(resources.values())
    resource_diversity = len([r for r in resources.values() if r > 0])
    
    return {
        "total_count": total_resources,
        "diversity": resource_diversity,
        "dominant_resource": max(resources.items(), key=lambda x: x[1])[0] if resources else None,
        "scarcity": [res for res, count in resources.items() if count == 0]
    }

def _analyze_building_potential(player_state) -> Dict[str, Any]:
    """Analyze building potential."""
    return {
        "can_build_settlement": player_state.settlements < 5,
        "can_build_city": player_state.cities < 4 and player_state.settlements > 0,
        "can_build_road": player_state.roads < 15,
        "building_slots_remaining": {
            "settlements": 5 - player_state.settlements,
            "cities": 4 - player_state.cities,
            "roads": 15 - player_state.roads
        }
    }

def _analyze_strategic_position(game_state: GameState, player_name: str) -> Dict[str, Any]:
    """Analyze strategic position relative to other players."""
    player_vp = game_state.player_states[player_name].victory_points
    other_players = [
        (name, state.victory_points) 
        for name, state in game_state.player_states.items() 
        if name != player_name
    ]
    
    return {
        "current_rank": _calculate_rank(player_vp, [vp for _, vp in other_players]),
        "vp_gap_to_leader": max([vp for _, vp in other_players]) - player_vp,
        "threatening_players": [name for name, vp in other_players if vp >= player_vp]
    }

def _calculate_rank(player_vp: int, other_vps: List[int]) -> int:
    """Calculate player's rank."""
    higher_vp_count = sum(1 for vp in other_vps if vp > player_vp)
    return higher_vp_count + 1

def _find_trade_opportunities(
    player_resources: Dict[str, int], 
    other_resources: Dict[str, int],
    other_player: str
) -> List[Dict[str, Any]]:
    """Find mutually beneficial trade opportunities."""
    opportunities = []
    
    # Find resources player has excess of and other player needs
    for resource, player_count in player_resources.items():
        if player_count > 3:  # Player has excess
            other_count = other_resources.get(resource, 0)
            if other_count < 2:  # Other player needs this resource
                opportunities.append({
                    "with_player": other_player,
                    "give": resource,
                    "give_amount": 1,
                    "reason": f"Player has excess {resource}, {other_player} needs it"
                })
    
    return opportunities

class CatanToolkit:
    """
    Complete toolkit for Catan Assistant operations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tools = [
            CatanScreenCaptureTool(config.get("capture", {})),
            CatanStateExtractionTool(config.get("extraction", {})),
            CatanDecisionTool(),
            analyze_board_position,
            calculate_victory_probability,
            suggest_trade_opportunities
        ]
    
    def get_tools(self) -> List[BaseTool]:
        """Get all available tools."""
        return self.tools
    
    def get_tool_names(self) -> List[str]:
        """Get names of all available tools."""
        return [tool.name for tool in self.tools]
    
    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all available tools."""
        return {tool.name: tool.description for tool in self.tools}
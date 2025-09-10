"""
Core game state management for Catan Assistant.
Handles parsing, validation, and manipulation of game states.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import json
from enum import Enum

class Resource(Enum):
    WOOD = "wood"
    BRICK = "brick"
    WHEAT = "wheat"
    ORE = "ore"
    SHEEP = "sheep"

class ActionType(Enum):
    BUILD_SETTLEMENT = "build_settlement"
    BUILD_CITY = "build_city"
    BUILD_ROAD = "build_road"
    BUY_DEVELOPMENT_CARD = "buy_development_card"
    PLAY_KNIGHT = "play_knight"
    TRADE = "trade"
    END_TURN = "end_turn"

@dataclass
class PlayerState:
    """Represents the state of a single player."""
    name: str
    resources: Dict[str, int] = field(default_factory=dict)
    development_cards: Dict[str, int] = field(default_factory=dict)
    victory_points: int = 0
    settlements: int = 0
    cities: int = 0
    roads: int = 0
    largest_army: bool = False
    longest_road: bool = False

@dataclass
class GameAction:
    """Represents a possible game action."""
    type: ActionType
    position: Optional[Tuple[int, int]] = None
    cost: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GameState:
    """Complete game state representation."""
    turn: int
    phase: str
    current_player: str
    players: List[str]
    dice_roll: Optional[Tuple[int, int]]
    board: Dict[str, Any]
    player_states: Dict[str, PlayerState]
    legal_actions: List[GameAction]
    
    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> 'GameState':
        """Create GameState from JSON representation."""
        # Parse player states
        player_states = {}
        for name, state_data in json_data.get("player_states", {}).items():
            player_states[name] = PlayerState(
                name=name,
                resources=state_data.get("resources", {}),
                development_cards=state_data.get("development_cards", {}),
                victory_points=state_data.get("victory_points", 0),
                settlements=state_data.get("settlements", 0),
                cities=state_data.get("cities", 0),
                roads=state_data.get("roads", 0),
                largest_army=state_data.get("largest_army", False),
                longest_road=state_data.get("longest_road", False)
            )
        
        # Parse legal actions
        legal_actions = []
        for action_data in json_data.get("legal_actions", []):
            action_type = ActionType(action_data["type"])
            legal_actions.append(GameAction(
                type=action_type,
                position=tuple(action_data["position"]) if "position" in action_data else None,
                cost=action_data.get("cost"),
                metadata=action_data.get("metadata", {})
            ))
        
        return cls(
            turn=json_data["game_info"]["turn"],
            phase=json_data["game_info"]["phase"],
            current_player=json_data["game_info"]["current_player"],
            players=json_data["game_info"]["players"],
            dice_roll=tuple(json_data["game_info"]["dice_roll"]) if json_data["game_info"].get("dice_roll") else None,
            board=json_data.get("board", {}),
            player_states=player_states,
            legal_actions=legal_actions
        )
    
    def to_json(self) -> Dict[str, Any]:
        """Convert GameState to JSON representation."""
        return {
            "game_info": {
                "turn": self.turn,
                "phase": self.phase,
                "current_player": self.current_player,
                "players": self.players,
                "dice_roll": list(self.dice_roll) if self.dice_roll else None
            },
            "board": self.board,
            "player_states": {
                name: {
                    "resources": state.resources,
                    "development_cards": state.development_cards,
                    "victory_points": state.victory_points,
                    "settlements": state.settlements,
                    "cities": state.cities,
                    "roads": state.roads,
                    "largest_army": state.largest_army,
                    "longest_road": state.longest_road
                }
                for name, state in self.player_states.items()
            },
            "legal_actions": [
                {
                    "type": action.type.value,
                    "position": list(action.position) if action.position else None,
                    "cost": action.cost,
                    "metadata": action.metadata
                }
                for action in self.legal_actions
            ]
        }
    
    def get_current_player_state(self) -> PlayerState:
        """Get the current player's state."""
        return self.player_states[self.current_player]
    
    def can_afford_action(self, action: GameAction) -> bool:
        """Check if current player can afford an action."""
        if not action.cost:
            return True
        
        current_resources = self.get_current_player_state().resources
        for resource, amount in action.cost.items():
            if current_resources.get(resource, 0) < amount:
                return False
        return True
    
    def get_affordable_actions(self) -> List[GameAction]:
        """Get only the actions the current player can afford."""
        return [action for action in self.legal_actions if self.can_afford_action(action)]
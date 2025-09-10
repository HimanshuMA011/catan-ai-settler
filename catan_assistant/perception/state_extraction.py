"""
State extraction module for Catan Assistant.
Extracts game state from screenshots using computer vision and OCR.
"""

import cv2
import numpy as np
import pytesseract
from typing import Dict, Any, List, Optional, Tuple
import json
import re
from ..core.game_state import GameState, ActionType

class StateExtractor:
    """Extracts Catan game state from screenshots."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ocr_engine = config.get("ocr_engine", "tesseract")
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
    
    async def extract_from_screenshot(
        self, 
        image_data: bytes, 
        mode: str = "full"
    ) -> Dict[str, Any]:
        """
        Extract game state from screenshot.
        
        Args:
            image_data: Screenshot image data
            mode: Extraction mode ('full', 'board_only', 'resources_only')
            
        Returns:
            Extracted game state as dictionary
        """
        # Convert bytes to OpenCV image
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if mode == "full":
            return await self._extract_full_state(img)
        elif mode == "board_only":
            return await self._extract_board_only(img)
        elif mode == "resources_only":
            return await self._extract_resources_only(img)
        else:
            raise ValueError(f"Unknown extraction mode: {mode}")
    
    async def _extract_full_state(self, img: np.ndarray) -> Dict[str, Any]:
        """Extract complete game state."""
        # For demo purposes, return a mock game state
        # In real implementation, this would use computer vision to detect:
        # - Board tiles and their numbers
        # - Settlements and cities
        # - Roads
        # - Player resources (from UI panels)
        # - Current turn indicator
        
        return {
            "game_info": {
                "turn": 5,
                "phase": "main_game",
                "dice_roll": [3, 4],
                "current_player": "Blue",
                "players": ["Red", "Blue", "White", "Orange"]
            },
            "board": {
                "tiles": self._extract_board_tiles(img),
                "settlements": self._extract_settlements(img),
                "roads": self._extract_roads(img),
                "robber": {"position": [1, 1]}
            },
            "player_states": self._extract_player_states(img),
            "legal_actions": self._generate_legal_actions()
        }
    
    def _extract_board_tiles(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """Extract board tile information."""
        # Mock implementation - in reality would use computer vision
        return [
            {"id": 0, "resource": "wheat", "number": 6, "coordinates": [0, 0]},
            {"id": 1, "resource": "wood", "number": 8, "coordinates": [1, 0]},
            {"id": 2, "resource": "brick", "number": 5, "coordinates": [2, 0]},
            {"id": 3, "resource": "ore", "number": 10, "coordinates": [0, 1]},
            {"id": 4, "resource": "sheep", "number": 9, "coordinates": [1, 1]},
        ]
    
    def _extract_settlements(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """Extract settlement and city positions."""
        # Mock implementation
        return [
            {"player": "Blue", "position": [0, 1], "type": "settlement"},
            {"player": "Red", "position": [1, 2], "type": "city"},
            {"player": "White", "position": [2, 1], "type": "settlement"},
        ]
    
    def _extract_roads(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """Extract road positions."""
        # Mock implementation
        return [
            {"player": "Blue", "start": [0, 1], "end": [0, 2]},
            {"player": "Red", "start": [1, 2], "end": [2, 2]},
            {"player": "White", "start": [2, 1], "end": [2, 0]},
        ]
    
    def _extract_player_states(self, img: np.ndarray) -> Dict[str, Any]:
        """Extract player resource and building information."""
        # Mock implementation - would use OCR on player panels
        return {
            "Blue": {
                "resources": {"wood": 2, "brick": 1, "wheat": 0, "ore": 1, "sheep": 1},
                "development_cards": {"knight": 1, "victory_point": 0, "road_building": 0},
                "victory_points": 4,
                "settlements": 2,
                "cities": 0,
                "roads": 8,
                "largest_army": False,
                "longest_road": False
            },
            "Red": {
                "resources": {"wood": 1, "brick": 2, "wheat": 1, "ore": 0, "sheep": 2},
                "development_cards": {"knight": 0, "victory_point": 1, "road_building": 1},
                "victory_points": 5,
                "settlements": 1,
                "cities": 1,
                "roads": 6,
                "largest_army": False,
                "longest_road": True
            },
            "White": {
                "resources": {"wood": 0, "brick": 0, "wheat": 2, "ore": 1, "sheep": 0},
                "development_cards": {"knight": 0, "victory_point": 0, "road_building": 0},
                "victory_points": 3,
                "settlements": 2,
                "cities": 0,
                "roads": 4,
                "largest_army": False,
                "longest_road": False
            },
            "Orange": {
                "resources": {"wood": 1, "brick": 1, "wheat": 1, "ore": 2, "sheep": 1},
                "development_cards": {"knight": 2, "victory_point": 0, "road_building": 0},
                "victory_points": 3,
                "settlements": 2,
                "cities": 0,
                "roads": 5,
                "largest_army": True,
                "longest_road": False
            }
        }
    
    def _generate_legal_actions(self) -> List[Dict[str, Any]]:
        """Generate legal actions based on current state."""
        return [
            {
                "type": "build_settlement", 
                "position": [2, 3], 
                "cost": {"wood": 1, "brick": 1, "wheat": 1, "sheep": 1}
            },
            {
                "type": "build_road", 
                "start": [0, 1], 
                "end": [0, 3], 
                "cost": {"wood": 1, "brick": 1}
            },
            {
                "type": "trade", 
                "offer": {"wheat": 4}, 
                "request": {"wood": 1}, 
                "ratio": "4:1"
            },
            {
                "type": "buy_development_card", 
                "cost": {"wheat": 1, "ore": 1, "sheep": 1}
            },
            {"type": "end_turn"}
        ]
    
    async def _extract_board_only(self, img: np.ndarray) -> Dict[str, Any]:
        """Extract only board information."""
        return {
            "board": {
                "tiles": self._extract_board_tiles(img),
                "settlements": self._extract_settlements(img),
                "roads": self._extract_roads(img),
                "robber": {"position": [1, 1]}
            }
        }
    
    async def _extract_resources_only(self, img: np.ndarray) -> Dict[str, Any]:
        """Extract only player resource information."""
        return {
            "player_states": self._extract_player_states(img)
        }
    
    def _ocr_text_region(self, img: np.ndarray, region: Tuple[int, int, int, int]) -> str:
        """Extract text from specific image region using OCR."""
        x, y, w, h = region
        roi = img[y:y+h, x:x+w]
        
        # Preprocess for better OCR
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Use Tesseract OCR
        text = pytesseract.image_to_string(thresh, config='--psm 7')
        return text.strip()
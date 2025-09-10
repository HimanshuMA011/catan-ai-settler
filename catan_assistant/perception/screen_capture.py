"""
Screen capture module for Catan Assistant.
Handles capturing screenshots and detecting game boards.
"""

import cv2
import numpy as np
import pyautogui
from typing import Dict, Any, Optional, Tuple
import io
from PIL import Image

class ScreenCapture:
    """Handles screen capture operations for Catan game detection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.default_region = config.get("default_region", {
            "x": 0, "y": 0, "width": 1920, "height": 1080
        })
        self.capture_delay = config.get("capture_delay", 0.5)
    
    async def capture_screen(self, region: Optional[Dict[str, int]] = None) -> bytes:
        """
        Capture screenshot of specified region.
        
        Args:
            region: Dictionary with x, y, width, height keys
            
        Returns:
            Screenshot as bytes
        """
        if region is None:
            region = self.default_region
        
        # Use pyautogui for cross-platform screenshot
        screenshot = pyautogui.screenshot(
            region=(region["x"], region["y"], region["width"], region["height"])
        )
        
        # Convert to bytes
        img_buffer = io.BytesIO()
        screenshot.save(img_buffer, format='PNG')
        return img_buffer.getvalue()
    
    def detect_catan_board(self, image_data: bytes) -> Optional[Dict[str, int]]:
        """
        Detect Catan board region in screenshot.
        
        Args:
            image_data: Screenshot image data
            
        Returns:
            Board region coordinates or None if not found
        """
        # Convert bytes to OpenCV image
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define range for typical Catan board colors (brown/beige)
        lower_board = np.array([10, 50, 50])
        upper_board = np.array([30, 255, 255])
        
        # Create mask and find contours
        mask = cv2.inRange(hsv, lower_board, upper_board)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest contour (likely the board)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Validate board size (should be reasonably large)
        if w < 200 or h < 200:
            return None
        
        return {"x": x, "y": y, "width": w, "height": h}
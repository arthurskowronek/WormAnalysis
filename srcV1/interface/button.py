import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from src.interface.theme import Theme

class Button:
    """Modern button with hover effects and animations"""
    def __init__(self, x: int, y: int, width: int, height: int, text: str, 
                 button_type: str = "default", icon: Optional[str] = None, phantom: bool = False):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.button_type = button_type
        self.icon = icon
        self.is_hovered = False
        self.is_pressed = False
        self.is_active = False
        self.animation_progress = 0.0
        self.phantom = phantom
        
    def contains_point(self, x: int, y: int) -> bool:
        return (self.x <= x <= self.x + self.width and 
                self.y <= y <= self.y + self.height)
    
    def get_color(self) -> Tuple[int, int, int]:
        theme = Theme()
        if self.is_active:
            base_color = theme.primary_color
        else:
            base_color = {
                "primary": theme.primary_color,
                "secondary": theme.secondary_color,
                "accent": theme.accent_color,
                "danger": theme.danger_color
            }.get(self.button_type, (148, 157, 165))
            
        if self.is_active:
            return base_color
        elif self.is_hovered:
            # Lighten color on hover
            return tuple(min(255, int(c * 1.2)) for c in base_color)
        else:
            return base_color
    
    def draw(self, img: np.ndarray):
        color = self.get_color()
        
        # Draw shadow
        shadow_offset = 4
        cv2.rectangle(img, 
                     (self.x + shadow_offset, self.y + shadow_offset),
                     (self.x + self.width + shadow_offset, self.y + self.height + shadow_offset),
                     (170, 170, 170, 20), -1)
        
        # Draw button background
        cv2.rectangle(img, (self.x, self.y), (self.x + self.width, self.y + self.height), 
                     color, -1)
        
        # Draw text
        text_color = (255, 255, 255) if self.is_active else Theme().text_primary
        font_scale = 1
        thickness = 2
        
        # Calculate text position for centering
        text_size = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = self.x + (self.width - text_size[0]) // 2
        text_y = self.y + (self.height + text_size[1]) // 2
        
        cv2.putText(img, self.text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

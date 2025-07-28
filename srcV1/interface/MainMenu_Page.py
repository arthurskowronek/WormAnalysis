import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from config import RESSOURCES_DIR
from src.interface.button import Button
from src.interface.theme import Theme

class MainMenu:
    """Modern UI management class"""
    def __init__(self):
        self.width = 2284
        self.height = 1524  
        self.theme = Theme()
        
        # State variables
        self.launch_scan = False
        self.launch_assist_acquisition = False
        self.launch_saved_positions = False
        self.quit = False
        
        # Mouse state
        self.mouse_x = 0
        self.mouse_y = 0
        
        # Create buttons with modern styling
        self.buttons = self._create_buttons()
        
    def _create_buttons(self) -> Dict[str, Button]:
        buttons = {}
        button_width = 400
        button_height = 80
        spacing = 30
        total_height = 4 * button_height + 3 * spacing
        start_y = (self.height - total_height) // 2
        center_x = (self.width - button_width) // 2

        labels = [
            ("AutoSearch", "Automatic worm search"),
            ("Assist", "Assist worm acquisition"),
            ("Load", "Load saved position"),
            ("Quit", "Quit")
        ]

        for idx, (key, label) in enumerate(labels):
            y = start_y + idx * (button_height + spacing)
            buttons[key] = Button(center_x, y, button_width, button_height, label, "primary")

        return buttons
    
    def update_button_states(self):
        """Update button states based on current application state"""
        self.buttons["AutoSearch"].is_active = self.launch_scan
        self.buttons["Assist"].is_active = self.launch_assist_acquisition
        self.buttons["Load"].is_active = self.launch_saved_positions
        self.buttons["Quit"].is_active = self.quit
    
    def update_hover_states(self):
        for button in self.buttons.values():
            button.is_hovered = button.contains_point(self.mouse_x, self.mouse_y)

    def draw_interface(self) -> np.ndarray:
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = self.theme.dark_bg  # Background

        for button in self.buttons.values():
            button.draw(img)

        return img
    
    def handle_click(self, x: int, y: int):
        """Handle mouse click events"""
        for name, button in self.buttons.items():
            if button.contains_point(x, y):
                if name == "AutoSearch":
                    self.launch_scan = True
                elif name == "Assist":
                    self.launch_assist_acquisition = True
                elif name == "Load":
                    self.launch_saved_positions = True
                elif name == "Quit":
                    self.quit = True
                return
        
        # Click outside input fields
        self.scan_input_active = False
        self.fluo_input_active = False
        
    def handle_mouse_move(self, x: int, y: int):
        """Handle mouse movement for hover effects"""
        self.mouse_x = x
        self.mouse_y = y

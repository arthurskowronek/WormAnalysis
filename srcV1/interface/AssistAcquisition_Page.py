
import cv2
import time
import keyboard
import numpy as np
from pathlib import Path
from typing import Dict

from config import RESSOURCES_DIR
from src.interface.button import Button
from src.interface.theme import Theme
from src.system.dataset import Dataset

class AssistAcquisitionPage:
    def __init__(self, mmc, init_pos_x, init_pos_y, OBJECTIVE_MAGNIFICATION_SCAN, OBJECTIVE_MAGNIFICATION_FLUO, DUALVIEW, worm_positions = None):
        
        # graphics parameters
        self.width = 2284
        self.height = 1524  
        self.theme = Theme()
        HelpKeyboardImage = cv2.imread(str(Path(RESSOURCES_DIR) / "HelpKeyboard_Live.png"))
        original_height, original_width = HelpKeyboardImage.shape[:2]
        aspect_ratio = original_width / original_height
        new_width = 700
        self.HelpKeyboardImage = cv2.resize(HelpKeyboardImage, (new_width, int(new_width / aspect_ratio)))
        
        # input parameters
        self.init_pos_x = init_pos_x
        self.init_pos_y = init_pos_y
        self.live_img = None
        self.pos_x = 0
        self.pos_y = 0
        self.worm_positions = worm_positions
        self.CORE = mmc
        self.map_camera = None
        self.obj_scan = OBJECTIVE_MAGNIFICATION_SCAN
        self.obj_fluo = OBJECTIVE_MAGNIFICATION_FLUO
        self.dual_view = DUALVIEW
        
        # state parameters
        self.analyse = False
        self.next_worm = False
        self.last_worm = False
        self.save_current_position = False
        self.find_worm = False
        
        # Mouse state
        self.mouse_x = 0
        self.mouse_y = 0
        
        # End
        self.exit = False
        
        # Create buttons with modern styling
        self.buttons = self._create_buttons()
        
    def set_image_position(self, img, pos_x, pos_y):
        img = cv2.resize(img,(1024,1024))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3) 

        self.live_img = img
        self.pos_x = pos_x
        self.pos_y = pos_y

    def _create_buttons(self) -> Dict[str, Button]:
        """Create all UI buttons with modern styling"""
        buttons = {}

        center_x = self.width // 2 - 200  # Center horizontally
        
        buttons["Analyse"] = Button(center_x, 1200, 400, 200, "Start analysis", "primary")

        # Arrow buttons - move through worms
        buttons["Next_Worm"] = Button(center_x + 900, 1200, 400, 80, "-> Go to next worm", "default") 
        buttons["Last_Worm"] = Button(center_x + 450, 1200, 400, 80, "<- Go to last worm", "default") 
        
        buttons["Save_Position"] = Button(center_x - 900, 1200, 400, 80, "Save position", "primary")  
        
        # End of program
        buttons["End"] = Button(2140, 40, 100, 50, "Exit", "danger") 
        
        return buttons

    def update_button_states(self):
        """Update button states based on current application state"""
        
        # Update scan objective buttons
        for name, button in self.buttons.items():
            if name == "Analyse":
                button.is_active = self.analyse
            elif name == "Next_Worm":
                button.is_active = self.next_worm
            elif name == "Last_Worm":
                button.is_active = self.last_worm
            elif name == "Save_Position":
                button.is_active = self.save_current_position

    def update_hover_states(self):
        """Update button hover states based on mouse position"""
        for button in self.buttons.values():
            button.is_hovered = button.contains_point(self.mouse_x, self.mouse_y)
    
    def draw_interface(self) -> np.ndarray:
        """Draw the complete interface"""
        img = np.ones((1524, 2284, 3), dtype=np.uint8) * 255  # Create a white image for display
    
        # Draw subtle gradient background
        for i in range(self.height):
            alpha = i / self.height
            gray_value = int(250 - alpha * 20)  # Subtle gradient
            img[i, :] = (gray_value, gray_value, gray_value)
        
        # Draw a white rectangle - show the size of the view with the next objective
        h, w, _ = self.live_img.shape
        if self.dual_view: 
            center_x, center_y = w // 4, h // 2
            center_x = 3*center_x + 15
        else:
            center_x, center_y = w // 2, h // 2
        rect_length = int(1024*self.obj_scan/self.obj_fluo)
        rect_width = rect_length/2 if self.dual_view else rect_length
        top_left = (center_x - rect_length // 2, center_y - rect_width // 2)
        bottom_right = (center_x + rect_length // 2, center_y + rect_width // 2)
        cv2.rectangle(self.live_img, top_left, bottom_right, (255,255,255), thickness=3)

        # Add the live image
        img[100:1124, 480:1504] = self.live_img
        
        
        # Add the map image
        img[100:800, 1544:2244] = self.map_camera
        
        # Add the table of worm positions
        img_table = self.worm_positions.show_table_worms_positions()
        height, width = img_table.shape[:2]
        img[100:100+height, 40:40+width] = img_table
        
        # Add the help keyboard image
        img[850:850+280, 1544:1544+700] = self.HelpKeyboardImage
        
        # Update button states
        self.update_button_states()
        self.update_hover_states()
        
        # Draw all buttons
        for button in self.buttons.values():
            if not button.phantom:
                button.draw(img)
                
        # Reset flags so buttons deactivate in the next frame
        self.reset_button_flags()
        
        return img
    
    def handle_click(self, x: int, y: int):
        """Handle mouse click events"""

        for name, btn in self.buttons.items():
            if btn.contains_point(x, y):
                self.last_clicked = name
                if name == "Analyse":
                    self.analyse = True
                elif name == "Next_Worm":
                    self.next_worm = True
                    self.go_to_next_worm()
                elif name == "Last_Worm":
                    self.last_worm = True
                    self.go_to_last_worm()
                elif name == "Save_Position":
                    self.save_current_position = True
                    self.add_new_worm()
                elif name == "End":
                    self.exit = True
                return
    
    def handle_mouse_move(self, x: int, y: int):
        self.mouse_x = x
        self.mouse_y = y

    def handle_key(self, key):
        if keyboard.is_pressed('droite'): # Move to next worm
            self.go_to_next_worm()
        elif keyboard.is_pressed('gauche'): # Move to last worm
            self.go_to_last_worm()
        elif key == ord(' ') or keyboard.is_pressed('space'): # Save worm position
            self.add_new_worm()
            
    def reset_button_flags(self):
        """Reset all one-time-use button flags after each action"""
        self.analyse = False
        self.next_worm = False
        self.last_worm = False
        self.save_current_position = False 
        self.find_worm = False 

    # ----- Action methods -----
    def go_to_next_worm(self):
        self.worm_positions.go_to_newt_worm()
        id = self.worm_positions.get_id_worm_seen()
        x,y = self.worm_positions.get_worm_position(id)
        time.sleep(0.01)
        self.CORE.setXYPosition(self.CORE.getXYStageDevice(), x, y)
        
    def go_to_last_worm(self):
        self.worm_positions.go_to_last_worm()
        id = self.worm_positions.get_id_worm_seen()
        x,y = self.worm_positions.get_worm_position(id)
        time.sleep(0.01)
        self.CORE.setXYPosition(self.CORE.getXYStageDevice(), x, y)

    def add_new_worm(self):
        self.worm_positions.add_worm_position(self.pos_x, self.pos_y)
        self.find_worm = True

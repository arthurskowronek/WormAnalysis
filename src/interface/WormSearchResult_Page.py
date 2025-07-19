import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional

from config import MODELS_DIR, RESSOURCES_DIR

from src.interface.button import Button
from src.interface.theme import Theme

class WormSearchResult:
    """Modern UI management class"""
    def __init__(self,
                 scan_shape: str = "square", 
                 stiching_img: Optional[np.ndarray] = None, 
                 worm_positions: Optional[Dict[str, Tuple[int, int]]] = None,
                 WORM_POSITIONS_PROPORTION = []):
        self.width = 2284
        self.height = 1524  
        self.theme = Theme()
        
        # State variables
        self.analyse = False
        self.add_worms = False
        
        # input parameters
        self.scan_shape = scan_shape
        self.stiching_img = stiching_img
        self.microscope_worm_positions = worm_positions
        
        # Mouse state
        self.mouse_x = 0
        self.mouse_y = 0

        # Create live track window
        self.live_window_x1 = 50
        self.live_window_x2 = 1474 if self.scan_shape == "square" else 1114
        self.live_window_y1 = 500 if self.scan_shape == "square" else 500
        self.live_window_y2 = 1924 if self.scan_shape == "square" else 2248

        self.initial_worm_position_proportion = WORM_POSITIONS_PROPORTION
        
        tab_worm_pos_proportion = []
        for worm in WORM_POSITIONS_PROPORTION:
            x = worm[0]*(self.live_window_y2-self.live_window_y1)+self.live_window_y1
            y = worm[1]*(self.live_window_x2-self.live_window_x1)+self.live_window_x1
            tab_worm_pos_proportion.append([x,y])

        self.mouse_worm_positions = tab_worm_pos_proportion
        
        # Buttons positions
        self.buttons_x = 50 if self.scan_shape == "square" else 1374 - 200
        self.buttons_y = 1174 if self.scan_shape == "square" else 1164
        
        # End
        self.exit = False
        
        # Create buttons with modern styling
        self.buttons = self._create_buttons()

        print("1. Proportion on the image")
        print(WORM_POSITIONS_PROPORTION)
        print("2. Mouse position")
        print(self.mouse_worm_positions)
        print("3. Microscope position")
        print(worm_positions)
        
    def _create_buttons(self) -> Dict[str, Button]:
        """Create all UI buttons with modern styling"""
        buttons = {}
        
        # Main action buttons - bottom center (between y=1224 and y=1424)
        center_x = self.width // 2 - 200  # Center horizontally
        buttons["Analyse"] = Button(self.buttons_x, self.buttons_y, 400, 120, "Go to the analyse", "primary")
        
        # Add or remove worms button - left
        (40, 512), (40+400, 512+220)
        left_x = 20
        left_y_start = 512
        buttons["Add_worm"] = Button(left_x + 60, left_y_start, 300, 120, "Add a worm", "default")
        buttons["Remove_worm"] = Button(left_x + 60 , left_y_start + 140, 300, 120, "Remove a worm", "default") 
        
        # End of program
        buttons["End"] = Button(2150, 40, 100, 50, "Exit", "danger") 
        
        return buttons
    
    def update_button_states(self):
        """Update button states based on current application state"""
        self.buttons["Analyse"].is_active = self.analyse
        
        # Update scan objective buttons
        for name, button in self.buttons.items():
            if name == "Add_worm":
                button.is_active = self.add_worms
            elif name == "Remove_worm":
                button.is_active = not self.add_worms
            
    def update_hover_states(self):
        """Update button hover states based on mouse position"""
        for button in self.buttons.values():
            button.is_hovered = button.contains_point(self.mouse_x, self.mouse_y)
    
    def draw_interface(self) -> np.ndarray:
        """Draw the complete modern interface"""
        img = np.ones((1524, 2284, 3), dtype=np.uint8) * 255  # Create a white image for display
    
        # Draw subtle gradient background
        for i in range(self.height):
            alpha = i / self.height
            gray_value = int(250 - alpha * 20)  # Subtle gradient
            img[i, :] = (gray_value, gray_value, gray_value)
        
        # draw bounding boxes on the stitching image
        img_live = self.stiching_img.copy()
        img_height, img_width = img_live.shape[:2]
        list_of_worm_position = self.initial_worm_position_proportion
        for worm in list_of_worm_position:
            x = worm[0] * img_width
            y = worm[1] * img_height
            cv2.rectangle(img_live, (int(x-15), int(y-15)), (int(x + 15), int(y + 15)), (0, 0, 255), 1)

        # Add the stiching image
        img[self.live_window_x1:self.live_window_x2, self.live_window_y1:self.live_window_y2] = img_live
           
        if self.add_worms:
            # Add text explaining the add/remove mode
            cv2.putText(img, "Click on the image",
                        (110, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.theme.text_secondary, 2) 
            cv2.putText(img, "where you want to add a worm",
                        (28, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.theme.text_secondary, 2) 
        else:
            # Add text explaining the add/remove mode
            cv2.putText(img, "Click on the image",
                        (110, 900), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.theme.text_secondary, 2) 
            cv2.putText(img, "on the worm you want to remove",
                        (23, 950), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.theme.text_secondary, 2) 
        
        # Update button states
        self.update_button_states()
        self.update_hover_states()
        
        # Draw all buttons
        for button in self.buttons.values():
            if not button.phantom:
                button.draw(img)
        
        return img
    
    def handle_click(self, x: int, y: int):
        """Handle mouse click events"""
        for name, button in self.buttons.items():
            if button.contains_point(x, y):
                if name == "Analyse":
                    self.analyse = not self.analyse    
                elif name == "Add_worm":
                    self.add_worms = True
                elif name == "Remove_worm":
                    self.add_worms = False
                elif name == "End":
                    self.exit = True
                return
        
        if self.live_window_x1 < y < self.live_window_x2 and \
           self.live_window_y1 < x < self.live_window_y2:
            # Click inside the live track window
            if self.add_worms:
                # compute the position with 0,0 in the left bottom corner in proportion of the image
                pos_x = 1 - (x - self.live_window_y1)/(self.live_window_y2-self.live_window_y1)
                pos_y = (y - self.live_window_x1)/(self.live_window_x2-self.live_window_x1)
                
                # transform values into microscope positions
                csv_path = Path(RESSOURCES_DIR) / "config_positions.csv"
                params = pd.read_csv(csv_path)
                start_corner_x = float(params['start_corner_x'].iloc[0])
                start_corner_y = float(params['start_corner_y'].iloc[0])
                end_corner_x = float(params['end_corner_x'].iloc[0])
                end_corner_y = float(params['end_corner_y'].iloc[0])
                
                microscope_pos_x = pos_x * (end_corner_x - start_corner_x) + start_corner_x
                microscope_pos_y = pos_y * (end_corner_y - start_corner_y) + start_corner_y
                
                screen_pos_x = pos_y
                screen_pos_y = 1-pos_x

                # Add worm position
                self.microscope_worm_positions.append([microscope_pos_y, microscope_pos_x])
                self.initial_worm_position_proportion.append([screen_pos_y, screen_pos_x])
                self.mouse_worm_positions.append([x, y])
                print(f"Worm added at position: {[pos_y, pos_x]}")
            else:
                # Remove worm position
                size_error = 15
                for i, pos in enumerate(self.mouse_worm_positions):
                    if abs(pos[0] - x) < size_error and abs(pos[1] - y) < size_error:
                        self.mouse_worm_positions.pop(i)
                        self.microscope_worm_positions.pop(i)
                        self.initial_worm_position_proportion.pop(i)
                        break
        
        # Click outside input fields
        self.scan_input_active = False
        self.fluo_input_active = False
             
    def handle_mouse_move(self, x: int, y: int):
        """Handle mouse movement for hover effects"""
        self.mouse_x = x
        self.mouse_y = y

# A SUPPRIMER - permet de faire des tests sans lancer l'interface complète
def mouse_callback(event, x, y, flags, param):
    """Mouse callback function"""
    ui = param
    
    if event == cv2.EVENT_LBUTTONDOWN:
        ui.handle_click(x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        ui.handle_mouse_move(x, y)

def worm_search_result():
    """Main application loop"""
    # Create UI instance
    SCAN_SHAPE = "rectangle" # "square"  
    if SCAN_SHAPE == "square":
        STICHING_IMG = np.ones((1424, 1424, 3), dtype=np.uint8) * 255
    else:
        STICHING_IMG = np.ones((1064, 1748, 3), dtype=np.uint8) * 255
    WORM_POSITIONS = []
    ui = WormSearchResult(SCAN_SHAPE, STICHING_IMG, WORM_POSITIONS)
    
    # Setup OpenCV window
    cv2.namedWindow("Worm Search", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Worm Search", ui.width, ui.height)
    cv2.setMouseCallback("Worm Search", mouse_callback, ui)
    
    # Main loop
    while True:
        # Draw interface
        img = ui.draw_interface()
        
        # Display
        cv2.imshow("Worm Search", img)
        
        # Handle keyboard input
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC to quit
            break
        
        if ui.analyse:
            break
    
    # Cleanup
    cv2.destroyAllWindows()
    
    return ui.worm_positions

if __name__ == "__main__":
    final_state = worm_search_result()
    print("\nFinal System State:")
    print(final_state)
    
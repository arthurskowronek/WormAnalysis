import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional

from config import RESSOURCES_DIR
from src.interface.button import Button
from src.interface.theme import Theme

class WormAcquisitionMenu:
    """Modern UI management class"""
    def __init__(self):
        
        # read parameters from the csv file
        param_file = Path(RESSOURCES_DIR) / "parameters.csv"
        params = pd.read_csv(param_file)
        DUALVIEW = bool(params['dual_view'].iloc[0])
        CAMERA_EXPOSURE_TIME = int(params['exposure_time'].iloc[0])
        OBJECTIVE_MAGNIFICATION_SCAN = int(params['obj_scan'].iloc[0])
        OBJECTIVE_MAGNIFICATION_FLUO = int(params['obj_fluo'].iloc[0])
        USER_DIRECTORY = params['user_directory'].iloc[0]
        
        self.width = 2284
        self.height = 1524  
        self.theme = Theme()
        HelpKeyboardImage = cv2.imread(str(Path(RESSOURCES_DIR) / "HelpKeyboard_Live.png"))
        original_height, original_width = HelpKeyboardImage.shape[:2]
        aspect_ratio = original_width / original_height
        new_width = 510 + 190
        self.HelpKeyboardImage = cv2.resize(HelpKeyboardImage, (new_width, int(new_width / aspect_ratio)))
        
        # State variables
        self.launch = False
        self.dual_view = DUALVIEW
        self.time_exposure = str(CAMERA_EXPOSURE_TIME)
        self.time_exposure_active = False
        
        if OBJECTIVE_MAGNIFICATION_SCAN in [4, 5, 10]:
            self.obj_scan = OBJECTIVE_MAGNIFICATION_SCAN
            self.scan_input = "..."
            self.scan_input_active = False
        else:
            self.obj_scan = OBJECTIVE_MAGNIFICATION_SCAN
            self.scan_input = str(OBJECTIVE_MAGNIFICATION_SCAN)
            self.scan_input_active = True
            
        if OBJECTIVE_MAGNIFICATION_FLUO in [10, 20, 40]:
            self.obj_fluo = OBJECTIVE_MAGNIFICATION_FLUO
            self.fluo_input = "..."
            self.fluo_input_active = False
        else:
            self.obj_fluo = OBJECTIVE_MAGNIFICATION_FLUO
            self.fluo_input = str(OBJECTIVE_MAGNIFICATION_FLUO)
            self.fluo_input_active = True
            
        self.user_directory = USER_DIRECTORY
        self.img_directory_active = False
        
        self.last_button_clicked = None
        
        # Mouse state
        self.mouse_x = 0
        self.mouse_y = 0
        
        # End
        self.exit = False
        
        # Create buttons with modern styling
        self.buttons = self._create_buttons()
        
    def _create_buttons(self) -> Dict[str, Button]:
        """Create all UI buttons with modern styling"""
        buttons = {}
        
        # Main action buttons - bottom center (between y=1224 and y=1424)
        center_x = self.width // 2 - 200  # Center horizontally
        buttons["Launch"] = Button(center_x, 1124, 400, 120, "Start Acquisition", "primary")
        
        # Scan objective buttons - bottom right, vertical arrangement
        scan_x = self.width - 820  # Right side
        scan_y_start = 1150
        buttons["Scan_4"] = Button(scan_x, scan_y_start, 130, 65, "4x", "default")
        buttons["Scan_5"] = Button(scan_x, scan_y_start + 80, 130, 65, "5x", "default")
        buttons["Scan_10"] = Button(scan_x, scan_y_start + 160, 130, 65, "10x", "default")
        buttons["Scan_Input"] = Button(scan_x, scan_y_start + 240, 130, 65, "", "default")
        
        # Fluorescence objective buttons - next to scan buttons
        fluo_x = scan_x + 200  # Next to scan buttons
        buttons["Fluo_10"] = Button(fluo_x, scan_y_start, 130, 65, "10x", "default")
        buttons["Fluo_20"] = Button(fluo_x, scan_y_start + 80, 130, 65, "20x", "default")
        buttons["Fluo_40"] = Button(fluo_x, scan_y_start + 160, 130, 65, "40x", "default")
        buttons["Fluo_Input"] = Button(fluo_x, scan_y_start + 240, 130, 65, "", "default")
        
        # View and shape controls - bottom left, vertical arrangement
        left_x = 640
        left_y_start = 1150
        buttons["DualView"] = Button(left_x, left_y_start + 50, 180, 80, "Dual View", "danger")
        buttons["Time_exposure"] = Button(left_x, left_y_start + 170, 180, 80, "", "default")
        
        # Img directory button - bottom left
        buttons["Img_directory"] = Button(center_x-50, 1315, 500, 100, "", "default")
        
        # End of program
        buttons["End"] = Button(2140, 0, 100, 50, "Exit", "danger") 
        
        return buttons
    
    def draw_section_labels(self, img: np.ndarray):
        """Draw section labels for better organization"""
        
        # Time exposure section
        cv2.putText(img, "Time exposure (ms) :", (380, 1370), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.theme.text_secondary, 2)
        
        # Bottom right section labels
        scan_x = self.width - 835
        cv2.putText(img, "Scan Objective", (scan_x, 1125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.theme.text_primary, 2)
        
        cv2.putText(img, "Text box (if you use another objective)", (scan_x+370, 1415),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.theme.text_secondary, 2)
        cv2.putText(img, "Validate input with 'enter'", (scan_x+450, 1445),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.theme.text_secondary, 2)
        
        fluo_x = scan_x + 200
        cv2.putText(img, "Fluo Objective", (fluo_x, 1125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.theme.text_primary, 2)
        
        cv2.putText(img, "Name of the directory where images will be saved", (867, 1450),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.theme.text_secondary, 2)
    
    def update_button_states(self):
        """Update button states based on current application state"""
        self.buttons["Launch"].is_active = self.launch
        self.buttons["DualView"].is_active = self.dual_view
        
        # Update scan objective buttons
        for name, button in self.buttons.items():
            if name.startswith("Scan_") and not name.endswith("_Input"):
                obj_value = float(name.split("_")[1])
                button.is_active = (self.obj_scan == obj_value) and not self.scan_input_active
            elif name.startswith("Fluo_") and not name.endswith("_Input"):
                obj_value = float(name.split("_")[1])
                button.is_active = (self.obj_fluo == obj_value) and not self.fluo_input_active
            elif name == "Scan_Input":
                button.is_active = self.scan_input_active
                button.text = self.scan_input + ("_" if self.scan_input_active else "")
            elif name == "Fluo_Input":
                button.is_active = self.fluo_input_active
                button.text = self.fluo_input + ("_" if self.fluo_input_active else "")
            elif name == "Time_exposure":
                button.is_active = self.time_exposure_active
                button.text = self.time_exposure + ("_" if self.time_exposure_active else "")
            elif name == "Img_directory":
                button.is_active = self.img_directory_active
                button.text = self.user_directory + ("_" if self.img_directory_active else "")
                
    
    def update_hover_states(self):
        """Update button hover states based on mouse position"""
        for button in self.buttons.values():
            button.is_hovered = button.contains_point(self.mouse_x, self.mouse_y)
    
    def draw_interface(self) -> np.ndarray:
        """Draw the complete modern interface"""
        # Create base image with gradient background
        img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        
        # Draw subtle gradient background
        for i in range(self.height):
            alpha = i / self.height
            gray_value = int(250 - alpha * 20)  # Subtle gradient
            img[i, :] = (gray_value, gray_value, gray_value)
        
        # Draw section labels
        self.draw_section_labels(img)
        
        # Draw a white square of 1024 pixels in (480,100)
        cv2.rectangle(img, (480, 50), (1504, 1074), (255, 255, 255), -1)
        
        # Draw a black rectangle (for the map)
        cv2.rectangle(img, (1544, 50), (1544+700, 50+700), (0, 0, 0), -1)
        
        # Draw a red rectangle (for the warning message)
        cv2.rectangle(img, (40, 512), (40+400, 512+220), (100, 100, 220), -1)
        # add text to the warning message
        cv2.putText(img, "Warning", (175, 560), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.putText(img, "Be sure to use", (140, 620), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, "the L camera", (148, 660), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add the help keyboard image
        img[800:800+280, 1544:1544+700] = self.HelpKeyboardImage
        
        
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
                if name == "Launch":
                    self.launch = not self.launch
                    self.last_button_clicked = name
                elif name.startswith("Scan_") and not name.endswith("_Input"):
                    self.obj_scan = float(name.split("_")[1])
                    self.last_button_clicked = name
                elif name.startswith("Fluo_") and not name.endswith("_Input"):
                    self.obj_fluo = float(name.split("_")[1])
                    self.last_button_clicked = name
                elif name == "DualView":
                    self.dual_view = not self.dual_view
                    self.last_button_clicked = name
                elif name == "Scan_Input":
                    self.scan_input_active = not self.scan_input_active
                    self.last_button_clicked = name
                elif name == "Fluo_Input":
                    self.fluo_input_active = not self.fluo_input_active
                    self.last_button_clicked = name
                elif name == "Time_exposure":
                    self.time_exposure_active = not self.time_exposure_active
                    self.last_button_clicked = name
                elif name == "Img_directory":
                    self.img_directory_active = not self.img_directory_active
                    self.last_button_clicked = name
                elif name == "End":
                    self.exit = True
                return
        
        # Click outside input fields
        self.scan_input_active = False
        self.fluo_input_active = False
        self.time_exposure_active = False
        self.img_directory_active = False
        
    
    def handle_key(self, key: int):
        """Handle keyboard input"""
        if self.scan_input_active and self.last_button_clicked == "Scan_Input":
            if key == 13:  # Enter
                self.scan_input_active = False
            elif key == 8 or key == 127:  # Backspace
                self.scan_input = self.scan_input[:-1]
            elif key == 27:  # Escape
                self.scan_input_active = False
            elif (48 <= key <= 57):  # Numbers 
                self.scan_input += chr(key)
        elif self.fluo_input_active and self.last_button_clicked == "Fluo_Input":
            if key == 13:  # Enter
                self.fluo_input_active = False
            elif key == 8 or key == 127:  # Backspace
                self.fluo_input = self.fluo_input[:-1]
            elif key == 27:  # Escape
                self.fluo_input_active = False
            elif (48 <= key <= 57):  # Numbers 
                self.fluo_input += chr(key)
        elif self.time_exposure_active and self.last_button_clicked == "Time_exposure":
            if key == 13:  # Enter
                self.time_exposure_active = False
            elif key == 8 or key == 127:  # Backspace
                self.time_exposure = self.time_exposure[:-1]
            elif key == 27:  # Escape
                self.time_exposure_active = False
            elif (48 <= key <= 57):  # Numbers 
                self.time_exposure += chr(key)
        elif self.img_directory_active and self.last_button_clicked == "Img_directory":
            if key == 13:
                self.img_directory_active = False
            elif key == 8 or key == 127:  # Backspace
                self.user_directory = self.user_directory[:-1]
            elif key == 27:  # Escape
                self.img_directory_active = False
            elif (48 <= key <= 57) or (65 <= key <= 90) or (97 <= key <= 122) or (key in [95, 45]):  # Alphanumeric and special characters
                self.user_directory += chr(key)
                
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

def worm_assist_acquisition_menu():
    """Main application loop"""
    # Create UI instance
    ui = WormAcquisitionMenu()
    
    # Setup OpenCV window
    cv2.namedWindow("Worm Acquisition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Worm Acquisition", ui.width, ui.height)
    cv2.setMouseCallback("Worm Acquisition", mouse_callback, ui)
    
    # Main loop
    while True:
        # Draw interface
        img = ui.draw_interface()
        
        # Display
        cv2.imshow("Worm Acquisition", img)
        
        # Handle keyboard input
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC to quit
            break
        elif key != 255:  # Any other key
            ui.handle_key(key)
        
        if ui.launch:
            break
    
    # Cleanup
    cv2.destroyAllWindows()
    
    # Return final state
    return {
        'obj_scan': ui.obj_scan if not ui.scan_input_active else ui.scan_input.replace('.', ''),  
        'obj_fluo': ui.obj_fluo if not ui.fluo_input_active else ui.fluo_input.replace('.', ''),
        'dual_view': ui.dual_view,
        'exposure_time': ui.time_exposure,
        'user_directory': ui.user_directory.replace(' ', '_')
    }

if __name__ == "__main__":
    final_state = worm_assist_acquisition_menu()
    print("\nFinal System State:")
    print("===================")
    for key, value in final_state.items():
        print(f"{key}: {value}")
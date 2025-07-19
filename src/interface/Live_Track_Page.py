import os
import cv2
import time
import random
import shutil
import keyboard
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from tifffile import imwrite
from ultralytics import YOLO
from typing import Dict, Tuple, Optional

from config import RESSOURCES_DIR, MODELS_DIR, USER_DIR, DATA_DIR
from src.interface.button import Button
from src.interface.theme import Theme
from src.system.dataset import Dataset

class LiveTrackPage:
    def __init__(self, mmc, init_pos_x, init_pos_y, best_scaler, best_model, USER_DIRECTORY = "Arthur_2025_07_16", worm_positions = None):
        
        # graphics parameters
        self.width = 2284
        self.height = 1524  
        self.theme = Theme()
        HelpKeyboardImage = cv2.imread(str(Path(RESSOURCES_DIR) / "HelpKeyboard_AssistAcquisition.png"))
        original_height, original_width = HelpKeyboardImage.shape[:2]
        aspect_ratio = original_width / original_height
        new_width = 700
        self.HelpKeyboardImage = cv2.resize(HelpKeyboardImage, (new_width, int(new_width / aspect_ratio)))
        
        LoadingImage = cv2.imread(str(Path(RESSOURCES_DIR) / "loading.png"))
        self.loading = cv2.resize(LoadingImage, (300, 300))
        
        # input parameters
        self.init_pos_x = init_pos_x
        self.init_pos_y = init_pos_y
        self.live_img = None
        self.pos_x = 0
        self.pos_y = 0
        self.worm_positions = worm_positions
        self.CORE = mmc
        self.user_directory = USER_DIRECTORY
        self.best_scaler = best_scaler
        self.best_model = best_model
        self._last_raw_frame = None
        
        # Model
        self.segmentation_model = YOLO(Path(MODELS_DIR) / "YOLO_segmentation.pt")


        # state parameters
        self.analyse = False
        self.next_worm = False
        self.last_worm = False
        self.next_mutant = False
        self.last_mutant = False
        self.classify_WT = False
        self.classify_mutant = False
        self.save_current_image = False
        self.trackbars_visible = False 
        self.trackbar_window = "Histogram"
        
        # Mouse state
        self.mouse_x = 0
        self.mouse_y = 0

        # End
        self.end = False
        
        # Create buttons with modern styling
        self.buttons = self._create_buttons()
          
    def set_image_position(self, img, pos_x, pos_y):
        # 1) resize
        img = cv2.resize(img, (1024, 1024))

        # 2) full‑range clip/stretch
        #    if your raw is 16‑bit, use its full range here:
        MAX_RAW = img.dtype == np.uint16 and 65535 or 255
        mn, mx = 0, MAX_RAW
        img_f = np.clip(img.astype(np.float32), mn, mx)
        img_u8 = ((img_f - mn) / (mx - mn) * 255.0).astype(np.uint8)

        # 3) convert to RGB for GUI
        img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)

        # 4) store for drawing
        self.live_img = img_rgb
        self.pos_x   = pos_x
        self.pos_y   = pos_y

    def _create_buttons(self) -> Dict[str, Button]:
        """Create all UI buttons with modern styling"""
        buttons = {}

        center_x = self.width // 2 - 200  # Center horizontally
        
        buttons["Analyse"] = Button(center_x, 1200, 400, 200, "Launch analysis", "primary")
        
        # Classification
        buttons["Classify_as_WT"] = Button(center_x - 700, 1164, 350, 120, "Classify as WT", "secondary") 
        buttons["Classify_as_Mutant"] = Button(center_x - 700, 1324, 350, 120, "Classify as mutant", "primary") 

        # Arrow buttons - move through worms
        buttons["Next_Worm"] = Button(center_x + 935, 1320, 140, 80, ">", "default") 
        buttons["Last_Worm"] = Button(center_x + 635, 1320, 140, 80, "<", "default") 
        buttons["Next_Mutant"] = Button(center_x + 785, 1220, 140, 80, "^", "default") 
        buttons["Last_Mutant"] = Button(center_x + 785, 1320, 140, 80, "v", "default") 
        
        # Save current image
        buttons["Saving"] = Button(center_x - 300, 1254, 200, 100, "Save image", "default")
        
        # Brightness buttons
        buttons["ToggleWindowing"] = Button(1250, 50, 250, 40, "", "secondary")

        
        # End of program
        buttons["End"] = Button(2150, 40, 100, 50, "Exit", "danger") 
        
        return buttons

    def update_button_states(self):
        """Update button states based on current application state"""
        
        # Update scan objective buttons
        for name, button in self.buttons.items():
            if name == "Analyse":
                button.is_active = self.analyse
            elif name == "Classify_as_WT":
                button.is_active = self.classify_WT
            elif name == "Classify_as_Mutant":
                button.is_active = self.classify_mutant
            elif name == "Next_Worm":
                button.is_active = self.next_worm
            elif name == "Last_Worm":
                button.is_active = self.last_worm
            elif name == "Next_Mutant":
                button.is_active = self.next_mutant
            elif name == "Last_Mutant":
                button.is_active = self.last_mutant
            elif name == "Saving":
                button.is_active = self.save_current_image

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
        
        # Add the live image
        img[100:1124, 480:1504] = self.live_img
        
        # Add the table of worm positions
        img_table = self.worm_positions.show_table_worms_positions()
        height, width = img_table.shape[:2]
        img[100:100+height, 40:40+width] = img_table
        
        # Add the help keyboard image
        img[850:850+280, 1544:1544+700] = self.HelpKeyboardImage
        
        # Add texts
        cv2.putText(img, "Go to next worm", (2050, 1370), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.theme.text_secondary, 2)
        cv2.putText(img, "Go to next mutant", (1700, 1200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.theme.text_secondary, 2)
        
        # Update button states
        self.update_button_states()
        self.update_hover_states()
        
        # Draw all buttons
        for button in self.buttons.values():
            if not button.phantom:
                button.draw(img)

        # Buttons text
        cv2.putText(img, "Brightness/Contrast", (1265, 78), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.theme.text_secondary, 2)
                
        # Reset flags so buttons deactivate in the next frame
        self.reset_button_flags()
        
        return img
    
    def handle_click(self, x: int, y: int):
        """Handle mouse click events"""

        for name, btn in self.buttons.items():
            if btn.contains_point(x, y):
                self.last_clicked = name
                if name == "End":
                    self.end = True
                elif name == "Analyse":
                    cv2.namedWindow("Loading", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Loading", 300, 300)
                    cv2.moveWindow("Loading", 1000, 600)
                    cv2.imshow("Loading", self.loading)
                    cv2.waitKey(1)   # force a refresh
                    
                    self.analyse = True
                    self.analyse_worm()
                    
                    cv2.destroyWindow("Loading")
                elif name == "Next_Worm":
                    self.next_worm = True
                    self.go_to_next_worm()
                elif name == "Last_Worm":
                    self.last_worm = True
                    self.go_to_last_worm()
                elif name == "Next_Mutant":
                    self.next_mutant = True
                    self.go_to_next_mutant()
                elif name == "Last_Mutant":
                    self.last_mutant = True
                    self.go_to_last_mutant()
                elif name == "Classify_as_WT":
                    self.classify_WT = True
                    self.classify_as_wt()
                elif name == "Classify_as_Mutant":
                    self.classify_mutant = True
                    self.classify_as_mutant()
                elif name == "Saving":
                    self.save_current_image = True
                    self.save_image()
                elif name == "ToggleWindowing":
                    self.trackbars_visible = not self.trackbars_visible
                    if self.trackbars_visible:
                        self._frozen_raw = self._last_raw_frame.copy()
                        self._create_trackbar_window()
                    else:
                        cv2.destroyWindow(self.trackbar_window)
                return
    
    def handle_mouse_move(self, x: int, y: int):
        self.mouse_x = x
        self.mouse_y = y

    def handle_key(self, key):
        
        if keyboard.is_pressed('droite'): # Move to next worm
            self.go_to_next_worm()
        elif keyboard.is_pressed('gauche'): # Move to last worm
            self.go_to_last_worm()
        elif keyboard.is_pressed('haut'): # Move to next mutant
            self.go_to_next_mutant()
        elif keyboard.is_pressed('bas'): # Move to last mutant
            self.go_to_last_mutant()
        elif keyboard.is_pressed('backspace'): # Erase the worm from the memory
            self.remove_worm()
        elif key in map(ord, ['n', 'b', 'h', 'j']): # Save worm position
            self.add_new_worm()
        elif key in map(ord, ['p', 'o', 'l', 'm']): # Save image
           self.save_image()
        elif key == 13 or keyboard.is_pressed('enter'): # Class worm as Wild-Type
            self.classify_as_wt()
        elif key in map(ord, ['w', 'q', 's', 'x', 'a', 'z', 'e', 'd', 'c']): # Class worm as Mutant
            self.classify_as_mutant()
        elif key == ord(' '): # Analyzing worm
            cv2.namedWindow("Loading", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Loading", 300, 300)
            cv2.moveWindow("Loading", 1000, 600)
            cv2.imshow("Loading", self.loading)
            cv2.waitKey(1)   # force a refresh
            
            self.analyse_worm()
            
            cv2.destroyWindow("Loading")
    
    def reset_button_flags(self):
        """Reset all one-time-use button flags after each action"""
        self.analyse = False
        self.next_worm = False
        self.last_worm = False
        self.next_mutant = False
        self.last_mutant = False
        self.classify_WT = False
        self.classify_mutant = False
        self.save_current_image = False    

    def _create_trackbar_window(self):
        MAX_RAW = 65535
        W = 700        # total window width
        H = 700        # total window height
        HIST_H = 300   # histogram height
        IMG_H = H - HIST_H

        cv2.namedWindow(self.trackbar_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.trackbar_window, W, H)
        cv2.moveWindow(self.trackbar_window, 100, 1544)

        def on_trackbar(_):
            # --- read sliders ---
            mn = cv2.getTrackbarPos("Min", self.trackbar_window)
            mx = cv2.getTrackbarPos("Max", self.trackbar_window)
            mn, mx = min(mn, mx), max(mn, mx)

            # --- window‑level & colorize frozen frame ---
            img = self._frozen_raw
            img = cv2.resize(img, (IMG_H, IMG_H))
            img_f = np.clip(img.astype(np.float32), mn, mx)
            img_u8 = ((img_f - mn) / (mx - mn) * 255.0).astype(np.uint8)
            img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)

            # --- build histogram over full 0–MAX_RAW range ---
            hist = cv2.calcHist([self._frozen_raw], [0], None, [256], [0, MAX_RAW]).flatten()
            hist = (hist / hist.max() * HIST_H).astype(np.int32)
            hist_img = np.zeros((HIST_H, W, 3), dtype=np.uint8)
            bin_w = int(np.ceil(W / hist.size))
            for i, h in enumerate(hist):
                x1 = i * bin_w
                x2 = min((i+1)*bin_w - 1, W-1)
                cv2.rectangle(hist_img,
                            (x1, HIST_H - h),
                            (x2, HIST_H),
                            (200, 200, 200),
                            thickness=-1)

            # --- draw the Min→Max diagonal line ---
            # map slider values into [0…255] bins → pixel x**
            bin_min = int(mn   / MAX_RAW * (hist.size - 1))
            bin_max = int(mx   / MAX_RAW * (hist.size - 1))
            x_min   = int(bin_min * bin_w + bin_w/2)
            x_max   = int(bin_max * bin_w + bin_w/2)

            # draw a red line from bottom at x_min up to top at x_max**
            cv2.line(hist_img,
                    (x_min, HIST_H),
                    (x_max, 0),
                    (0, 0, 255),    # red BGR
                    2)              # thickness**

            # --- stack & show ---
            img_rgb_resized = cv2.resize(img_rgb, (W, IMG_H))
            canvas = np.vstack((hist_img, img_rgb_resized))
            cv2.imshow(self.trackbar_window, canvas)

        # create trackbars
        cv2.createTrackbar("Min", self.trackbar_window,   0,       MAX_RAW, on_trackbar)
        cv2.createTrackbar("Max", self.trackbar_window,   MAX_RAW, MAX_RAW, on_trackbar)

        # initial draw
        on_trackbar(None)

    def _destroy_trackbar_window(self):
        if cv2.getWindowProperty(self.trackbar_window, 0) >= 0:
            cv2.destroyWindow(self.trackbar_window)


    # ----- Action methods -----
    def find_worm_segmentation(self, img):
        """
        Segment worm from background using YOLO
        
        Args:
            img: Input image (2D grayscale or 3D color)
            
        Returns:
            img after applying mask on the segmentation (same shape as input)
        """
        
        model = self.segmentation_model
        image = img.copy()
        
        # Normalize image for YOLO
        """threshold = 3000
        image = np.clip(image, 0, threshold).astype(np.uint16)"""
        
        # Normalize image for YOLO
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # Save temporary image
        temp_path = Path(MODELS_DIR) / "temp_converted_image.png"
        cv2.imwrite(str(temp_path), image)
        
        # Predict
        prediction = model.predict(source=str(temp_path), save=False, verbose=False)
        os.remove(temp_path)
        
        masks = prediction[0].masks
        
        if masks is None or masks.data.shape[0] == 0:
            # No mask detected
            return np.zeros_like(image)

        # Get image center
        h, w = image.shape[:2]
        center = np.array([w // 2, h // 2])

        # Find the mask closest to the center
        min_dist = float('inf')
        closest_mask = None

        for i, mask in enumerate(masks.data):
            mask = mask.cpu().numpy()
            yx = np.column_stack(np.nonzero(mask))
            if yx.size == 0:
                continue
            xy = yx[:, ::-1]  # (x, y)

            distances = np.linalg.norm(xy - center, axis=1)
            min_distance = distances.min()

            if min_distance < min_dist:
                min_dist = min_distance
                closest_mask = mask
        
        resized_mask = cv2.resize(closest_mask.astype(np.uint8), (w,h), interpolation=cv2.INTER_NEAREST)
        mask_bool = resized_mask.astype(bool)

        result = np.zeros_like(image)

        if image.ndim == 2:
            # Image grayscale 2D
            result[mask_bool] = image[mask_bool]
        else:
            # Image couleur 3D (rare dans ton cas)
            for c in range(image.shape[2]):
                result[..., c][mask_bool] = image[..., c][mask_bool]

        return result

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

    def go_to_next_mutant(self):
        self.worm_positions.go_to_next_mutant()
        id = self.worm_positions.get_id_worm_seen()
        x,y = self.worm_positions.get_worm_position(id)
        time.sleep(0.01)
        self.CORE.setXYPosition(self.CORE.getXYStageDevice(), x, y)

    def go_to_last_mutant(self):
        self.worm_positions.go_to_last_mutant()
        id = self.worm_positions.get_id_worm_seen()
        x,y = self.worm_positions.get_worm_position(id)
        time.sleep(0.01)
        self.CORE.setXYPosition(self.CORE.getXYStageDevice(), x, y)

    def classify_as_wt(self):
        id = self.worm_positions.get_id_worm_seen()
        self.worm_positions.update_worm_label(id, 'Wild-Type')
        
        # save image in the corresponding directory
        filename = f"{id}.tif"
        WT_path = Path(DATA_DIR) / "WT_prediction" / filename
        Mutant_path = Path(DATA_DIR) / "Mutant_prediction" / filename
        final_directory = Path(DATA_DIR) / "WT"
        file_count = len(list(final_directory.glob("*")))
        new_filename = f"WT_{file_count}.tif"
        classified_path = final_directory / new_filename
        if WT_path.exists() or Mutant_path.exists():
            unclassified_path = WT_path if WT_path.exists() else Mutant_path
            shutil.move(str(unclassified_path), str(classified_path))
            
            # update label in the big dataset
            big_dataset = Dataset()
            big_dataset.load_images(compute=False, name_dataset="big_dataset")
            big_dataset.update_label_by_filename(filename, "WT", new_filename)
            
            # update model_performance file
            new_line = {
                'date': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')],
                'best_scaler_name': [self.best_scaler],
                'best_model_name': [self.best_model],
                'label_predicted': [self.worm_positions.get_worm_prediction(id)],
                'label_true': ["WT"]
            }
            df_new_results = pd.DataFrame(new_line)
            csv_path = Path(MODELS_DIR) / "model_performance.csv"
            df_existing_results = pd.read_csv(csv_path)
            df_combined_results = pd.concat([df_existing_results, df_new_results], ignore_index=True)
            df_combined_results.to_csv(csv_path, index=False, mode='w')
            
        else:
            img = self.find_worm_segmentation(self.live_img)
            cv2.imwrite(str(classified_path), img)

    def classify_as_mutant(self):
        id = self.worm_positions.get_id_worm_seen()
        self.worm_positions.update_worm_label(id, 'Mutant')
        
        # save image in the corresponding directory
        filename = f"{id}.tif"
        WT_path = Path(DATA_DIR) / "WT_prediction" / filename
        Mutant_path = Path(DATA_DIR) / "Mutant_prediction" / filename
        final_directory = Path(DATA_DIR) / "Mutant"
        file_count = len(list(final_directory.glob("*")))
        new_filename = f"Mut_{file_count}.tif"
        classified_path = final_directory / new_filename
        if WT_path.exists() or Mutant_path.exists():
            unclassified_path = WT_path if WT_path.exists() else Mutant_path
            shutil.move(str(unclassified_path), str(classified_path))
            
            # update label in the big dataset
            big_dataset = Dataset()
            big_dataset.load_images(compute=False, name_dataset="big_dataset")
            big_dataset.update_label_by_filename(filename, "Mutant", new_filename)
            
            # update model_performance file
            new_line = {
                'date': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')],
                'best_scaler_name': [self.best_scaler],
                'best_model_name': [self.best_model],
                'label_predicted': [self.worm_positions.get_worm_prediction(id)],
                'label_true': ["Mutant"]
            }
            df_new_results = pd.DataFrame(new_line)
            csv_path = Path(MODELS_DIR) / "model_performance.csv"
            df_existing_results = pd.read_csv(csv_path)
            df_combined_results = pd.concat([df_existing_results, df_new_results], ignore_index=True)
            df_combined_results.to_csv(csv_path, index=False, mode='w')
        else:
            img = self.find_worm_segmentation(self.live_img)
            cv2.imwrite(str(classified_path), img)

    def analyse_worm(self):
        # Step 1: Segment the image and save it
        img = self.find_worm_segmentation(self.live_img)
        id = self.worm_positions.get_id_worm_seen()
        unclassified_path = Path(DATA_DIR) / "Unclassified" / f"{id}.tif"
        imwrite(str(unclassified_path), img)
        
        # Step 2: Try to predict with model, fallback to random
        try:
            dataset = Dataset()
            dataset.load_images()
            dataset.set_features()
            model = dataset.get_model()
            pred = model.predict(dataset.get_features_selected()[0])[0]
            print(f"Model-derived prediction : {pred:.2f}")
            
            big_dataset = Dataset()
            big_dataset.load_images(compute=False, name_dataset="big_dataset")
            big_dataset.merge_with(dataset)
        except Exception as e:
            pred = random.uniform(0.01, 1.0)
            print(f"Random prediction (error : {e}) → {pred:.2f}")
            time.sleep(2)
        
        # Step 3: Save image in the corresponding directory 
        directory = Path(DATA_DIR) / ("Mutant_prediction" if pred > 0.5 else "WT_prediction")
        classified_path = directory / f"{id}.tif" 
        shutil.move(str(unclassified_path), str(classified_path))
        
        # Step 4: Update prediction in worm database
        self.worm_positions.update_worm_prediction(id, pred)

    def save_image(self):
         # save image in the user directory
        print("Saving image in the user directory")
        filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.tif"
        user_directory = Path(USER_DIR) / self.user_directory
        if not user_directory.exists():
            user_directory.mkdir(parents=True, exist_ok=True)
        image_path = user_directory / filename
        cv2.imwrite(str(image_path), self.live_img)

    def add_new_worm(self):
        self.worm_positions.add_worm_position(self.pos_x, self.pos_y)
        
    def remove_worm(self):
        id = self.worm_positions.get_id_worm_seen()
        self.worm_positions.delete_worm(id)

    def end_of_program(self):
        self.CORE.setXYPosition(self.CORE.getXYStageDevice(), self.init_pos_x, self.init_pos_y)
            
        # train model with new data
        big_dataset = Dataset()
        big_dataset.set_features(compute=False, name_dataset="big_dataset")
        big_dataset.remove_unclassified()
        big_dataset.get_model(compute=True)
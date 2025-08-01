
import cv2
import datetime
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from config import RESSOURCES_DIR, MODELS_DIR, USER_DIR
from src.interface.theme import Theme
from src.system.dataset_manager import Dataset_Manager

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




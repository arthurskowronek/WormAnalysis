import os
import sys
import cv2
import yaml
import glob
import time
import shutil
import datetime
import subprocess
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from tifffile import imwrite
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageColor, ImageDraw
from tkinter.scrolledtext import ScrolledText
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        
from config import RESSOURCES_DIR, DATA_DIR, MODELS_DIR, USER_DIR, LOG_DIR, TRAINING_DIR, PARAMETERS_FILE, DATE_FORMAT, EXPOSURE_TIME_LIVE_CONFIG, NAME_CAMERA, EXPOSURE_TIME_ANALYSIS, MICROSCOPE, load_config_file, log_error, increment_user_statistics, update_user_statistics, clear_scan_directory, log_debug_coordinate

from src.interface.Tooltip import Tooltip
from src.system.ScanSlice import ScanSlice
from src.interface.colorTheme import ColorTheme
from src.system.dataset_manager import Dataset_Manager
from src.system.Worm_Position_Manager import WormPositionManager
from src.system.preprocessing import Preprocessing

class WormAnalysisApp:
    def __init__(self, root, mmc = None, initial_dark_mode=True, first_page = "automatic_scan", initial_show_parameters = True, initial_live_image = True, initial_id_worm_seen = 0):
        """
        Initializes the Worm Analysis Application interface.

        Parameters:
            root (tk.Tk): The root window of the Tkinter application.
            mmc (object): The microscope core object for communication with hardware. Defaults to None.
            initial_dark_mode (bool): If True, enables dark mode by default. Defaults to False.
            first_page (str): The initial page to display. Options include:
                              "automatic_scan", "scan_result", "load_position", "documentation", "tutorial", "machine_config".
                              Defaults to "automatic_scan".
            initial_show_parameters (bool): If True, shows parameters section on startup. Defaults to True.
            initial_live_image (bool): If True, enables live image display on startup. Defaults to True.

        This constructor:
            - Sets up main window dimensions and title.
            - Initializes state variables and default values (e.g., prediction score, mutation proportion).
            - Loads and binds parameter settings from a YAML configuration file.
            - Sets up the visual theme (colors, fonts, icons).
            - Initializes and displays the main frame layout.
            - Displays the specified initial page.
        """
        self.root = root
        self.CORE = mmc
        self.root.title("Worm Analysis")
        self.root.geometry("1550x960")
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.PARAMS_FILE = PARAMETERS_FILE
        self.context_error = ""
        
        # keep track of scheduled after callbacks
        if not hasattr(self, "_after_ids"):
            self._after_ids = []

        # flag used to stop live loops safely
        if not hasattr(self, "_live_running"):
            self._live_running = False

        # load the heavy model only once (guard)
        if not hasattr(self, "segmentation_model") or self.segmentation_model is None:
            self.segmentation_model = YOLO(Path(MODELS_DIR) / "YOLO_segmentation.pt")


        # Initialize variables
        self.show_parameters = initial_show_parameters
        if self.CORE is None:
            if first_page == "training":
                self.current_page = "training"
            else:
                self.current_page = "loading_page"
        else:
            self.current_page = first_page if self.CORE is not None else "loading_page"
        self.dark_mode = initial_dark_mode
        self.worms_position = None
        self.prediction = "--"
        self.id_worm_seen = initial_id_worm_seen
        self.worm_scan_result_mode = "add"
        self.live_image = initial_live_image
        self.bounding_box_size = 15 # Size of the bounding box around worms in pixels
        self.loaded_params = load_config_file()
        self.set_parameters()
        self.enable_parameters_buttons = ["exposure_time","binning","shutter","dual_view","display_mode","scan_objective","fluo_objective","scan_shape", "model_name", "fov_size_um"]
        self.list_files_to_annotate = []

        # Initialize preprocessing
        self.preprocessing = Preprocessing()

        self.contrast_win = None
        self.vmin_var = None
        self.vmax_var = None
        self.hist_fig = None
        self.hist_ax = None
        self.hist_canvas = None
        self.last_hist_update_time = 0.0
        self.hist_update_interval = 1  # seconds: 1Hz histogram updates
        self._contrast_slider_active = False
        self._sensor_min_possible = 0
        self._sensor_max_possible = 65535
        self._image_update_lock = False
        self.save_in_live_mode = False

        self._dragging_worm_id = None            # id du ver en cours de déplacement
        self._drag_offset_prop = (0.0, 0.0)      # offset entre le centre de la box et le point cliqué (en proportions)


        # Theme (color, font, icon)
        self.font = 'Inter'
        self.update_colors()
        self.set_color_theme()
        self.load_icon()
        
        # Main container
        self.main_frame = tk.Frame(root, bg=self.colors.theme["primary_background"])
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.create_layout()

        # Show appropriate page
        if self.current_page == "automatic_scan":
            self.show_automatic_scan_page()
        elif self.current_page == "scan_result":   
            self.show_result_scan_page()
        elif self.current_page == "load_position":
            self.show_load_position_page()
        elif self.current_page == "length_analysis":
            self.show_length_analysis_page()
        elif self.current_page == "documentation":
            self.show_documentation_page()
        elif self.current_page == "configuration":
            self.show_machine_configuration_page()
        elif self.current_page == "training":
            self.show_training_model_page()
        elif self.current_page == "loading_page":
            self.show_loading_page()
    
    # --- Initalization helper function ---
    def _cleanup_for_reinit(self):
        """
        Minimal cleanup before re-initializing the UI.
        Cancels scheduled after callbacks, stops live loops and unbinds/destroys
        widgets that background callbacks might touch.
        This is intentionally minimal to avoid large refactors.
        """
        # Stop live loop (if you use such a loop, it should check this flag)
        try:
            self._live_running = False
        except Exception:
            pass

        # Cancel all stored after callbacks
        try:
            if hasattr(self, "_after_ids"):
                for aid in list(self._after_ids):
                    try:
                        self.root.after_cancel(aid)
                    except Exception:
                        pass
                self._after_ids.clear()
        except Exception:
            pass

        # Unbind/destroy widgets that background callbacks might update
        try:
            if hasattr(self, "img_label") and getattr(self, "img_label") is not None:
                try:
                    self.img_label.unbind("<Button-1>")
                    self.img_label.unbind("<B1-Motion>")
                except Exception:
                    pass
                # don't force-destroy here if you prefer pack_forget(); we try safe destroy
                try:
                    if self.img_label.winfo_exists():
                        self.img_label.destroy()
                except Exception:
                    pass
        except Exception:
            pass

        try:
            if hasattr(self, "live_image_label") and getattr(self, "live_image_label") is not None:
                try:
                    if self.live_image_label.winfo_exists():
                        self.live_image_label.destroy()
                except Exception:
                    pass
        except Exception:
            pass

        # Try to close/cleanup worms_position if it has a close/stop method
        try:
            if hasattr(self, "worms_position") and self.worms_position is not None:
                if hasattr(self.worms_position, "close"):
                    try:
                        self.worms_position.close()
                    except Exception:
                        pass
        except Exception:
            pass
        
        # Close histogram window before UI refresh
        try:
            if hasattr(self, "contrast_win") and self.contrast_win:
                if self.contrast_win.winfo_exists():
                    self.contrast_win.destroy()
            self.contrast_win = None
            
            # Clean up matplotlib resources
            if hasattr(self, "hist_fig"):
                plt.close(self.hist_fig)
                del self.hist_fig
            if hasattr(self, "hist_ax"):
                del self.hist_ax
            if hasattr(self, "hist_canvas"):
                del self.hist_canvas
        except Exception:
            pass

    def set_parameters(self):
        """
        Initializes and binds UI parameters using Tkinter variables.

        This method:
            - Initializes `StringVar` and `BooleanVar` objects for each parameter.
            - Sets default values from the loaded YAML configuration (`self.loaded_params`).
            - Adds trace callbacks to automatically save the parameters when they change.
            - In the case of the 'shape' parameter, also triggers a resize of the scan content area.

        Parameters initialized include:
            - shape, exposure_time, binning, shutter, dual_view,
            display_mode, scan_objective, fluo_objective, user_directory.
        """
        self.shape = tk.StringVar(value=self.loaded_params.get("shape", "Square"))
        self.shape.trace_add("write", lambda *args: self.resize_scan_content_area())
        self.shape.trace_add("write", lambda *args: self.save_parameters())

        self.exposure_time = tk.StringVar(value=self.loaded_params.get("exposure_time", 100))
        
        self.binning = tk.StringVar(value=self.loaded_params.get("binning", "2x2"))
        
        """self.shutter = tk.BooleanVar(value=self.loaded_params.get("shutter", False))
        self.shutter.trace_add("write", lambda *args: self.save_parameters())"""
        
        self.dual_view = tk.BooleanVar(value=self.loaded_params.get("dual_view", False))
        self.dual_view.trace_add("write", lambda *args: self.save_parameters())
        
        self.display_mode = tk.StringVar(value=self.loaded_params.get("display_mode", 'Grayscale'))
        self.display_mode.trace_add("write", lambda *args: self.save_parameters())
        
        self.scan_objective = tk.StringVar(value=self.loaded_params.get("scan_objective", '4x'))
        self.scan_objective.trace_add("write", lambda *args: self.save_parameters())
        
        self.fluo_objective = tk.StringVar(value=self.loaded_params.get("fluo_objective", '10x'))
        self.fluo_objective.trace_add("write", lambda *args: self.save_parameters())
        
        self.user_directory = tk.StringVar(value=self.loaded_params.get("user_directory", 'Arthur_2025_07_24'))
        self.user_directory.trace_add("write", lambda *args: self.save_parameters())
        
        self.name_model = tk.StringVar(value=self.loaded_params.get("name_model"))
        self.name_model.trace_add("write", lambda *args: self.save_parameters())
        
        self.fov_size_um = tk.StringVar(value=self.loaded_params.get("fov_size_um", 1000))
        self.fov_size_um.trace_add("write", lambda *args: self.save_parameters())
        
        # machine parameters
        self.machine_has_dual_view = tk.BooleanVar(value=self.loaded_params.get("machine_has_dual_view", True))
        self.machine_has_dual_view.trace_add("write", lambda *args: self.save_parameters())

        self.scan_width = tk.StringVar(value=self.loaded_params.get("scan_width", '8'))
        self.scan_width.trace_add("write", lambda *args: self.save_parameters())

        self.scan_height = tk.StringVar(value=self.loaded_params.get("scan_height", '8'))
        self.scan_height.trace_add("write", lambda *args: self.save_parameters())
        
        self.scan_height_length = tk.StringVar(value=self.loaded_params.get("scan_height_length"))
        self.scan_height_length.trace_add("write", lambda *args: self.save_parameters())
        
        self.scan_width_length = tk.StringVar(value=self.loaded_params.get("scan_width_length"))
        self.scan_width_length.trace_add("write", lambda *args: self.save_parameters())
        
        self.microscope_step_size = tk.StringVar(value=self.loaded_params.get("microscope_step_size"))
        self.microscope_step_size.trace_add("write", lambda *args: self.save_parameters())
        
        if MICROSCOPE == "Macrozoom":
            self.microscope_objective_size_1 = tk.StringVar(value="2")
            self.microscope_objective_size_2 = tk.StringVar(value="3")
            self.microscope_objective_size_3 = tk.StringVar(value="4")
            self.microscope_objective_size_4 = tk.StringVar(value="5")
            self.microscope_objective_size_5 = tk.StringVar(value="6")
            self.microscope_objective_size_6 = tk.StringVar(value="7")
        else:
            self.microscope_objective_size_1 = tk.StringVar(value=self.loaded_params.get("microscope_objective_size_1"))
            self.microscope_objective_size_1.trace_add("write", lambda *args: self.save_parameters())
            
            self.microscope_objective_size_2 = tk.StringVar(value=self.loaded_params.get("microscope_objective_size_2"))
            self.microscope_objective_size_2.trace_add("write", lambda *args: self.save_parameters())
            
            self.microscope_objective_size_3 = tk.StringVar(value=self.loaded_params.get("microscope_objective_size_3"))
            self.microscope_objective_size_3.trace_add("write", lambda *args: self.save_parameters())
            
            self.microscope_objective_size_4 = tk.StringVar(value=self.loaded_params.get("microscope_objective_size_4"))
            self.microscope_objective_size_4.trace_add("write", lambda *args: self.save_parameters())
            
            self.microscope_objective_size_5 = tk.StringVar(value=self.loaded_params.get("microscope_objective_size_5"))
            self.microscope_objective_size_5.trace_add("write", lambda *args: self.save_parameters())
            
            self.microscope_objective_size_6 = tk.StringVar(value=self.loaded_params.get("microscope_objective_size_6"))
            self.microscope_objective_size_6.trace_add("write", lambda *args: self.save_parameters())

        self.exposure_time.trace_add("write", self.update_exposure_and_save)
        self.binning.trace_add("write", self.update_binning_and_save)
                  
    def update_exposure_and_save(self, *args):
        self.save_parameters()
        try:
            self.CORE.setExposure(int(self.exposure_time.get()))
        except Exception:
            pass

    def update_binning_and_save(self, *args):
        self.save_parameters()
        try:
            self.CORE.setProperty(NAME_CAMERA, "Binning", self.binning.get()) 
        except Exception:
            pass

    def save_parameters(self):
        """
        Updates the parameters in the YAML file
        with current application parameters.
        """
        # New parameters to update
        params = {
            "exposure_time": self.exposure_time.get(),
            "binning": self.binning.get(),
            "dual_view": self.dual_view.get(),
            "display_mode": self.display_mode.get(),
            "scan_objective": self.scan_objective.get(),
            "fluo_objective": self.fluo_objective.get(),
            "shape": self.shape.get(),
            "user_directory": self.user_directory.get(),
            "name_model": self.name_model.get(),
            "machine_has_dual_view": self.machine_has_dual_view.get(),
            "scan_width": self.scan_width.get(),
            "scan_height": self.scan_height.get(),
            "scan_height_length": self.scan_height_length.get(),
            "scan_width_length": self.scan_width_length.get(),
            "microscope_objective_size_1": self.microscope_objective_size_1.get(),
            "microscope_objective_size_2": self.microscope_objective_size_2.get(),
            "microscope_objective_size_3": self.microscope_objective_size_3.get(),
            "microscope_objective_size_4": self.microscope_objective_size_4.get(),
            "microscope_objective_size_5": self.microscope_objective_size_5.get(),
            "microscope_objective_size_6": self.microscope_objective_size_6.get(),
            "fov_size_um": self.fov_size_um.get()
        }
        
        # Read and parse the existing YAML file
        try:
            with open(self.PARAMS_FILE, "r") as f:
                config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            config = {}

        # Update only the corner position parameters
        config.update(params)

        # Write the updated configuration back to file
        with open(self.PARAMS_FILE, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
    def set_binning(self):
        """
        Set the binning parameter
        """
        self.CORE.setProperty(NAME_CAMERA, "Binning", str(self.loaded_params.get("binning"))) 

    def update_colors(self):
        """
        Updates the application's color theme based on the current dark mode setting.

        Creates a new instance of `ColorTheme` and stores it in `self.colors`.
        This object holds all color values used throughout the UI.
        """
        self.colors = ColorTheme(self.dark_mode)     
    
    def set_color_theme(self):
        """
        Configures the visual appearance of custom-themed ttk comboboxes.

        This method:
            - Applies a theme ("clam") to ttk styles.
            - Maps widget state styles (disabled, readonly) for foreground, background, borders, etc.
            - Configures fonts and layout for dropdown menus.
            - Repositions the combobox arrow to the left side for a custom look.

        Affects the appearance of all comboboxes using the 'MyCombobox.TCombobox' style.
        """
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.map('MyCombobox.TCombobox',
                        fieldbackground=[
                            ('disabled', self.colors.theme["parameters_button_disabled_background"]),
                            ('readonly', self.colors.theme["parameters_button_background"])
                        ],
                        selectbackground=[
                            ('disabled', self.colors.theme["parameters_button_disabled_background"]),
                            ('readonly', self.colors.theme["parameters_button_background"])
                        ],
                        selectforeground=[
                            ('disabled', self.colors.theme["tertiary_text"]),
                            ('readonly', self.colors.theme["tertiary_text"])
                        ],
                        foreground=[
                            ('disabled', self.colors.theme["tertiary_text"]),
                            ('readonly', self.colors.theme["tertiary_text"])
                        ],
                        background=[
                            ('disabled', self.colors.theme["parameters_button_disabled_background"]),
                            ('readonly', self.colors.theme["parameters_button_background"])
                        ],
                        arrowcolor=[
                            ('disabled', self.colors.theme["tertiary_text"]),
                            ('readonly', self.colors.theme["tertiary_text"])
                        ],
                        bordercolor=[
                            ('disabled', self.colors.theme["parameters_button_disabled_background"]),
                            ('readonly', self.colors.theme["parameters_button_background"])
                        ],
                        darkcolor=[
                            ('disabled', self.colors.theme["parameters_button_disabled_background"]),
                            ('readonly', self.colors.theme["parameters_button_background"])
                        ],
                        lightcolor=[
                            ('disabled', self.colors.theme["parameters_button_disabled_background"]),
                            ('readonly', self.colors.theme["parameters_button_background"])
                        ]
                    )

        self.style.configure('TCombobox.Popdown',
                             background=self.colors.theme["parameters_button_background"],
                             foreground=self.colors.theme["tertiary_text"],
                             selectbackground=self.colors.theme["parameters_button_background"],
                             selectforeground=self.colors.theme["tertiary_text"]
                             )

        self.style.configure('TCombobox.Popdown.Listbox',
                             font=(self.font, 10), 
                             background=self.colors.theme["parameters_button_background"],
                             foreground=self.colors.theme["tertiary_text"],
                             selectbackground=self.colors.theme["parameters_button_background"],
                             selectforeground=self.colors.theme["tertiary_text"]
                             )
        self.style.configure('TCombobox.downarrow',
                             foreground=self.colors.theme["secondary_background"], # Color of the arrow itself
                             background=self.colors.theme["parameters_button_background"], # Background behind the arrow
                             arrowsize=25,
                             relief="flat"
                             )
        self.style.configure('TCombobox.button',
                     background=self.colors.theme["parameters_button_background"],
                     bordercolor=self.colors.theme["parameters_button_background"], 
                     relief="flat", 
                     lightcolor=self.colors.theme["parameters_button_background"],
                     darkcolor=self.colors.theme["parameters_button_background"],
                     padding=[10, 0, 10, 0] 
                     )
        
        combobox_layout = [
            ('Combobox.downarrow', {'side': 'left', 'sticky': 'ns'}),
            ('Combobox.field', {'sticky': 'nswe', 'children': [
                ('Combobox.padding', {'sticky': 'nswe', 'children': [
                    ('Combobox.textarea', {'sticky': 'nswe', 'expand': 1}) # expand=1 ensures it takes remaining space
                ]})
            ]})
        ]
        self.style.layout('MyCombobox.TCombobox', combobox_layout)
    
    def load_icon(self):   
        """
        Loads and processes all application icons from disk.

        This includes icons for:
            - Parameter toggles (e.g. open/close, filters, clock).
            - Navigation menu (e.g. scan, validation, load, modify, quit).
            - Main content actions (e.g. play, info, plus, live, snap).
            - Worm classification (wildtype, mutant).
            - Worm management (add/remove).

        Each icon is recolored and resized using `flatten_and_resize_icon()`
        to match the current theme (background/foreground colors).
        """
        # ---------------- Parameters icon ----------------     
        # Process toggle_open.png
        open_img_path = Path(RESSOURCES_DIR) / "icon" / "toggle_open.png"
        self.toggle_open_icon = self.flatten_and_resize_icon(open_img_path, 34, 14, self.colors.theme["secondary_background"], self.colors.theme["toggle_button"])

        # Process toggle_close.png
        close_img_path = Path(RESSOURCES_DIR) / "icon" / "toggle_close.png"
        self.toggle_close_icon = self.flatten_and_resize_icon(close_img_path, 34, 14, self.colors.theme["secondary_background"], self.colors.theme["toggle_button"])
        
        # Process filtre.png
        filtre_path = Path(RESSOURCES_DIR) / "icon" / "filtre.png" 
        self.icon_parameter = self.flatten_and_resize_icon(filtre_path, 18, 18, self.colors.theme["secondary_background"], self.colors.theme["icon"])
        
        # Process clock.png
        clock_path = Path(RESSOURCES_DIR) / "icon" / "clock.png" 
        self.clock_icon = self.flatten_and_resize_icon(clock_path, 18, 18, self.colors.theme["parameters_button_background"], self.colors.theme["tertiary_text"])
        self.clock_icon_disabled = self.flatten_and_resize_icon(clock_path, 18, 18, self.colors.theme["parameters_button_disabled_background"], self.colors.theme["tertiary_text"])
        
        # ---------------- Menu icon ----------------
        # Process scan.png
        """scan_path = Path(RESSOURCES_DIR) / "icon" / "scan.png" 
        self.scan_icon = self.flatten_and_resize_icon(scan_path, 18, 18, self.colors.theme["primary_background"], self.colors.theme["icon"])
        self.scan_icon_hover = self.flatten_and_resize_icon(scan_path, 18, 18, self.colors.theme["secondary_background"], self.colors.theme["icon"])
        """
        # Process loupe.png
        scan_path = Path(RESSOURCES_DIR) / "icon" / "loupe.png" 
        self.scan_icon = self.flatten_and_resize_icon(scan_path, 18, 18, self.colors.theme["primary_background"], self.colors.theme["icon"])
        self.scan_icon_hover = self.flatten_and_resize_icon(scan_path, 18, 18, self.colors.theme["secondary_background"], self.colors.theme["icon"])
        
        # Process validation.png
        validation_path = Path(RESSOURCES_DIR) / "icon" / "validation.png" 
        self.validation_icon = self.flatten_and_resize_icon(validation_path, 18, 18, self.colors.theme["primary_background"], self.colors.theme["icon"])
        self.validation_icon_hover = self.flatten_and_resize_icon(validation_path, 18, 18, self.colors.theme["secondary_background"], self.colors.theme["icon"])
        
        # Process load.png
        """load_path = Path(RESSOURCES_DIR) / "icon" / "load.png" 
        self.loading_icon = self.flatten_and_resize_icon(load_path, 18, 18, self.colors.theme["primary_background"], self.colors.theme["icon"])
        self.loading_icon_hover = self.flatten_and_resize_icon(load_path, 18, 18, self.colors.theme["secondary_background"], self.colors.theme["icon"])
        """
        
        # Process analyse.png
        load_path = Path(RESSOURCES_DIR) / "icon" / "analyse.png" 
        self.loading_icon = self.flatten_and_resize_icon(load_path, 18, 18, self.colors.theme["primary_background"], self.colors.theme["icon"])
        self.loading_icon_hover = self.flatten_and_resize_icon(load_path, 18, 18, self.colors.theme["secondary_background"], self.colors.theme["icon"])
        
        # Process machine_parameters.png
        machine_parameters_path = Path(RESSOURCES_DIR) / "icon" / "machine_parameters.png" 
        self.machine_parameters_icon = self.flatten_and_resize_icon(machine_parameters_path, 18, 18, self.colors.theme["primary_background"], self.colors.theme["icon"])
        self.machine_parameters_icon_hover = self.flatten_and_resize_icon(machine_parameters_path, 18, 18, self.colors.theme["secondary_background"], self.colors.theme["icon"])
        
        # Process modify.png
        modify_path = Path(RESSOURCES_DIR) / "icon" / "modify.png" 
        self.modify_icon = self.flatten_and_resize_icon(modify_path, 18, 18, self.colors.theme["primary_background"], self.colors.theme["icon"])
        self.modify_icon_hover = self.flatten_and_resize_icon(modify_path, 18, 18, self.colors.theme["secondary_background"], self.colors.theme["icon"])
        
        # Process page.png
        page_path = Path(RESSOURCES_DIR) / "icon" / "page.png" 
        self.page_icon = self.flatten_and_resize_icon(page_path, 18, 18, self.colors.theme["primary_background"], self.colors.theme["icon"])
        self.page_icon_hover = self.flatten_and_resize_icon(page_path, 18, 18, self.colors.theme["secondary_background"], self.colors.theme["icon"])
        
        # Process question.png
        question_path = Path(RESSOURCES_DIR) / "icon" / "question.png" 
        self.question_icon = self.flatten_and_resize_icon(question_path, 18, 18, self.colors.theme["primary_background"], self.colors.theme["icon"])
        self.question_icon_hover = self.flatten_and_resize_icon(question_path, 18, 18, self.colors.theme["secondary_background"], self.colors.theme["icon"])
        
        # Process training_model_icon.png
        training_model_path = Path(RESSOURCES_DIR) / "icon" / "training_model.png" 
        self.training_model_icon = self.flatten_and_resize_icon(training_model_path, 18, 18, self.colors.theme["primary_background"], self.colors.theme["icon"])
        self.training_model_icon_hover = self.flatten_and_resize_icon(training_model_path, 18, 18, self.colors.theme["secondary_background"], self.colors.theme["icon"])
        
        # Process quit.png
        quit_path = Path(RESSOURCES_DIR) / "icon" / "quit.png" 
        self.quit_icon = self.flatten_and_resize_icon(quit_path, 18, 18, self.colors.theme["quit_button_background"], self.colors.theme["quit_button_text"])
        self.quit_icon_hover = self.flatten_and_resize_icon(quit_path, 18, 18, self.colors.theme["quit_button_background_hover"], self.colors.theme["quit_button_text"])
        
        # ---------------- Main content icon ----------------
        # Process play.png
        play_path = Path(RESSOURCES_DIR) / "icon" / "play.png" 
        self.play_icon = self.flatten_and_resize_icon(play_path, 40, 40, self.colors.theme["primary_background"], self.colors.theme["stroke_button"])
        self.play_icon_hover = self.flatten_and_resize_icon(play_path, 40, 40, self.colors.theme["secondary_background"], self.colors.theme["stroke_button"])
        
        # Process info.png
        info_path = Path(RESSOURCES_DIR) / "icon" / "info.png" 
        self.info_icon = self.flatten_and_resize_icon(info_path, 16, 16, self.colors.theme["primary_background"], self.colors.theme["secondary_text"])
        self.info_icon_secondary = self.flatten_and_resize_icon(info_path, 16, 16, self.colors.theme["secondary_background"], self.colors.theme["secondary_text"])
           
        # Process plus.png
        plus_path = Path(RESSOURCES_DIR) / "icon" / "plus.png" 
        self.plus_icon = self.flatten_and_resize_icon(plus_path, 60, 60, self.colors.theme["secondary_background"], self.colors.theme["stroke_button"])
        self.plus_icon_hover = self.flatten_and_resize_icon(plus_path, 60, 60, self.colors.theme["tertiary_background"], self.colors.theme["stroke_button"])
        
        # Process live.png
        live_path = Path(RESSOURCES_DIR) / "icon" / "live.png" 
        self.live_icon = self.flatten_and_resize_icon(live_path, 40, 40, self.colors.theme["primary_background"], self.colors.theme["stroke_button"])
        self.live_icon_hover = self.flatten_and_resize_icon(live_path, 40, 40, self.colors.theme["tertiary_background"], self.colors.theme["stroke_button"])
        
        # Process snap.png
        snap_path = Path(RESSOURCES_DIR) / "icon" / "snap.png" 
        self.snap_icon = self.flatten_and_resize_icon(snap_path, 40, 40, self.colors.theme["primary_background"], self.colors.theme["stroke_button"])
        self.snap_icon_hover = self.flatten_and_resize_icon(snap_path, 40, 40, self.colors.theme["tertiary_background"], self.colors.theme["stroke_button"])
        
        # Process wildtype.png
        wildtype_path = Path(RESSOURCES_DIR) / "icon" / "wildtype.png" 
        self.wildtype_icon = self.flatten_and_resize_icon(wildtype_path, 40, 40, self.colors.theme["primary_background"], self.colors.theme["stroke_button"])
        self.wildtype_icon_hover = self.flatten_and_resize_icon(wildtype_path, 40, 40, self.colors.theme["secondary_background"], self.colors.theme["stroke_button"])
        
        # Process mutant.png
        mutant_path = Path(RESSOURCES_DIR) / "icon" / "mutant.png" 
        self.mutant_icon = self.flatten_and_resize_icon(mutant_path, 50, 50, self.colors.theme["primary_background"], self.colors.theme["stroke_button"])
        self.mutant_icon_hover = self.flatten_and_resize_icon(mutant_path, 50, 50, self.colors.theme["secondary_background"], self.colors.theme["stroke_button"])
        
        # Process next.png
        next_path = Path(RESSOURCES_DIR) / "icon" / "next.png" 
        self.next_icon = self.flatten_and_resize_icon(next_path, 36, 36, self.colors.theme["primary_background"], self.colors.theme["stroke_button"])
        self.next_icon_hover = self.flatten_and_resize_icon(next_path, 36, 36, self.colors.theme["secondary_background"], self.colors.theme["stroke_button"])
        
        # Process last.png
        last_path = Path(RESSOURCES_DIR) / "icon" / "last.png" 
        self.last_icon = self.flatten_and_resize_icon(last_path, 36, 36, self.colors.theme["primary_background"], self.colors.theme["stroke_button"])
        self.last_icon_hover = self.flatten_and_resize_icon(last_path, 36, 36, self.colors.theme["secondary_background"], self.colors.theme["stroke_button"])
        
        # Process add_worm.png
        add_worm_path = Path(RESSOURCES_DIR) / "icon" / "add_worm.png" 
        self.add_worm_icon = self.flatten_and_resize_icon(add_worm_path, 20, 20, self.colors.theme["primary_background"], self.colors.theme["stroke_button"])
        self.add_worm_icon_hover = self.flatten_and_resize_icon(add_worm_path, 20, 20, self.colors.theme["tertiary_background"], self.colors.theme["stroke_button"])
        
        # Process move_worm.png
        move_worm_path = Path(RESSOURCES_DIR) / "icon" / "move.png" 
        self.move_worm_icon = self.flatten_and_resize_icon(move_worm_path, 20, 20, self.colors.theme["primary_background"], self.colors.theme["stroke_button"])
        self.move_worm_icon_hover = self.flatten_and_resize_icon(move_worm_path, 20, 20, self.colors.theme["tertiary_background"], self.colors.theme["stroke_button"])
        
        # Process remove_worm.png
        remove_worm_path = Path(RESSOURCES_DIR) / "icon" / "remove_worm.png" 
        self.remove_worm_icon = self.flatten_and_resize_icon(remove_worm_path, 20, 20, self.colors.theme["primary_background"], self.colors.theme["stroke_button"])
        self.remove_worm_icon_hover = self.flatten_and_resize_icon(remove_worm_path, 20, 20, self.colors.theme["tertiary_background"], self.colors.theme["stroke_button"])
               
        # Machine config icon
        # Process toggle_open.png - big size
        big_open_toggle_path = Path(RESSOURCES_DIR) / "icon" / "toggle_open.png"
        self.big_toggle_open_icon = self.flatten_and_resize_icon(big_open_toggle_path, 68, 24, self.colors.theme["primary_background"], self.colors.theme["toggle_button"])

        # Process toggle_close.png - big size
        big_close_toggle_path = Path(RESSOURCES_DIR) / "icon" / "toggle_close.png"
        self.big_toggle_close_icon = self.flatten_and_resize_icon(big_close_toggle_path, 68, 24, self.colors.theme["primary_background"], self.colors.theme["toggle_button"])
                                                          
    def flatten_and_resize_pil(self, img_pil, width, height, bg_color, fg_color):
        """
        Helper that performs the actual PIL operations to resize and recolor an image.
        
        Args:
            img_pil (PIL.Image): The source PIL image (should be RGBA).
            width (int): Target width.
            height (int): Target height.
            bg_color (str): Background color.
            fg_color (str): Foreground color.
            
        Returns:
            PIL.Image: The processed PIL image (RGB).
        """
        # Resize while preserving aspect ratio
        img_pil_resized = img_pil.copy()
        img_pil_resized.thumbnail((width, height), Image.LANCZOS)

        # Separate alpha channel
        if img_pil_resized.mode != 'RGBA':
            img_pil_resized = img_pil_resized.convert('RGBA')
            
        r, g, b, alpha = img_pil_resized.split()

        # Create a new solid image with the desired foreground color (primary_text)
        fg_rgb = ImageColor.getrgb(fg_color)  # Converts "#FFFFFF" -> (255, 255, 255)
        color_image = Image.new("RGBA", img_pil_resized.size, fg_rgb + (255,))  

        # Apply the original alpha mask to the new color
        recolored_icon = Image.composite(color_image, Image.new("RGBA", img_pil_resized.size), alpha)

        # Create the full-size background image
        background = Image.new("RGB", (width, height), bg_color)

        # Compute offset to center the icon
        offset_x = (width - recolored_icon.width) // 2
        offset_y = (height - recolored_icon.height) // 2

        # Paste the recolored icon using alpha as mask
        background.paste(recolored_icon, (offset_x, offset_y), alpha)
        
        return background

    def flatten_and_resize_icon(self, img_path, width, height, bg_color, fg_color):
        """
        Loads, recolors, and resizes an image icon.

        Args:
            img_path (Path or str): Path to the image file to process.
            width (int): Target width of the final icon.
            height (int): Target height of the final icon.
            bg_color (str): Background color (hex or named color) to use behind the icon.
            fg_color (str): Foreground color used to recolor the icon.

        Returns:
            ImageTk.PhotoImage: A Tkinter-compatible image object for UI usage.

        Notes:
            - Preserves original aspect ratio when resizing.
            - Applies alpha mask to recolor only the visible icon shape.
            - Centers the recolored icon within the desired dimensions.
        """
        img_pil = Image.open(str(img_path)).convert("RGBA")
        background = self.flatten_and_resize_pil(img_pil, width, height, bg_color, fg_color)
        return ImageTk.PhotoImage(background)

    # --- Create global interface ---
    def create_layout(self):
        """
        Creates the main layout of the application window.

        This method sets up the visual structure of the GUI by organizing:
        - The top bar (title and controls)
        - A horizontal frame that holds:
            - A sidebar containing the menu (left)
            - A parameters panel (right)
            - A central content area
        It ensures proper packing order to achieve the desired spatial distribution.
        """
        # Top bar with title and controls - full width
        self.create_top_bar()

        # Below top bar: main horizontal container (sidebar + content + parameters)
        self.body_frame = tk.Frame(self.main_frame, bg=self.colors.theme["primary_background"])
        self.body_frame.pack(fill=tk.BOTH, expand=True)

        # Sidebar (LEFT)
        self.create_sidebar()

        # Parameters (RIGHT)
        if self.CORE is not None:
            self.create_parameters_panel()

        # Content frame (CENTER) - fills the remaining space
        self.content_frame = tk.Frame(self.body_frame, bg=self.colors.theme["primary_background"])
        self.content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Main content inside content frame
        self.main_content = tk.Frame(self.content_frame, bg=self.colors.theme["primary_background"])
        self.main_content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def create_top_bar(self):
        """
        Creates the top bar containing the application title and control buttons.

        Adds a title label, a dark mode toggle button, and a parameters panel toggle button.
        Also includes a subtle border line under the top bar for separation.
        """
        # Create top bar
        top_frame = tk.Frame(self.main_frame, bg=self.colors.theme["primary_background"], height=64)
        top_frame.pack(fill=tk.X)
        top_frame.pack_propagate(False)
        
        # Add a border
        border_frame = tk.Frame(self.main_frame,bg=self.colors.theme["stroke"],height=1, relief=tk.RIDGE)
        border_frame.pack(fill=tk.X)
        border_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(top_frame, text="Worm Analysis App", bg=self.colors.theme["primary_background"], fg=self.colors.theme["primary_text"],
                              font=(self.font, 13, 'bold'))
        title_label.pack(side=tk.LEFT, padx=80)
        
        # Create a frame that will contain the 2 buttons
        controls_frame = tk.Frame(top_frame, bg=self.colors.theme["primary_background"]) 
        controls_frame.pack(side=tk.RIGHT, padx=30)
        
        # Add the Dark mode button (still a standard Tkinter button)
        text_dark_mode_button = "Dark mode" if not self.dark_mode else "Light mode"
        dark_btn = self.create_rounded_button(
            parent=controls_frame,
            text=text_dark_mode_button,
            command=self.toggle_dark_mode,
            bg_color=self.colors.theme["dark_mode_button_background"],
            text_color=self.colors.theme["dark_mode_button_text"],
            hover_color=self.colors.theme["dark_mode_button_background_hover"],
            font=(self.font, 11),
            width_pixels=100, # Define width in pixels 
            height_pixels=40, # Define height in pixels 
            corner_radius=20, # Define the radius of the rounded corners
            side=tk.RIGHT,
            padx=5
        )
        dark_btn.pack(side=tk.RIGHT)

        self.create_rounded_button(
            parent=controls_frame,
            text="...",
            command=self.toggle_parameters,
            bg_color=self.colors.theme["secondary_background"],
            text_color=self.colors.theme["primary_text"],
            hover_color=self.colors.theme["tertiary_background"],
            font=(self.font, 16),
            width_pixels=50, # Define width in pixels
            height_pixels=40, # Define height in pixels
            corner_radius=20, # Define the radius of the rounded corners
            side=tk.RIGHT,
            padx=5,
            pady_text=6
        )
    
    def create_sidebar(self):
        """
        Creates the sidebar on the left side of the layout.

        The sidebar includes:
        - A title label ("Menu")
        - Menu sections with navigation buttons (e.g., Detection, Analysis, Help)
        - A Quit button at the bottom

        Each button uses icons and hover effects for enhanced UI experience.
        """
        # Create the side bar
        self.sidebar = tk.Frame(self.body_frame, bg=self.colors.theme["primary_background"], width=230)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)
        
        # Add a border
        border_sidebar = tk.Frame(self.body_frame, bg=self.colors.theme["stroke"], width=1)
        border_sidebar.pack(side=tk.LEFT, fill=tk.Y)
        border_sidebar.pack_propagate(False)
        
        # Menu title
        title_label = tk.Label(self.sidebar, text="Menu", bg=self.colors.theme["primary_background"], fg=self.colors.theme["primary_text"],
                              font=(self.font, 14, 'bold'), anchor='w')
        title_label.pack(fill=tk.X, padx=15, pady=(20, 5))
        
        # Menu sections
        self.create_menu_section("Detection", [
            ("Automatic Scan", "automatic_scan", self.scan_icon, self.scan_icon_hover),
            ("Scan result", "scan_result", self.validation_icon, self.validation_icon_hover)
        ])
        
        self.create_menu_section("Analysis", [
            ("Analyse worms", "load_position", self.loading_icon, self.loading_icon_hover),
            ("Length analysis", "length_analysis", self.loading_icon, self.loading_icon_hover),
            ("Training model", "training", self.training_model_icon, self.training_model_icon_hover)
        ])
        
        self.create_menu_section("Help", [
            ("Documentation", "documentation", self.page_icon, self.page_icon_hover),
            ("Machine Config", "configuration", self.machine_parameters_icon, self.machine_parameters_icon_hover)
        ])
        
        # Quit button at bottom       
        self.create_rounded_button(
            parent=self.sidebar,
            text="Quit",
            icon=self.quit_icon,
            icon_hover=self.quit_icon_hover,
            command=lambda: self.end_of_program(),
            bg_color=self.colors.theme["quit_button_background"],
            text_color=self.colors.theme["quit_button_text"],
            hover_color=self.colors.theme["quit_button_background_hover"],
            font=(self.font, 12),
            width_pixels=211, # Define width in pixels
            height_pixels=40, # Define height in pixels
            corner_radius=20, # Define the radius of the rounded corners
            side=tk.BOTTOM,
            pady=25
        )
        
    def create_menu_section(self, title, items):
        """
        Creates a collapsible menu section in the sidebar.

        Args:
            title (str): The section title (e.g., "Detection", "Help").
            items (list of tuple): Each item is a tuple of:
                (str) button label,
                (str) page ID,
                (ImageTk.PhotoImage) icon,
                (ImageTk.PhotoImage) hover icon.

        Buttons created in the section allow switching pages via `self.switch_page`.
        """        
        # Section title
        title_label = tk.Label(self.sidebar, text=title, bg=self.colors.theme["primary_background"], fg=self.colors.theme["primary_text"],
                              font=(self.font, 11, 'bold'), anchor='w')
        title_label.pack(fill=tk.X, padx=15, pady=(20, 5))
        
        # Menu items
        for text, page_id, icon, icon_hover in items:
            # Add button
            bg_color = self.colors.theme["secondary_background"] if page_id == self.current_page else self.colors.theme["primary_background"]
            if page_id == self.current_page: icon = icon_hover
            self.create_rounded_button(
                parent=self.sidebar,
                text=text,
                icon=icon,
                icon_hover=icon_hover,
                command=lambda p=page_id: self.switch_page(p),
                bg_color=bg_color,
                text_color=self.colors.theme["primary_text"],
                hover_color=self.colors.theme["secondary_background"],
                font=(self.font, 12),
                width_pixels=211,
                height_pixels=40,
                corner_radius=20,
                side=tk.TOP,
                anchor='w',
                padx_text=90,
                pady=2
            )
                         
    def create_parameters_panel(self):
        """
        Creates the parameters panel on the right side of the layout.

        This panel is shown only if `self.show_parameters` is True. It includes:
        - A header with the "Parameters" label and an icon
        - A scrollable content area populated by `create_parameters_content`
        - A "Name directory" input field at the bottom
        """
        # Create the frame
        self.params_frame = tk.Frame(self.body_frame, bg=self.colors.theme["secondary_background"], width=230)
        if self.show_parameters:
            if hasattr(self, "content_frame"):
                self.params_frame.pack(side=tk.RIGHT, fill=tk.Y, before=self.content_frame)
            else:
                self.params_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.params_frame.pack_propagate(False)

        # Parameters header
        header_frame = tk.Frame(self.params_frame, bg=self.colors.theme["secondary_background"])
        header_frame.pack(fill=tk.X, pady=(20, 10), padx=20)

        tk.Label(header_frame, text="Parameters", bg=self.colors.theme["secondary_background"],
                 fg=self.colors.theme["primary_text"], font=(self.font, 14, "bold")).pack(side=tk.LEFT)

        tk.Label(header_frame, image=self.icon_parameter, bg=self.colors.theme["secondary_background"]).pack(side=tk.RIGHT, padx=(0,50))

        # Content frame for parameters 
        self.params_content_frame = tk.Frame(self.params_frame, bg=self.colors.theme["secondary_background"])
        self.params_content_frame.pack(fill=tk.BOTH, expand=True, padx=20)

        # Populate the main content of the parameters panel
        self.create_parameters_content()

        # Name directory at the bottom, packed directly into params_frame
        name_dir_label = tk.Label(self.params_frame, text="Name directory", bg=self.colors.theme["secondary_background"],
                                  fg=self.colors.theme["secondary_text"], font=(self.font, 10))
        name_dir_label.pack(anchor='w', pady=(5, 5), padx=20)

        _, self.name_directory_entry = self.create_rounded_input(
            self.params_frame, self.user_directory, padx=20
        )

    def create_parameters_content(self):
        """
        Populates the parameters panel with configurable options.

        The options include:
        - Exposure time (ms)
        - Binning mode (e.g., 2x2, 3x3)
        - Toggles for shutter and dual view
        - Display mode selector
        - Scan and fluorescence objective choices
        - Scan shape selection

        Each parameter respects its enabled/disabled state and visual design.
        """
        # Scan Objective
        list_scan_objective = [
            str(self.loaded_params.get(f"microscope_objective_size_{i}")) + "x"
            for i in range(1, 7)
            if str(self.loaded_params.get(f"microscope_objective_size_{i}")) != ""
        ]
        bg = "parameters_button_background" if "scan_objective" in self.enable_parameters_buttons else "parameters_button_disabled_background"
        
        # Frame for label + info icon
        objective_label_frame = tk.Frame(self.params_content_frame, bg=self.colors.theme["secondary_background"])
        objective_label_frame.pack(anchor='w', pady=(5, 0), fill=tk.X)
        
        tk.Label(objective_label_frame, text="Scan objective", bg=self.colors.theme["secondary_background"], fg=self.colors.theme["secondary_text"], font=(self.font, 10)).pack(side=tk.LEFT)
        
        # Info icon with tooltip
        info_icon_label = tk.Label(objective_label_frame, image=self.info_icon_secondary, bg=self.colors.theme["secondary_background"], cursor="hand2")
        info_icon_label.pack(side=tk.LEFT, padx=(5, 0))
        
        if MICROSCOPE == "Macrozoom":
            Tooltip(info_icon_label, "Enter the zoom used on the wheel (from 1 to 8). 3 is recommended. Also, always use the 2x objective for the scan.", title="Scan Objective", theme="info", posy=20, posx=-200)
        elif MICROSCOPE == "Nikon": 
            Tooltip(info_icon_label, "Enter the objective used for the scan (x4 is recommended).", title="Scan Objective", theme="info", posy=20, posx=-200)

        _, self.scan_objective_dropdown = self.create_rounded_dropdown(
            self.params_content_frame, list_scan_objective, self.scan_objective, bg
        )
        
        # Scan shape
        bg = "parameters_button_background" if "scan_shape" in self.enable_parameters_buttons else "parameters_button_disabled_background"
        tk.Label(self.params_content_frame, text="Scan shape", bg=self.colors.theme["secondary_background"], fg=self.colors.theme["secondary_text"], font=(self.font, 10)).pack(anchor='w', pady=(5, 0))
        _, self.scan_shape_dropdown = self.create_rounded_dropdown(
            self.params_content_frame, ["Square", "Rectangle"], self.shape, bg
        )
        
        # Dual view
        if self.machine_has_dual_view.get():
            self.dual_view_toggle = self.create_custom_toggle(self.params_content_frame, "Dual view", self.dual_view)
        else:
            self.dual_view_toggle = None
        
        # Exposure time
        bg = "parameters_button_background" if "exposure_time" in self.enable_parameters_buttons else "parameters_button_disabled_background"
        icon = self.clock_icon if "exposure_time" in self.enable_parameters_buttons else self.clock_icon_disabled
        tk.Label(self.params_content_frame, text="Exposure time (ms)", bg=self.colors.theme["secondary_background"], fg=self.colors.theme["secondary_text"], font=(self.font, 10)).pack(anchor='w', pady=(5, 0))
        _, self.exposure_time_entry = self.create_rounded_input_with_icon(
            self.params_content_frame, self.exposure_time, icon, bg
        )
        
        # Binning
        list_binning = list(self.CORE.getAllowedPropertyValues("Camera-1","Binning"))
        bg = "parameters_button_background" if "binning" in self.enable_parameters_buttons else "parameters_button_disabled_background"
        tk.Label(self.params_content_frame, text="Binning", bg=self.colors.theme["secondary_background"], fg=self.colors.theme["secondary_text"], font=(self.font, 10)).pack(anchor='w', pady=(5, 0))
        _, self.binning_dropdown = self.create_rounded_dropdown(
            self.params_content_frame, list_binning, self.binning, bg
        )

        # Shutter toggle
        #self.shutter_toggle = self.create_custom_toggle(self.params_content_frame, "Shutter", self.shutter)

        # Display mode
        """bg = "parameters_button_background" if "display_mode" in self.enable_parameters_buttons else "parameters_button_disabled_background"
        tk.Label(self.params_content_frame, text="Display mode", bg=self.colors.theme["secondary_background"], fg=self.colors.theme["secondary_text"], font=(self.font, 10)).pack(anchor='w', pady=(5, 0))
        _, self.display_mode_dropdown = self.create_rounded_dropdown(
            self.params_content_frame, ["Grayscale"], self.display_mode, bg
        )"""

        # Fluo objective
        bg = "parameters_button_background" if "fluo_objective" in self.enable_parameters_buttons else "parameters_button_disabled_background"
        tk.Label(self.params_content_frame, text="Live objective", bg=self.colors.theme["secondary_background"], fg=self.colors.theme["secondary_text"], font=(self.font, 10)).pack(anchor='w', pady=(5, 0))
        _, self.fluo_objective_dropdown = self.create_rounded_dropdown(
            self.params_content_frame, list_scan_objective, self.fluo_objective, bg
        )
        
        # Model name
        list_model = [d.name.replace("model_prediction_", "").replace(".pkl", "") for d in MODELS_DIR.iterdir() if "model_prediction_" in d.name and not d.name.startswith("._")]
        bg = "parameters_button_background" if "model_name" in self.enable_parameters_buttons else "parameters_button_disabled_background"
        tk.Label(self.params_content_frame, text="Model name", bg=self.colors.theme["secondary_background"], fg=self.colors.theme["secondary_text"], font=(self.font, 10)).pack(anchor='w', pady=(5, 0))
        _, self.model_name_dropdown = self.create_rounded_dropdown(
            self.params_content_frame, list_model, self.name_model, bg
        )
        
        # FOV Size 
        bg = "parameters_button_background" if "fov_size_um" in self.enable_parameters_buttons else "parameters_button_disabled_background" # enabled by default if not in enable_parameters_buttons check logic, or needs adding to that list
        tk.Label(self.params_content_frame, text="Microscope step size", bg=self.colors.theme["secondary_background"], fg=self.colors.theme["secondary_text"], font=(self.font, 10)).pack(anchor='w', pady=(5, 0))
        _, self.fov_size_um_entry = self.create_rounded_input(
            self.params_content_frame, self.fov_size_um, bg
        )
    
    # --- Button ---
    def create_rounded_input(self, parent, variable, bg = "parameters_button_background", width = 190, height = 35, padx=0): 
        """
        Creates a rounded entry input widget inside a canvas.

        This method draws a custom-styled input field with rounded corners and
        embeds a Tkinter `Entry` widget bound to a StringVar.

        Args:
            parent (tk.Widget): The parent widget where the input is placed.
            variable (str | tk.StringVar | None): The initial value or variable to bind. 
                If a string is passed, it will be wrapped in a StringVar.
            bg (str, optional): Key name for the background color in the theme. 
                Defaults to "parameters_button_background".

        Returns:
            tk.StringVar: The variable bound to the input field.
        """
        # Change the variable into a StringVar if needed
        if isinstance(variable, str):
            variable = tk.StringVar(value=variable)
        elif variable is None:
            variable = tk.StringVar() 
            
        # Dimensions of the canvas
        canvas_width = width 
        canvas_height = height
        radius = 20 

        # Create the canvas
        canvas = tk.Canvas(parent, width=canvas_width, height=canvas_height,
                           bg=parent.cget("bg"), highlightthickness=0, takefocus=0) # Use parent's bg for canvas
        canvas.pack(fill=tk.X, pady=(0, 15), padx=padx)

        # Draw the rounded background
        self.draw_rounded_rect(canvas, 0, 0, canvas_width, canvas_height,
                               radius, fill=self.colors.theme[bg],
                               outline=self.colors.theme[bg], tag="input_bg")

        # Create the Entry widget        
        entry = tk.Entry(canvas, textvariable=variable, font=(self.font, 10), bd=0, relief="flat", highlightthickness=0,
                 bg=self.colors.theme[bg], fg=self.colors.theme["tertiary_text"],
                 insertbackground=self.colors.theme["primary_text"],
                 disabledbackground=self.colors.theme[bg],
                 disabledforeground=self.colors.theme["tertiary_text"])

        # Place the entry widget inside the canvas. Adjust x, y for padding.
        entry_width = canvas_width - 2 * radius # Approximate width of the entry part
        entry_height = canvas_height - 10 # Approximate height of the entry part
        canvas.create_window(radius, canvas_height / 2, window=entry, anchor="w",
                             width=entry_width, height=entry_height)
        canvas.variable = variable

        return variable, entry
    
    def create_rounded_input_with_icon(self, parent, variable, icon, bg = "parameters_button_background"):
        """
        Creates a rounded entry input widget with an icon on the left.

        The icon can be an image or a string (e.g., emoji or text). The entry is
        styled with a rounded background and bound to a `StringVar`.

        Args:
            parent (tk.Widget): The parent widget to contain the input field.
            variable (str | tk.StringVar | None): The value or variable to bind.
            icon (Union[str, ImageTk.PhotoImage]): The icon to display at the start of the input.
            bg (str, optional): Key name for the background color in the theme. 
                Defaults to "parameters_button_background".

        Returns:
            Tuple[tk.StringVar, tk.Entry]: A tuple containing the bound variable and the Entry widget itself.
        """
        # Change the variable into a StringVar if needed
        if isinstance(variable, str):
            variable = tk.StringVar(value=variable)
        elif variable is None:
            variable = tk.StringVar() 
        
        # Dimentions of the canvas
        canvas_width = 190
        canvas_height = 35
        radius = 20
        icon_width = 35

        # Create the canvas
        canvas = tk.Canvas(parent, width=canvas_width, height=canvas_height,
                        bg=parent.cget("bg"), highlightthickness=0, takefocus=0)
        canvas.pack(fill=tk.X, pady=(0, 0))

        # Draw the background
        self.draw_rounded_rect(canvas, 0, 0, canvas_width, canvas_height,
                            radius, fill=self.colors.theme[bg],
                            outline=self.colors.theme[bg], tag="input_bg")

        # Add the icon
        if isinstance(icon, str):
            # It's a text/emoji icon
            tk.Label(canvas, text=icon, bg=self.colors.theme[bg],
                    fg=self.colors.theme["secondary_text"], font=(self.font, 12)).place(x=5, rely=0.5, anchor="w", takefocus=0)
        else:
            # Assume it's an image (PhotoImage or ImageTk.PhotoImage)
            canvas.create_image(10, canvas_height // 2, anchor="w", image=icon)
            canvas.image = icon  # Prevent garbage collection


        # Create the Entry widget
        entry = tk.Entry(canvas, textvariable=variable, font=(self.font, 10), bd=0, relief="flat", highlightthickness=0,
                 bg=self.colors.theme[bg], fg=self.colors.theme["tertiary_text"],
                 insertbackground=self.colors.theme["primary_text"], 
                 disabledbackground=self.colors.theme[bg],
                 disabledforeground=self.colors.theme["tertiary_text"])

        # Place the entry widget inside the canvas. Adjust x, y for padding.
        entry_width = canvas_width - icon_width - radius
        entry_height = canvas_height - 10
        canvas.create_window(icon_width, canvas_height // 2, window=entry, anchor="w",
                            width=entry_width, height=entry_height)

        return variable, entry
     
    def create_rounded_dropdown(self, parent, options, variable, bg = "parameters_button_background"):
        """Creates a styled dropdown (combobox) with a rounded background.

        The dropdown is placed inside a canvas and styled to match the application's theme.
        It uses a readonly ttk.Combobox bound to a StringVar.

        Args:
            parent (tk.Widget): The parent widget where the dropdown is rendered.
            options (List[str]): List of selectable string options in the dropdown.
            variable (str | tk.StringVar | None): Initial value or variable to bind.
            bg (str, optional): Key name for the background color in the theme. 
                Defaults to "parameters_button_background".

        Returns:
            Tuple[tk.StringVar, ttk.Combobox]: The bound variable and the dropdown widget.
        """
        # Change the variable into a StringVar if needed
        if isinstance(variable, str):
            variable = tk.StringVar(value=variable)
        elif variable is None:
            variable = tk.StringVar()        
        
        # Dimentions of the canvas
        canvas_width = 190 
        canvas_height = 35 
        radius = 20  # Corner radius

        # Create the canvas
        canvas = tk.Canvas(parent, width=canvas_width, height=canvas_height,
                        bg=parent.cget("bg"), highlightthickness=0, takefocus=0)
        canvas.pack(fill=tk.X, pady=(0, 0))

        # Draw the background
        self.draw_rounded_rect(canvas, 0, 0, canvas_width, canvas_height,
                            radius, fill=self.colors.theme[bg],
                            outline=self.colors.theme[bg], tag="dropdown_bg")

        # Create the widget
        combo = ttk.Combobox(
            canvas,
            values=options,
            textvariable=variable, 
            font=(self.font, 10),
            state='readonly',
            justify='left',
            style='MyCombobox.TCombobox'
        )

        # Place the widget inside the canvas. Adjust x, y for padding.
        combo_width = canvas_width - 10
        combo_height = canvas_height - 10
        canvas.create_window(5, canvas_height / 2, window=combo, anchor="w",
                            width=combo_width, height=combo_height)

        return variable, combo
    
    def create_custom_toggle(self, parent, label, boolean_var, size="small", bg="secondary_background"):
        """
        Creates a custom toggle switch with an icon that reflects its state.

        This toggle displays a label and an image-based switch that changes appearance 
        when toggled. It binds to a BooleanVar and visually updates when clicked.

        Args:
            parent (tk.Widget): The parent widget where the toggle will be added.
            label (str): The text label describing the toggle's function.
            boolean_var (tk.BooleanVar): The variable controlling the toggle's state. 
                Clicking the toggle updates this variable.

        Returns:
            tk.Canvas: The canvas containing the toggle, which includes the image.
        """
        # Create the frame where the toggle will be set
        frame = tk.Frame(parent, bg=self.colors.theme[bg])
        frame.pack(fill=tk.X, pady=(5, 5))

        # Add the label of the toggle
        tk.Label(frame, text=label, bg=self.colors.theme[bg],
                fg=self.colors.theme["secondary_text"], font=(self.font, 10), takefocus=0).pack(side=tk.LEFT)

        # Create the toggle
        if size == "small":
            open_toggle_icon = self.toggle_open_icon
            close_toggle_icon = self.toggle_close_icon
        elif size == "big":
            open_toggle_icon = self.big_toggle_open_icon
            close_toggle_icon = self.big_toggle_close_icon
            
        toggle_canvas = tk.Canvas(frame, width=open_toggle_icon.width(),
                                height=open_toggle_icon.height(),
                                bg=self.colors.theme["primary_background"], highlightthickness=0, takefocus=0)
        toggle_canvas.pack(side=tk.RIGHT, padx=3)

        def draw_toggle():
            toggle_canvas.delete("all")
            image = open_toggle_icon if not boolean_var.get() else close_toggle_icon
            toggle_canvas.create_image(0, 0, image=image, anchor=tk.NW)

        def toggle_command(event=None):
            boolean_var.set(not boolean_var.get())  # This will trigger trace_add
            draw_toggle()

        toggle_canvas.toggle_command = toggle_command
        toggle_canvas.bind("<Button-1>", toggle_command)
        draw_toggle()
        
        return toggle_canvas
          
    def create_rounded_button(self, parent, text, command, bg_color, text_color,
                          hover_color, font, width_pixels, height_pixels,
                          corner_radius, side, padx=0, pady=0, padx_text=0, pady_text=0,
                          anchor='center', border_width=0, border_color=None, icon=None, icon_hover=None, 
                          icon_path=None, icon_hover_path=None, autoresize=False, expand=False, fill=None):
        """Creates a stylized button with rounded corners and optional icon.

        The button is rendered on a canvas and supports hover effects, click binding,
        and image swapping when hovered. It can display text only or an icon with text.
        
        If autoresize is True, the button will redraw itself to fill the canvas width/height
        on resize events.

        Args:
            parent (tk.Widget): The parent widget for the button.
            text (str): The button label.
            command (Callable): The function to execute on click.
            bg_color (str): The background color of the button.
            text_color (str): The text color.
            hover_color (str): The background color on hover.
            font (Tuple): Font tuple for the button text.
            width_pixels (int): Width of the button in pixels (initial width if autoresize=True).
            height_pixels (int): Height of the button in pixels (initial height if autoresize=True).
            corner_radius (int): Radius of the button's rounded corners.
            side (str): Packing side for the button (e.g., tk.LEFT, tk.RIGHT).
            padx (int, optional): Horizontal padding around the button. Defaults to 0.
            pady (int, optional): Vertical padding around the button. Defaults to 0.
            padx_text (int, optional): X offset for the text inside the button. Defaults to 0.
            pady_text (int, optional): Y offset for the text inside the button. Defaults to 0.
            anchor (str, optional): Anchor point for the text placement. Defaults to 'center'.
            border_width (int, optional): Width of the border around the button. Defaults to 0.
            border_color (str, optional): Border color. If None, uses theme default stroke. Defaults to None.
            icon (PhotoImage, optional): Optional icon to display before the text. Defaults to None.
            icon_hover (PhotoImage, optional): Optional icon to display on hover. Defaults to None.
            icon_path (Path, optional): Path to the icon image for dynamic resizing.
            icon_hover_path (Path, optional): Path to the hover icon image for dynamic resizing.   
            autoresize (bool, optional): If True, the button will resize with its container. Defaults to False.
            expand (bool, optional): Whether to expand the button in its parent. Defaults to False.
            fill (str, optional): How to fill the button in its parent (tk.X, tk.Y, tk.BOTH). Defaults to None.

        Returns:
            tk.Canvas: The canvas containing the button.
        """
        # Default color border
        if border_color is None:
            border_color = self.colors.theme["stroke_button"]

        # Create Canvas
        canvas = tk.Canvas(
            parent,
            width=width_pixels,
            height=height_pixels,
            bg=parent.cget("bg"),
            highlightthickness=0,
            takefocus=0
        )
        canvas.pack(side=side, padx=padx, pady=pady, expand=expand, fill=fill)
        
        # We need to store these for the redraw
        canvas.bg_color_val = bg_color
        canvas.hover_color_val = hover_color
        canvas.border_color_val = border_color
        canvas.text_val = text
        canvas.icon_val = icon
        canvas.icon_hover_val = icon_hover
        canvas.corner_radius_val = corner_radius
        canvas.border_width_val = border_width
        canvas.padx_text_val = padx_text
        canvas.pady_text_val = pady_text
        canvas.anchor_val = anchor
        canvas.font_val = font
        canvas.text_color_val = text_color
        
        # Determine icon color from theme if possible, otherwise default to stroke_button
        # This is a bit hacky, ideally we'd pass fg_color explicitly or deduce it.
        # But looking at load_icon usage, most icons use stroke_button or icon color.
        # We will use stroke_button as default if we need to resize.
        icon_fg_color = self.colors.theme["stroke_button"]

        # Load PIL images if paths provided
        canvas.icon_pil = None
        canvas.icon_hover_pil = None
        canvas.icon_ratio = 0.6  # Default ratio target_height / button_height
        
        if icon_path:
            try:
                canvas.icon_pil = Image.open(str(icon_path)).convert("RGBA")
                
                # Determine initial size ratio from the passed icon if it exists, otherwise use default
                if icon:
                     canvas.icon_ratio = icon.height() / height_pixels
            except Exception as e:
                print(f"Error loading icon path: {e}")

        if icon_hover_path:
             try:
                canvas.icon_hover_pil = Image.open(str(icon_hover_path)).convert("RGBA")
             except Exception as e:
                print(f"Error loading hovering icon path: {e}")

        
        # Helper to draw the button graphics
        def draw_button(w, h, current_bg_color):
            canvas.delete("all")
            
            # Draw border
            self.draw_rounded_rect(
                canvas,
                0, 0,
                w, h,
                canvas.corner_radius_val,
                fill=canvas.border_color_val,
                outline=canvas.border_color_val,
                tag="button_border"
            )

            # Draw main shape inset by border_width
            inset = canvas.border_width_val        
            self.draw_rounded_rect(
                canvas,
                inset, inset,
                w - inset, h - inset,
                max(canvas.corner_radius_val - inset, 0),
                fill=current_bg_color,
                outline=current_bg_color,
                tag="button_shape"
            )

            # Build label (icon + text or text-only)
            nonlocal label_widget, icon_label, label_frame, text_label
            
            # If we are just redrawing the background, we don't need to destroy the label widget if it exists
            # We just need to reposition it.
            
            canvas.create_window(
                w / 2 - canvas.padx_text_val,
                h / 2 - canvas.pady_text_val,
                window=label_widget,
                anchor=canvas.anchor_val,
                tags="button_label"
            )
            # Ensure label widget background matches
            if icon:
                label_frame.config(bg=current_bg_color)
                icon_label.config(bg=current_bg_color)
                text_label.config(bg=current_bg_color)
            else:
                text_label.config(bg=current_bg_color)


        # Initial creation of widgets (only once)
        if icon:
            label_frame = tk.Frame(canvas, bg=bg_color)
            icon_label = tk.Label(label_frame, image=icon, bg=bg_color, takefocus=0)
            icon_label.image = icon
            
            # We will handle hover image swapping manually in on_enter/on_leave using config
            # But we store initial references in the label to allow easy access if needed
            icon_label.image_normal = icon
            icon_label.image_hover = icon_hover
            
            icon_label.pack(side=tk.LEFT, padx=(0, 5))
            text_label = tk.Label(label_frame, text=text, bg=bg_color, fg=text_color, font=font, takefocus=0)
            text_label.pack(side=tk.LEFT)
            label_widget = label_frame
        else:
            text_label = tk.Label(canvas, text=text, bg=bg_color, fg=text_color, font=font, takefocus=0)
            label_widget = text_label
            label_frame = None # Not used
            icon_label = None

        # Draw initially
        draw_button(width_pixels, height_pixels, bg_color)


        # Event handlers
        def on_enter(event):
            canvas.itemconfig("button_shape", fill=hover_color, outline=hover_color)
            if icon:
                # Update icon background
                icon_label.config(bg=hover_color)
                
                # Update icon image if available
                # If we are using PIL resizing, we need to use the resized hover image
                current_h = canvas.winfo_height()
                if canvas.icon_hover_pil and current_h > 1:
                     # We rely on on_resize to have generated the correct sized image
                     # But on_resize might update 'icon_label.image_hover'
                     if hasattr(icon_label, 'image_hover_resized'):
                         icon_label.config(image=icon_label.image_hover_resized)
                     elif icon_hover:
                         icon_label.config(image=icon_hover)
                elif icon_hover:
                    icon_label.config(image=icon_hover)
                    
                label_frame.config(bg=hover_color)
            text_label.config(bg=hover_color)

        def on_leave(event):
            canvas.itemconfig("button_shape", fill=bg_color, outline=bg_color)
            if icon:
                icon_label.config(bg=bg_color)
                
                # Restore normal icon
                if hasattr(icon_label, 'image_normal_resized'):
                    icon_label.config(image=icon_label.image_normal_resized)
                else: 
                     icon_label.config(image=icon_label.image_normal) # defaulting to initial icon if no resize logic ran
                     
                label_frame.config(bg=bg_color)
            text_label.config(bg=bg_color)
            
        def on_resize(event):
            # Only redraw if size actually changed to avoid cycles
            if event.width != canvas.winfo_reqwidth() or event.height != canvas.winfo_reqheight():
                 draw_button(event.width, event.height, bg_color)
                 
                 # Dynamic icon resizing
                 if canvas.icon_pil and event.height > 10:
                     # Calculate new icon size
                     target_icon_h = int(event.height * canvas.icon_ratio)
                     # Ensure even dimensions to avoid centering issues sometimes
                     if target_icon_h % 2 != 0: target_icon_h -= 1
                     target_icon_w = target_icon_h # Assuming square icons for now as per load_icon usage
                     
                     if target_icon_h > 4: # Minimal size check
                         # Resize normal icon
                         bg_col = bg_color # normal bg
                         # We need to match the current state color if we are hovering? 
                         # Actually complicated because on_resize happens independently of hover state.
                         # Simpler: Generate both versions (normal and hover) with their respective backgrounds.
                         
                         # Normal state
                         img_normal = self.flatten_and_resize_pil(
                             canvas.icon_pil, target_icon_w, target_icon_h, bg_color, icon_fg_color
                         )
                         photo_normal = ImageTk.PhotoImage(img_normal)
                         icon_label.image_normal_resized = photo_normal
                         
                         # Hover state
                         img_src_hover = canvas.icon_hover_pil if canvas.icon_hover_pil else canvas.icon_pil
                         img_hover = self.flatten_and_resize_pil(
                             img_src_hover, target_icon_w, target_icon_h, hover_color, icon_fg_color
                         )
                         photo_hover = ImageTk.PhotoImage(img_hover)
                         icon_label.image_hover_resized = photo_hover
                         
                         # Update current display based on mouse position?
                         # Or just default to normal and let on_enter/leave handle it.
                         # Since resize usually happens when user drags window (mouse might be anywhere),
                         # safer to check under mouse or just reset to normal logic. 
                         # But simplest is to just update the 'image' config
                         
                         # If we are currently hovering (how to know?), we should show hover image.
                         # For now, let's just update the specific attributes and redisplay what is expected.
                         # On resize, usually reset to normal or keep current.
                         
                         # Let's just update the displayed image to the new Normal one, 
                         # unless we are properly tracking state. Canvas doesn't easily convert "is mouse over me".
                         # We can force a re-check or just update the stored images and set the current one.
                         
                         icon_label.config(image=photo_normal) # Reset to normal on resize for stability
                         
                         # Also text font resizing could go here if we wanted to be fancy, but out of scope.

        if autoresize:
            canvas.bind("<Configure>", on_resize)


        def on_click(event):
            command()

        # Function to bind events to a widget and all its children recursively
        def bind_events_recursive(widget):
            widget.bind("<Button-1>", on_click)
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)
            
            # If the widget has children, bind them too
            try:
                for child in widget.winfo_children():
                    bind_events_recursive(child)
            except:
                pass  # Some widgets might not have children

        # Bind canvas for click and hover
        bind_events_recursive(canvas)
        
        # Also bind the label widget and all its children
        bind_events_recursive(label_widget)
    
        return canvas

    def draw_rounded_rect(self, canvas, x1, y1, x2, y2, radius, fill, outline, tag):
        """
        Draws a rounded rectangle on a Tkinter canvas.

        This method creates a smoothed polygon approximating a rounded rectangle
        using a list of points and splines.

        Args:
            canvas (tk.Canvas): The canvas where the shape will be drawn.
            x1 (int): Left coordinate.
            y1 (int): Top coordinate.
            x2 (int): Right coordinate.
            y2 (int): Bottom coordinate.
            radius (int): The radius of the corners.
            fill (str): Fill color.
            outline (str): Outline color.
            tag (str): A tag name to assign to the drawn shape for later reference.
        """
        points = [
            (x1 + radius, y1),
            (x2 - radius, y1),
            (x2, y1),
            (x2, y1 + radius),
            (x2, y2 - radius),
            (x2, y2),
            (x2 - radius, y2),
            (x1 + radius, y2),
            (x1, y2),
            (x1, y2 - radius),
            (x1, y1 + radius),
            (x1, y1),
        ]
        canvas.create_polygon(points, fill=fill, outline=outline, smooth=True, splinesteps=36, tags=tag)

    # --- Command ---            
    def refresh_ui(self):   
        """
        Refreshes the UI in a minimal, safer way by performing a cleanup
        before re-initializing. This avoids common crashes caused by
        pending after callbacks or live-update loops referencing destroyed widgets.
        """
        try:
            # 1) Minimal cleanup to cancel pending callbacks and stop live loops
            try:
                self._cleanup_for_reinit()
            except Exception:
                pass

            # 2) Attempt to remove main widgets (so __init__ can rebuild cleanly)
            try:
                if hasattr(self, "main_frame"):
                    self.main_frame.destroy()
                # Clear references to destroyed widgets to prevent TclError in re-init
                self.content_frame = None
                self.params_frame = None
                self.sidebar = None
                self.main_content = None
                self.body_frame = None
            except Exception:
                pass

            # 3) Re-initialize UI state by calling __init__ (kept for minimal change)
            #    We pass existing CORE and state flags so the app restarts in the same mode.
            try:
                self.__init__(self.root, self.CORE, self.dark_mode, self.current_page, self.show_parameters, self.live_image, self.id_worm_seen)
            except Exception as e:
                # fallback: try a safer recreate of main_frame if __init__ fails
                self.context_error = log_error(e, "Refresh UI reinit failed")
        except Exception as e:
            self.context_error = log_error(e, "Refresh UI failed")

    def refresh_parameters_interface(self):
        """
        Destroys and recreates the parameters panel.

        This is a more focused refresh than `refresh_ui` and is used to update
        the parameter widgets without rebuilding the entire application. It is
        useful when parameters might have changed and need to be redrawn.
        """
        if hasattr(self, "params_frame"):
            self.params_frame.destroy()
        self.create_parameters_panel()
        
    def update_parameter_widgets_state(self, disabled_widgets):
        """
        Updates the state of parameter widgets, enabling or disabling them
        based on the application's current mode or settings.

        This ensures that users can only interact with relevant parameters at
        any given time, preventing invalid input. For example, some parameters
        might be disabled during a live scan.

        Args:
            disabled_widgets (list): A list of strings, where each string is the
                                    key of a widget to be disabled.
        """
        all_widgets = {
            "exposure_time": self.exposure_time_entry,
            "binning": self.binning_dropdown,
            #"shutter": self.shutter_toggle,
            "dual_view": self.dual_view_toggle,
            #"display_mode": self.display_mode_dropdown,
            "scan_objective": self.scan_objective_dropdown,
            "fluo_objective": self.fluo_objective_dropdown,
            "scan_shape": self.scan_shape_dropdown,
            "model_name": self.model_name_dropdown,
            "fov_size_um": self.fov_size_um_entry
        }
        
        for key, widget in all_widgets.items():
            if key in disabled_widgets:
                try:
                    if isinstance(widget, tk.Canvas):  # Handle custom toggle
                        widget.unbind("<Button-1>")
                    else:
                        widget.configure(state="disabled")
                except:
                    pass # Can happen when there is no dual view
            else:
                try:
                    if isinstance(widget, ttk.Combobox):
                        widget.configure(state="readonly")
                    elif isinstance(widget, tk.Canvas):
                        widget.bind("<Button-1>", widget.toggle_command)
                    else:
                        widget.configure(state="normal")
                except:
                    pass # Can happen when there is no dual view
        
        # Store only enabled widget keys
        self.enable_parameters_buttons = [key for key in all_widgets if key not in disabled_widgets]

    def toggle_dark_mode(self):
        """
        Toggles the application's color theme between light and dark mode.

        This method updates the internal `dark_mode` state, retrieves the new
        color palette, and then calls `refresh_ui` to apply the changes
        to all widgets.
        """
        self.dark_mode = not self.dark_mode
        self.update_colors()
        try:
            self.refresh_ui()
        except Exception as e:
            self.context_error = log_error(e, "Toggle dark mode failed")

    def toggle_parameters(self):
        """
        Toggles the visibility of the parameters panel.

        This method shows or hides the `params_frame` and then resizes the
        main content area if the current page is a scan-related page, ensuring
        the layout adapts correctly to the change in panel visibility.
        """
        try:
            self.show_parameters = not self.show_parameters
            if self.show_parameters:
                if hasattr(self, "content_frame") and self.content_frame.winfo_manager():
                     self.params_frame.pack(side=tk.RIGHT, fill=tk.Y, before=self.content_frame)
                else:
                     self.params_frame.pack(side=tk.RIGHT, fill=tk.Y)
            else:
                self.params_frame.pack_forget()
            
            # Store the after_id and schedule resizing with error handling
            if hasattr(self, 'main_content') and self.main_content.winfo_exists() and (self.current_page == "automatic_scan" or self.current_page == "scan_result"):
                after_id = self.main_content.after(50, self.resize_scan_content_area)
                if not hasattr(self, '_after_ids'):
                    self._after_ids = []
                self._after_ids.append(after_id)
        except Exception as e:
            self.context_error = log_error(e, "Toggle parameters panel failed")
    
    def toggle_mode_worm_scan_result(self, mode):
        """
        Toggles the state for adding a new worm scan result and refreshes the
        scan result page.

        This is used in a specific workflow where the user is adding new data
        to the result set. It triggers the `show_result_scan_page` method to
        update the UI.
        """
        try:
            if mode == "add":
                self.worm_scan_result_mode = "add"
            elif mode == "delete":
                self.worm_scan_result_mode = "delete"
            elif mode == "move":
                self.worm_scan_result_mode = "move"
            self.show_result_scan_page()
        except Exception as e:
            self.context_error = log_error(e, f"Toggle add worm scan result failed")

    def switch_page(self, page_id):
        """
        Switches the application to a new page.

        This is the core navigation method. It updates the internal `current_page`
        state and calls `refresh_ui` to rebuild the interface for the new page.

        Args:
            page_id (str): A string identifier for the new page.
        """
        try:
            self.current_page = page_id
            self.refresh_ui()
            self._show_enhanced_preview = False
            self.root.unbind_all("<Left>")
            self.root.unbind_all("<Right>")
            self.root.unbind_all("<Up>")
            self.root.unbind_all("<Down>")
            self.root.unbind_all("<space>")
            self.root.update_idletasks()
        except Exception as e:
            self.context_error = log_error(e, f"Switch page {page_id} failed")
    
    def resize_scan_content_area(self, event=None):
        """
        Resizes the main content area for scan-related pages (`automatic_scan`
        and `scan_result`) to fit within its container, maintaining a specified
        aspect ratio (e.g., square or rectangle).

        This ensures that the live image or scan result image is always centered
        and properly sized within the available space. It also resizes the
        displayed image accordingly.
        """
        try:
            if self.current_page == "automatic_scan":
                middle_container = self.middle_container_ref
                content_area = self.content_area_ref
            elif self.current_page == "scan_result":
                middle_container = self.middle_result_container_ref
                content_area = self.content_area_result_container_ref
            else:
                return

            # Use event dimensions if available and relevant to the container
            if event and event.widget == middle_container:
                container_width = event.width
                container_height = event.height
            else:
                container_width = middle_container.winfo_width()
                container_height = middle_container.winfo_height()

            if self.shape.get() == 'Square':
                side = min(container_width, container_height)
                width = height = side
            elif self.shape.get() == 'Rectangle':
                x_length = int(self.loaded_params.get("scan_height_length"))
                y_length = int(self.loaded_params.get("scan_width_length"))
                proportion = y_length/x_length
                height = min(container_height, container_width / proportion) 
                width = proportion * height # Here is the only way of modifying the size of the result scan container
            else:
                height = min(container_height, container_width)
                width = height

            x = (container_width - width) / 2
            y = (container_height - height) / 2

            content_area.place(x=x, y=y, width=width, height=height)
            
            self.last_scan_area_size = (int(width), int(height))
            if width <= 0 or height <= 0:
                return
            
            # --- Resize image accordingly ---
            if hasattr(self, 'img_canvas') and self.img_canvas.winfo_exists():
                self.update_canvas_image(width=int(width), height=int(height))
                self.draw_worms_on_canvas(width=int(width), height=int(height))
        except Exception as e:
            self.context_error = log_error(e, f"Resize scan content area failed")
     
    def resize_automatic_live_image(self, event):
        """
        Keeps the live image label a centered square inside the wrapper frame.
        event.width/event.height are the wrapper's inner size.
        """
        try:
            w = event.width
            h = event.height
            size = min(w, h)
            x = (w - size) // 2
            y = (h - size) // 2
            # Avoid negative sizes
            if size <= 0:
                size = 0
            if hasattr(self, "automatic_live_image_label") and self.automatic_live_image_label.winfo_exists():
                self.automatic_live_image_label.place(x=x, y=y, width=size, height=size)
        except Exception:
            # don't let UI break on odd races during teardown
            pass
        
    def resize_live_image(self, event):
        """
        Resizes and centers the live image container for the `assist_acquisition`
        and `load_position` pages.

        This function is typically bound to a `Configure` event, allowing the
        live image display to dynamically resize as the main window changes.

        Args:
            event (tk.Event): The event object from the `Configure` event,
                            containing the new width and height.
        """
        try:
            w, h = event.width, event.height
            
            # New case for length_analysis page (maximize square in container)
            if self.current_page == "length_analysis":
                 size = min(w, h)
                 x = (w - size) // 2
                 y = (h - size) // 2
                 self.live_analysis_container_ref.place(x=x, y=y, width=size, height=size)
                 return

            size = min(w, h - 80)  # leave space for bottom button
            x = (w - size) // 2
            if self.current_page == "assist_acquisition":
                self.live_assist_container_ref.place(x=x, y=0, width=size, height=size)
            elif self.current_page == "load_position":
                self.live_analysis_container_ref.place(x=x, y=0, width=size, height=size)
        except Exception as e:
            self.context_error = log_error(e, f"Resize live image failed")
    
    def resize_map_assist(self, event):
        """
        Resizes and repositions the map container for the `assist_acquisition` page.

        This ensures the map remains a square and is positioned correctly at the
        bottom of the container, adapting to window size changes.

        Args:
            event (tk.Event): The event object from the `Configure` event.
        """
        w, h = event.width, event.height
        size = min(w, h) 
        x = (w - size) // 2
        y = h - size - 10  # 10 px from bottom
        self.map_assist_containter_ref.place(x=x, y=y, width=size, height=size)
       
    def resize_prediction_result_box(self, event):
        """
        Resizes and repositions a rounded rectangle on a canvas to center it
        within the canvas.

        This function is used to create a visually centered and responsive
        container for a prediction result label.

        Args:
            event (tk.Event): The event object from the `Configure` event.
        """
        try:
            canvas_width = event.width
            canvas_height = event.height
            self.top_label_canvas.coords(self.top_label_frame_window, canvas_width / 2, canvas_height / 2)   
            
            # Draw the rectangle
            canvas_width = event.width
            canvas_height = event.height

            rect_width = 299
            rect_height = 98

            x1 = (canvas_width - rect_width) / 2
            y1 = (canvas_height - rect_height) / 2
            x2 = x1 + rect_width
            y2 = y1 + rect_height

            # Remove previous rectangle if any
            self.top_label_canvas.delete("rounded_bg")

            self.draw_rounded_rect(
                self.top_label_canvas,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                radius=20,
                fill=self.colors.theme["primary_background"],
                outline=self.colors.theme["secondary_text"],
                tag="rounded_bg"
            )
        except Exception as e:
            self.context_error = log_error(e, f"Resize size prediction box failed")
               
    def launch_scan(self):
        """
        Initiates and manages the worm scanning process.

        This method orchestrates a series of steps:
        1. Displays status messages to the user.
        2. Initializes the scanner and starts the physical scan.
        3. Saves the positions of any detected worms.
        4. Reconstructs the final image from the scan.
        5. Switches the page to display the scan results.
        """

        # Show "Starting scan" message
        try:
            self.scan_status_label.config(text="Launching scan... please wait.")
            self.scan_status_label.update_idletasks()
            clear_scan_directory()
            try:
                self.CORE.setExposure(int(self.exposure_time.get()))
                self.CORE.setProperty(NAME_CAMERA, "Binning", self.binning.get())
            except Exception:
                pass
            scanner = ScanSlice(self.CORE, self.scan_objective, self.dual_view, self.shape)
        except Exception as e:
            self.context_error = log_error(e, "Initialize scan failed")

        # Update: scanning
        self.scan_status_label.config(text="      Scanning in progress...      ")
        self.scan_status_label.update_idletasks()
        try:
            increment_user_statistics('nb_scans')
            # Get worms and corners
            worms_microscope_position, corners = scanner.scan()
            log_debug_coordinate(f"[Scan] Detected corners: {corners}")
            log_debug_coordinate(f"[Scan] Detected {len(worms_microscope_position)} worms")
            for i, w in enumerate(worms_microscope_position):
                log_debug_coordinate(f"[Scan] Worm {i} detected at microscope pos: {w}")

            
            # --- ATOMIC UPDATE OF PARAMETERS ---
            # We explicitly update the YAML file with both the new corners AND the new dimensions.
            # This prevents the race condition where updating UI variables (scan_width/height) 
            # might trigger a save that overwrites the corners if they weren't saved yet, 
            # or where reading the file gets stale data.
            
            try:
                # 1. Read current config
                current_config = {}
                if os.path.exists(self.PARAMS_FILE):
                    with open(self.PARAMS_FILE, "r") as f:
                        current_config = yaml.safe_load(f) or {}

                # 2. Update with new values (Corners + Dimensions)
                # Note: We must convert numpy/special types to standard python types if needed, 
                # but corners are likely floats and scan_width/height are ints.
                current_config.update(corners)
                current_config["scan_width"] = scanner.scan_width
                current_config["scan_height"] = scanner.scan_height
                
                # 3. Write back to file atomically (as much as possible)
                with open(self.PARAMS_FILE, "w") as f:
                    yaml.dump(current_config, f, default_flow_style=False, sort_keys=False)
                    
            except Exception as e:
                log_error(e, "Failed to save parameters atomically in launch_scan")

            # 4. Update UI variables (will trigger their own callbacks, but they will read the correct file now)
            self.init_pos_x = scanner.start_x
            self.init_pos_y = scanner.start_y
            self.scan_width.set(scanner.scan_width)
            self.scan_height.set(scanner.scan_height)

            # 5. Check if Core was reloaded during scan (e.g. error recovery)
            if scanner.mmc is not self.CORE:
                print("⚠️ Core instance changed during scan! Updating application reference.")
                self.CORE = scanner.mmc
                # Re-apply critical settings that might be lost or need ensuring
                try:
                    self.CORE.setExposure(int(self.exposure_time.get()))
                    self.CORE.setProperty(NAME_CAMERA, "Binning", self.binning.get())
                except Exception as e_settings:
                    log_error(e_settings, "Failed to re-apply settings after Core update")
        except Exception as e:
            self.context_error = log_error(e, "Launch scan failed")
        
        # Update: saving worm positions
        self.scan_status_label.config(text="Saving worm positions...")
        self.scan_status_label.update_idletasks()
        try:
            self.worms_position = WormPositionManager(table_worm_position=worms_microscope_position, corners=corners)
            update_user_statistics('nb_vers_detected', self.worms_position.get_number_of_worms())
        except Exception as e:
            self.context_error = log_error(e, "Saving worm position failed")
        
        # Update: reconstructing image
        self.scan_status_label.config(text="Reconstructing scan result...")
        self.scan_status_label.update_idletasks()
        try:
            scanner.reconstruct_slice()
        except Exception as e:
            self.context_error = log_error(e, "Reconstruct slice failed")

        # Update: switching page
        self.scan_status_label.config(text="Scan complete. Displaying results...")
        self.scan_status_label.update_idletasks()
        self.id_worm_seen = 0
        self.switch_page("scan_result")
    
    def end_of_program(self): 
        """
        Performs cleanup tasks and prepares the application for shutdown.

        This includes:
        - Returning the microscope stage to its initial position.
        - Training a new machine learning model with the collected data.
        - Quitting the main application window.
        
        This method is designed to be robust, using a `try...except...finally`
        block to ensure that the application window quits even if an error
        occurs during the cleanup process.
        """ 
        try:    
            self.CORE.setXYPosition(self.CORE.getXYStageDevice(), self.init_pos_x, self.init_pos_y)
                
            # train model with new data
            big_dataset = Dataset_Manager()
            big_dataset.set_features(compute=False, name_dataset="big_dataset")
            big_dataset.remove_unclassified()
            big_dataset.get_model(compute=True)
            
            self.CORE.unloadAllDevices() 
            self.CORE.shutdown() 
        
        except Exception as e:
            self.context_error = log_error(e, f"Erreur durant le nettoyage du Core ou du stage")
            pass 
        
        finally:
            self.root.quit()
        
    def load_base_image(self):
        """
        Loads the stitched scan image into self.base_stitched_image.
        Initializes worm positions if needed.
        """
        # Load original image
        try:
            image_path = Path(RESSOURCES_DIR) / "stitched_final.jpg"
            pil_image = Image.open(image_path)
            # Ensure RGB
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            self.base_stitched_image = pil_image
        except Exception as e:
            log_error(e, "Failed to load stitched_final.jpg")
            # Create a placeholder black image if loading fails
            self.base_stitched_image = Image.new("RGB", (1000, 1000), "black")

        # Initialize worms positions if needed
        if self.worms_position is None:
            self.worms_position = WormPositionManager(new_acquisition=False)

    def update_canvas_image(self, width=None, height=None):
        """
        Resizes self.base_stitched_image to fit the current canvas size 
        and updates the canvas background image.
        """
        if self.base_stitched_image is None:
            return

        if width is None or height is None:
            canvas_width = self.img_canvas.winfo_width()
            canvas_height = self.img_canvas.winfo_height()
        else:
            canvas_width = width
            canvas_height = height
        
        if canvas_width <= 1 or canvas_height <= 1:
            return

        # Calculate new size while preserving aspect ratio is handled by resize_scan_content_area logic
        # But here we just need to resize the image to the canvas size (which is already set correcty)
        image_width, image_height = self.base_stitched_image.size
        
        # calculate scale to fit the image into the canvas
        # Actually, the canvas size IS the desired image size effectively because of resize_scan_content_area
        # So we just resize the image to match the canvas dimensions exactly
        
        resized_image = self.base_stitched_image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        self.displayed_image = ImageTk.PhotoImage(resized_image)
        
        # Update canvas image item
        # We use a tag 'base_image' to easily update or delete it
        self.img_canvas.delete("base_image") 
        self.img_canvas.create_image(0, 0, image=self.displayed_image, anchor="nw", tags="base_image")
        self.img_canvas.tag_lower("base_image") # Ensure it's behind everything

        # Store scale factor for coordinate conversion
        # width_scale = current_width / original_width
        self.image_scale_x = canvas_width / image_width
        self.image_scale_y = canvas_height / image_height

    def draw_worms_on_canvas(self, width=None, height=None, exclude_id=None):
        """
        Clears existing worm boxes and redraws them based on self.worms_position
        using a single PIL overlay image for performance.
        
        Args:
            width (int, optional): Explicit width for the canvas.
            height (int, optional): Explicit height for the canvas.
            exclude_id (int, optional): ID of a worm to exclude from the static
                                        overlay (e.g., because it's being dragged).
        """
        # Clear the old overlay
        self.img_canvas.delete("worms_overlay")
        
        if self.worms_position is None:
            return

        all_worm_data = self.worms_position.get_all_worm_proportion_position()
        # all_worm_data is list of [worm_id, prop_x, prop_y]

        if width is None or height is None:
            canvas_width = self.img_canvas.winfo_width()
            canvas_height = self.img_canvas.winfo_height()
        else:
            canvas_width = width
            canvas_height = height
            
        if canvas_width <= 1 or canvas_height <= 1:
            return
    
        # Pre-calculate bounding box pixel size
        orig_width, orig_height = self.base_stitched_image.size
        
        curr_box_radius_x = self.bounding_box_size * (canvas_width / orig_width)
        curr_box_radius_y = self.bounding_box_size * (canvas_height / orig_height)
        
        # Ensure at least 2 pixels so it's visible
        curr_box_radius_x = max(2, curr_box_radius_x)
        curr_box_radius_y = max(2, curr_box_radius_y)

        # Create a transparent RGBA image for the overlay
        overlay_image = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay_image)

        # Pre-fetch all labels to avoid O(N^2) lookup
        # We can iterate the df directly or creating a map
        # Since 'all_worm_data' is separate, let's create a map
        if not self.worms_position.df.empty:
            # Create a map: worm_id -> label
            id_to_label = dict(zip(self.worms_position.df['worm_id'], self.worms_position.df['user_label']))
        else:
            id_to_label = {}

        for worm in all_worm_data:
            worm_id = worm[0]
            
            # Skip the excluded worm (it will be drawn separately, e.g. while dragging)
            if exclude_id is not None and worm_id == exclude_id:
                continue
                
            prop_x = worm[1]
            prop_y = worm[2]
            
            cx = prop_x * canvas_width
            cy = prop_y * canvas_height
            
            x1 = cx - curr_box_radius_x
            y1 = cy - curr_box_radius_y
            x2 = cx + curr_box_radius_x
            y2 = cy + curr_box_radius_y
            
            color = "red"
            # Fast lookup
            label = id_to_label.get(worm_id, 'None')
            if label == 'Mutant':
                color = "#00FF00" # Green

            # Draw rectangle on the PIL image
            draw.rectangle([x1, y1, x2, y2], outline=color, width=1)

        # Convert to PhotoImage and display
        self.worms_overlay_photo = ImageTk.PhotoImage(overlay_image)
        self.img_canvas.create_image(0, 0, image=self.worms_overlay_photo, anchor="nw", tags="worms_overlay")

        
    def on_stitching_image_click(self, event):
        """
        Handles click events on the stitched scan image to either remove an
        existing worm or add a new one.
        """
        # Get clicked coordinates in displayed image
        x_display, y_display = event.x, event.y

        # Get displayed image size
        display_width = self.img_canvas.winfo_width()
        display_height = self.img_canvas.winfo_height()
        
        if display_width == 0 or display_height == 0:
            return
            
        # Compute relative position
        x_mouse = float(x_display / display_width)
        y_mouse = float(y_display / display_height)   
        
        x_bounding_box_proportion = float(self.bounding_box_size / display_width)
        y_bounding_box_proportion = float(self.bounding_box_size / display_height)
        
        # Get scan image associated for a futur annotation and improvement of the model
        if self.worm_scan_result_mode == "add":
            # Simplified logic for adding worms - skipping complex image crop saving for performance/stability now
            # If that logic is needed, it should be re-added carefully with correct Canvas coordinate mapping
            
            # Check if we clicked on an existing worm (to avoid Adding on top of existing)
            for _, (id, x, y) in enumerate(self.worms_position.get_all_worm_proportion_position()):
                if x - x_bounding_box_proportion <= x_mouse <= x + x_bounding_box_proportion and \
                   y - y_bounding_box_proportion <= y_mouse <= y + y_bounding_box_proportion:
                    return

            # Add new worm (convert proportion to microscope)
            x_microscope, y_microscope = self.worms_position.transform_proportion_into_microscope_positions(x_mouse, y_mouse)
            self.worms_position.add_worm_microscope_position(x_microscope, y_microscope)
            
            # Redraw only worms
            self.draw_worms_on_canvas()

        elif self.worm_scan_result_mode == "move":
            # Start dragging
            for _, (id, x, y) in enumerate(self.worms_position.get_all_worm_proportion_position()):
                if x - x_bounding_box_proportion <= x_mouse <= x + x_bounding_box_proportion and \
                   y - y_bounding_box_proportion <= y_mouse <= y + y_bounding_box_proportion:
                    self._dragging_worm_id = id
                    self._drag_offset_prop = (x_mouse - x, y_mouse - y)
                    
                    # Create temporary scan-box for dragging
                    # We hide the static one by redrawing the overlay without this worm
                    # self.draw_worms_on_canvas(exclude_id=id)
                    
                    # Create a lightweight canvas rect for the dragged worm
                    orig_width, orig_height = self.base_stitched_image.size
                    curr_box_radius_x = self.bounding_box_size * (display_width / orig_width)
                    curr_box_radius_y = self.bounding_box_size * (display_height / orig_height)
                    curr_box_radius_x = max(2, curr_box_radius_x)
                    curr_box_radius_y = max(2, curr_box_radius_y)

                    cx = x * display_width
                    cy = y * display_height
                    
                    x1 = cx - curr_box_radius_x
                    y1 = cy - curr_box_radius_y
                    x2 = cx + curr_box_radius_x
                    y2 = cy + curr_box_radius_y
                    
                    color = "red"
                    label = self.worms_position.get_worm_label(id)
                    if label == 'Mutant':
                        color = "#00FF00" 

                    self.drag_rect = self.img_canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=1, tags="drag_rect")
                    break

        elif self.worm_scan_result_mode == "delete":
            to_delete = None
            for _, (id, x, y) in enumerate(self.worms_position.get_all_worm_proportion_position()):
                # Delete logic
                if x - x_bounding_box_proportion <= x_mouse <= x + x_bounding_box_proportion and \
                   y - y_bounding_box_proportion <= y_mouse <= y + y_bounding_box_proportion:
                    to_delete = id
                    break
            
            if to_delete is not None:
                self.worms_position.delete_worm(to_delete)
                self.draw_worms_on_canvas()

    def on_stitching_image_drag(self, event):
        """
        Handles drag events on the stitched scan image.
        """
        display_width = self.img_canvas.winfo_width()
        display_height = self.img_canvas.winfo_height()
        if display_width == 0 or display_height == 0:
            return

        x_mouse = float(event.x / display_width)
        y_mouse = float(event.y / display_height)

        if self.worm_scan_result_mode == "move" and self._dragging_worm_id is not None:
            # new center
            new_x_prop = x_mouse - self._drag_offset_prop[0]
            new_y_prop = y_mouse - self._drag_offset_prop[1]

            # clamp
            new_x_prop = max(0.0, min(1.0, new_x_prop))
            new_y_prop = max(0.0, min(1.0, new_y_prop))

            # Optimize: Update the temporary drag rect
            if hasattr(self, 'drag_rect') and self.drag_rect:
                # Calculate new pixel coordinates
                # We need the box size again
                orig_width, orig_height = self.base_stitched_image.size
                curr_box_radius_x = self.bounding_box_size * (display_width / orig_width)
                curr_box_radius_y = self.bounding_box_size * (display_height / orig_height)
                curr_box_radius_x = max(2, curr_box_radius_x)
                curr_box_radius_y = max(2, curr_box_radius_y)

                cx = new_x_prop * display_width
                cy = new_y_prop * display_height
                
                x1 = cx - curr_box_radius_x
                y1 = cy - curr_box_radius_y
                x2 = cx + curr_box_radius_x
                y2 = cy + curr_box_radius_y
                
                self.img_canvas.coords(self.drag_rect, x1, y1, x2, y2)
            
            # Defer update to release for performance
            return

        if self.worm_scan_result_mode == "delete":
            x_bounding_box_proportion = float(self.bounding_box_size / display_width)
            y_bounding_box_proportion = float(self.bounding_box_size / display_height)

            removed = False
            # Iterate over a copy or collect ids to delete first to avoid runtime error if we modified iterator?
            # get_all_worm_proportion_position returns a list (copy), so it is safe.
            for _, (id, x, y) in enumerate(self.worms_position.get_all_worm_proportion_position()):
                if x - x_bounding_box_proportion <= x_mouse <= x + x_bounding_box_proportion and \
                y - y_bounding_box_proportion <= y_mouse <= y + y_bounding_box_proportion:
                    
                    success = self.worms_position.fast_delete_worm(id)
                    if success:
                        removed = True
                        self.has_pending_deletions = True
                        # Break after one deletion per event to avoid deleting everything in one path?
                        # Or delete all under cursor? Usually one.
                        break

            if removed:
                self.draw_worms_on_canvas()

    def on_stitching_image_release(self, event):
        """
        Fin du drag : on libère l'état de dragging.
        """
        # If we were dragging, we need to save the final position now
        if self._dragging_worm_id is not None and hasattr(self, 'drag_rect'):
             # Logic similar to drag event to get final position
            display_width = self.img_canvas.winfo_width()
            display_height = self.img_canvas.winfo_height()
            
            if display_width > 0 and display_height > 0:
                x_mouse = float(event.x / display_width)
                y_mouse = float(event.y / display_height)
                
                # new center
                new_x_prop = x_mouse - self._drag_offset_prop[0]
                new_y_prop = y_mouse - self._drag_offset_prop[1]

                # clamp
                new_x_prop = max(0.0, min(1.0, new_x_prop))
                new_y_prop = max(0.0, min(1.0, new_y_prop))
                
                # Update DB once
                x_mic, y_mic = self.worms_position.transform_proportion_into_microscope_positions(new_x_prop, new_y_prop)
                self.worms_position.update_worm_position(self._dragging_worm_id, x_mic, y_mic)

        self._dragging_worm_id = None
        self._drag_offset_prop = (0.0, 0.0)
        
        # Remove drag rect
        if hasattr(self, 'drag_rect'):
            self.img_canvas.delete("drag_rect")
        self._dragging_worm_id = None
        self.drag_rect = None

        # Handle pending deletions commit
        if self.worm_scan_result_mode == "delete" and getattr(self, 'has_pending_deletions', False):
            self.worms_position.commit_deletions()
            self.draw_worms_on_canvas()
            self.has_pending_deletions = False
            
        # Standard full redraw to be safe/clean
        # (Already handled above for delete, but needed for Move end)
        if self.worm_scan_result_mode == "move":
             self.draw_worms_on_canvas()
        
    def delete_all_worms(self):
        """
        Deletes all recorded worm positions from the dataset and updates the image displayed.
        """
        self.worms_position.delete_all_worms()
        self.draw_worms_on_canvas()
        
    # Assist acquisition page
    def add_worm_assist_acquisition(self):
        """
        Adds a new worm position to the dataset based on the current microscope
        stage coordinates.

        This method is used on the `assist_acquisition` page to manually record
        the position of a worm that the user has found. It retrieves the
        current x and y coordinates from the microscope core and adds them
        to the worm position manager.
        """
        x_microscope, y_microscope = self.CORE.getXYPosition()
        self.worms_position.add_worm_microscope_position(x_microscope, y_microscope)
    
    # load position page
    def on_live_image_press(self, event):
        """
        Handles mouse press events on the live image to initialize drag or click.
        """
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.is_dragging = False

    def on_live_image_drag(self, event):
        """
        Handles mouse drag events on the live image to move the microscope (pan).
        """
        if not hasattr(self, 'drag_start_x') or not hasattr(self, 'drag_start_y'):
            return

        # Threshold to consider it a drag
        if not self.is_dragging:
            if abs(event.x - self.drag_start_x) > 5 or abs(event.y - self.drag_start_y) > 5:
                self.is_dragging = True
            else:
                return

        # Calculate delta
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        
        # Update start position for next drag event (incremental move)
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        
        # Get displayed image size
        display_width = self.live_image_label.winfo_width()
        display_height = self.live_image_label.winfo_height()
        
        # Calculate scale factor
        try:
            fov_size_um = float(self.fov_size_um.get())
        except ValueError:
             fov_size_um = 1000
             print("[WARNING] FOV size is not a valid number. Using default value of 1000.")
        
        if MICROSCOPE == "Macrozoom":
            delta_stage_x = (dx / display_width) * fov_size_um
            delta_stage_y = (dy / display_height) * fov_size_um
        elif MICROSCOPE == "Nikon":
            delta_stage_y = (dx / display_width) * fov_size_um
            delta_stage_x = -(dy / display_height) * fov_size_um
        
        # Move microscope
        if self.CORE:
            try:
                current_x, current_y = self.CORE.getXYPosition()
                new_x = current_x + delta_stage_x
                new_y = current_y + delta_stage_y
                self.CORE.setXYPosition(self.CORE.getXYStageDevice(), new_x, new_y)
                
                # Update worm position if we are tracking one
                if self.worms_position:
                    id = self.worms_position.get_id_worm_seen()
                    self.worms_position.update_worm_position(id, new_x, new_y)
            except Exception as e:
                log_error(e, "Drag move failed")

    def on_live_image_release(self, event):
        """
        Handles mouse release events. If it was a click (not drag), center the image.
        """
        if hasattr(self, 'is_dragging') and not self.is_dragging and MICROSCOPE == "Nikon":
            self.on_live_image_click(event)
        
        # Reset state
        self.is_dragging = False

    def on_live_image_click(self, event):
        """
        Handles click events on the live image to move the camera to the position being clicked.

        Args:
            event (tk.Event): The event object from the click, containing
                            the `x` and `y` coordinates of the click.
        """
        if MICROSCOPE == "Nikon":
            self.clear_enhanced_preview()
        
        # Get clicked coordinates in displayed image
        x_display, y_display = event.x, event.y

        # Get displayed image size
        display_width = self.live_image_label.winfo_width()
        display_height = self.live_image_label.winfo_height()
        
        # Compute position
        config = load_config_file()
        objective = int(self.fluo_objective.get().replace("x", ""))
        y_mouse = 1 - float(x_display / display_width)
        x_mouse = float(y_display / display_height)  
        display_real_size = int(int(config.get("microscope_step_size")) / objective)
        
        # Move microscope to the clicked position
        x_microscope, y_microscope = self.CORE.getXYPosition()
        x_new = x_microscope + (x_mouse - 0.5) * display_real_size
        y_new = y_microscope + (y_mouse - 0.5) * display_real_size
        self.CORE.setXYPosition(self.CORE.getXYStageDevice(), x_new, y_new)

        # Save new position in the worm position csv file
        id = self.worms_position.get_id_worm_seen()
        self.worms_position.update_worm_position(id, x_new, y_new)

    def move_microscope_relative(self, direction):
        """
        Moves the microscope stage relative to the current position based on direction.
        
        Args:
            direction (str): 'up', 'down', 'left', 'right'
        """
        try:
            # Get current position
            if self.CORE is None:
                return
                
            x_current, y_current = self.CORE.getXYPosition()
            
            try:
                step_size = float(self.fov_size_um.get()) / 10
            except ValueError:
                step_size = 100
                print("[WARNING] FOV size is not a valid number. Using default value of 1000.")
                
            if MICROSCOPE == "Macrozoom":
                # Calculate new position based on direction
                if direction == 'left':
                    x_new = x_current + step_size
                    y_new = y_current
                elif direction == 'right':
                    x_new = x_current - step_size
                    y_new = y_current
                elif direction == 'up':
                    x_new = x_current
                    y_new = y_current + step_size
                elif direction == 'down':
                    x_new = x_current
                    y_new = y_current - step_size
                else:
                    return
            elif MICROSCOPE == "Nikon":
                # Calculate new position based on direction
                if direction == 'down':
                    x_new = x_current + step_size
                    y_new = y_current
                elif direction == 'up':
                    x_new = x_current - step_size
                    y_new = y_current
                elif direction == 'left':
                    x_new = x_current
                    y_new = y_current + step_size
                elif direction == 'right':
                    x_new = x_current
                    y_new = y_current - step_size
                else:
                    return


            self.CORE.setXYPosition(self.CORE.getXYStageDevice(), x_new, y_new)
            
            # Update worm position
            if self.worms_position:
                 id = self.worms_position.get_id_worm_seen()
                 self.worms_position.update_worm_position(id, x_new, y_new)
                 
        except Exception as e:
            log_error(e, f"Relative move {direction} failed")

    def go_to_next_worm(self, event=None):
        """
        Navigates the microscope stage to the position of the next worm in the
        recorded list.

        This method updates the internal state to point to the next worm,
        updates the UI label showing the current worm ID, and then commands
        the microscope stage to move to the new worm's coordinates.
        """
        try:
            self.clear_enhanced_preview()
            self.worms_position.go_to_next_worm() # set "seen" to True to the next worm
            self.id_worm_seen = self.worms_position.get_id_path_worm_seen() # get the id of the newt worm
            self.id_worm_seen_label.config(text=f"{self.id_worm_seen+1}/{self.worms_position.get_number_of_worms()}")

            x,y = self.worms_position.get_worm_microscope_position(self.worms_position.get_id_worm_seen())
            time.sleep(0.01)
        except Exception as e:
            self.context_error = log_error(e, "Get go to next worm position failed")
            
        try:
            self.CORE.setXYPosition(self.CORE.getXYStageDevice(), x, y)
        except Exception as e:
            self.context_error = log_error(e, "Microscope move to next worm failed")
        
    def go_to_last_worm(self, event=None):
        """
        Navigates the microscope stage to the position of the last seen worm in
        the recorded list.

        This method updates the internal state to point to the last worm,
        updates the UI label showing the current worm ID, and then commands
        the microscope stage to move to the new worm's coordinates. This is
        useful for reviewing or re-analyzing a previously seen worm.
        """
        try:
            self.clear_enhanced_preview()
            self.worms_position.go_to_last_worm() # set "seen" to True to the last worm
            self.id_worm_seen = self.worms_position.get_id_path_worm_seen() # get the id of the newt worm
            self.id_worm_seen_label.config(text=f"{self.id_worm_seen+1}/{self.worms_position.get_number_of_worms()}")

            x,y = self.worms_position.get_worm_microscope_position(self.worms_position.get_id_worm_seen())
            time.sleep(0.01)
        except Exception as e:
            self.context_error = log_error(e, f"Get go to last worm position failed")
            
        try:
            self.CORE.setXYPosition(self.CORE.getXYStageDevice(), x, y)
        except Exception as e:
            self.context_error = log_error(e, f"Microscope move to last worm failed")

    def go_to_next_mutant(self, event=None):
        """
        Navigates the microscope stage to the position of the next mutant worm.
        """
        try:
            self.clear_enhanced_preview()
            self.worms_position.go_to_next_mutant() 
            self.id_worm_seen = self.worms_position.get_id_path_worm_seen() 
            self.id_worm_seen_label.config(text=f"{self.id_worm_seen+1}/{self.worms_position.get_number_of_worms()}")

            x,y = self.worms_position.get_worm_microscope_position(self.worms_position.get_id_worm_seen())
            time.sleep(0.01)
        except Exception as e:
            self.context_error = log_error(e, "Get go to next mutant position failed")
            
        try:
            self.CORE.setXYPosition(self.CORE.getXYStageDevice(), x, y)
        except Exception as e:
            self.context_error = log_error(e, "Microscope move to next mutant failed")

    def go_to_last_mutant(self, event=None):
        """
        Navigates the microscope stage to the position of the previous mutant worm.
        """
        try:
            self.clear_enhanced_preview()
            self.worms_position.go_to_last_mutant() 
            self.id_worm_seen = self.worms_position.get_id_path_worm_seen() 
            self.id_worm_seen_label.config(text=f"{self.id_worm_seen+1}/{self.worms_position.get_number_of_worms()}")

            x,y = self.worms_position.get_worm_microscope_position(self.worms_position.get_id_worm_seen())
            time.sleep(0.01)
        except Exception as e:
            self.context_error = log_error(e, "Get go to last mutant position failed")
            
        try:
            self.CORE.setXYPosition(self.CORE.getXYStageDevice(), x, y)
        except Exception as e:
            self.context_error = log_error(e, "Microscope move to last mutant failed")
    
    def classify_as_wt(self):
        """
        Classifies the currently viewed worm as "Wild-Type" (WT).

        This function performs several actions:
        1. Updates the worm's label in the position manager.
        2. Updates the UI to show the new proportions of WT and mutant worms.
        3. Moves the segmented image of the worm from a prediction directory
        to a final "WT" directory.
        4. Updates a master dataset and a model performance tracking file
        with the new classification.
        """
        try:
            id = self.worms_position.get_id_worm_seen()
            self.worms_position.update_worm_label(id, 'Wild-Type')
            self.proportion_wt_label_ref.config(text=f"{int(100*(1-self.worms_position.get_mutant_proportion()))}%")
            self.proportion_mutant_label_ref.config(text=f"{int(100*(self.worms_position.get_mutant_proportion()))}%")
            
            if False:
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
                    big_dataset = Dataset_Manager()
                    big_dataset.load_images(compute=False, name_dataset="big_dataset")
                    big_dataset.update_label_by_filename(filename, "WT", new_filename)
                    
                    # update model_performance file
                    # Get variables
                    csv_path_best_model = Path(MODELS_DIR) / "best_model_tracking.csv"
                    df_best_model = pd.read_csv(csv_path_best_model)
                    best_row = df_best_model.loc[df_best_model['best_score'].idxmax()]
                    best_scaler = best_row['best_scaler_name']
                    best_model = best_row['best_model_name']
                    new_line = {
                        'date': [pd.Timestamp.now().strftime(DATE_FORMAT)],
                        'best_scaler_name': [best_scaler], 
                        'best_model_name': [best_model],
                        'label_predicted': [self.worms_position.get_worm_prediction(id)],
                        'label_true': ["WT"]
                    }
                    df_new_results = pd.DataFrame(new_line)
                    csv_path = Path(MODELS_DIR) / "model_performance.csv"
                    df_existing_results = pd.read_csv(csv_path)
                    df_combined_results = pd.concat([df_existing_results, df_new_results], ignore_index=True)
                    df_combined_results.to_csv(csv_path, index=False, mode='w')
                    
                else:
                    snap_img = self.snap_image()
                    mask = self.find_worm_segmentation(snap_img)
                    img = np.zeros_like(snap_img)
                    img[mask] = snap_img[mask]
                    cv2.imwrite(str(classified_path), img)
        except Exception as e:
            self.context_error = log_error(e, f"Classify as WT failed")
    
    def classify_as_mutant(self):
        """
        Classifies the currently viewed worm as "Mutant".

        This function performs the same actions as `classify_as_wt`, but for the
        "Mutant" class. It updates the worm's label, the UI, moves the segmented
        image to the "Mutant" directory, and updates the master dataset and
        model performance tracking file.
        """
        try:
            id = self.worms_position.get_id_worm_seen()
            self.worms_position.update_worm_label(id, 'Mutant')
            self.proportion_wt_label_ref.config(text=f"{int(100*(1-self.worms_position.get_mutant_proportion()))}%")
            self.proportion_mutant_label_ref.config(text=f"{int(100*(self.worms_position.get_mutant_proportion()))}%")
            
            if False:
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
                    big_dataset = Dataset_Manager()
                    big_dataset.load_images(compute=False, name_dataset="big_dataset")
                    big_dataset.update_label_by_filename(filename, "Mutant", new_filename)
                    
                    # update model_performance file
                    # Get variables
                    csv_path_best_model = Path(MODELS_DIR) / "best_model_tracking.csv"
                    df_best_model = pd.read_csv(csv_path_best_model)
                    best_row = df_best_model.loc[df_best_model['best_score'].idxmax()]
                    best_scaler = best_row['best_scaler_name']
                    best_model = best_row['best_model_name']
                    new_line = {
                        'date': [pd.Timestamp.now().strftime(DATE_FORMAT)],
                        'best_scaler_name': [best_scaler],
                        'best_model_name': [best_model],
                        'label_predicted': [self.worms_position.get_worm_prediction(id)],
                        'label_true': ["Mutant"]
                    }
                    df_new_results = pd.DataFrame(new_line)
                    csv_path = Path(MODELS_DIR) / "model_performance.csv"
                    df_existing_results = pd.read_csv(csv_path)
                    df_combined_results = pd.concat([df_existing_results, df_new_results], ignore_index=True)
                    df_combined_results.to_csv(csv_path, index=False, mode='w')
                    
                else:
                    snap_img = self.snap_image()
                    mask = self.find_worm_segmentation(snap_img)
                    img = np.zeros_like(snap_img)
                    img[mask] = snap_img[mask]
                    cv2.imwrite(str(classified_path), img)
        except Exception as e:
            self.context_error = log_error(e, f"Classify as mutant failed")
     
    def find_worm_segmentation(self, img, verbose=False):
        """
        Segments a worm from the background in an image using a YOLO model.

        This function takes an input image, normalizes it, and uses a pre-trained
        YOLO segmentation model to find a mask for the worm. It identifies the
        mask closest to the center of the image to ensure the correct worm is
        selected, and then applies this mask to the original image to isolate
        the worm.

        Args:
            img (np.ndarray): The input image (2D grayscale or 3D color).

        Returns:
            np.ndarray: The mask of the segmented worm as a booleana array.
        """
        try:
            model = self.segmentation_model
            image = img.copy()
            
            # Normalize image for YOLO
            #image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            
            def auto_contrast(image, percentile_low=0.5, percentile_high=99.5):
                img = image.astype(np.float32)
                vmin = np.percentile(img, percentile_low)
                vmax = np.percentile(img, percentile_high)
                if vmax <= vmin:
                    vmax = vmin + 1.0
                # Scale to 0-255 and clip
                img_scaled = np.clip((img - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
                return img_scaled

            # In find_worm_segmentation:
            image = auto_contrast(image)
            
            
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
            
            if verbose:
                # Show the original image
                plt.figure(figsize=(10,5))
                plt.subplot(1,2,1)
                plt.imshow(image, cmap='gray')
                plt.title("Original Image")
                plt.axis('off')

                # Show the mask
                plt.subplot(1,2,2)
                plt.imshow(mask_bool, cmap='gray')  # or cmap='Reds' to highlight
                plt.title("Detected Mask")
                plt.axis('off')

                plt.show()

            return mask_bool
        except Exception as e:
            self.context_error = log_error(e, f"Find worm segmentation failed")
            return None
    
    def analyse_worm(self):
        """
        Performs a prediction on the currently viewed worm using a trained model.

        This function is activated when the application is in "snap" mode. It
        first segments the worm from the image, saves the segmented image, and
        then uses a machine learning model to predict its class (e.g., WT or
        mutant). The prediction result is saved and displayed to the user.
        If the model fails, a default prediction is used.
        """
        VERBOSE = False
        # Step 0: Tell the user the analysis is starting
        self.prediction_label_2.configure(text=f"with a probability of : computing...")
        self.root.update() 
        
        # Step 1: Segment the image and save it
        img = self.snap_image(analysis_mode=True)
        mask = self.find_worm_segmentation(img, verbose=VERBOSE) 
        id = self.worms_position.get_id_worm_seen()
        unclassified_path = Path(DATA_DIR) / "Unclassified" / f"{id}.tif"
        imwrite(str(unclassified_path), img)
        self.prediction_label_2.configure(text=f"with a probability of : segmenting...")
        self.root.update() 

        # Step 2: Try to predict with model
        try:
            dataset = Dataset_Manager()
            _, _, _, self.enhanced_image = dataset.load_images(visualize=True, bool_mask=mask)
            dataset.set_features()
            self.prediction_label_2.configure(text=f"with a probability of : set features...")
            self.root.update() 

            model, _ = dataset.get_model(model_name = str(self.name_model.get()))
            pred = model.predict(dataset.get_features_selected()[0])[0]
            print(f"Model-derived prediction : {pred:.2f}")
            
            big_dataset = Dataset_Manager()
            big_dataset.load_images(compute=False, name_dataset="big_dataset")
            big_dataset.merge_with(dataset)

            # Step 3: Update prediction in worm database
            self.worms_position.update_worm_prediction(id, pred)
            self.prediction = str(int(100*pred))
            self.prediction_label_2.configure(text=f"with a probability of {self.prediction}%")
            self.root.update() 
        except Exception as e:
            self.context_error = log_error(e, f"Prediction failed")
            self.prediction_label_2.configure(text=f"with a probability of : doesn't succeed to make a prediction")
            self.root.update() 
            pred = -1
        
        # Step 4: Save image in the corresponding directory 
        directory = Path(DATA_DIR) / ("Mutant_prediction" if pred > 0.5 else "WT_prediction")
        if pred == -1:
            directory = Path(DATA_DIR) / "Error" 
        id = len(list(directory.glob("*")))
        classified_path = directory / f"{id}.tif" 
        shutil.move(str(unclassified_path), str(classified_path))
        
        
        # Step 5 : Display enhanced image after prediction
        self._show_enhanced_preview = True
        
        # ensure we are on the load_position page / that live label exists
        try:
            self.show_load_position_page()
            # give widgets a short time to exist, then display enhanced image
            try:
                # immediate attempt (if widgets ready)
                self.display_enhanced_image()
            except Exception:
                # schedule a short delay if widgets are still being created
                self.root.after(100, self.display_enhanced_image)
        except Exception:
            # fallback: try direct display
            try:
                self.display_enhanced_image()
            except Exception:
                pass
            
        self._show_enhanced_preview = True
        self.display_enhanced_image()
        self._show_enhanced_return_button()
                
    def start_live(self, switch_to_load_position=True):
        """
        Starts the live image acquisition loop.

        This function sets the `live_image` flag to True, switches the UI to the
        `load_position` page (optional), and begins the continuous `update_live_image` loop.
        """
        self._show_enhanced_preview = False
        self._hide_enhanced_return_button()
        self.live_image = True
        if switch_to_load_position:
            self.show_load_position_page()

        # ensure histogram window is open
        try:
            if not hasattr(self, "hist_canvas") or not self.hist_canvas.get_tk_widget().winfo_exists():
                self.open_contrast_histogram_window()
        except Exception:
            pass

        # cancel any previously scheduled loop to avoid duplicates
        try:
            if hasattr(self, "_live_after_id") and self._live_after_id:
                try:
                    self.root.after_cancel(self._live_after_id)
                except Exception:
                    pass
        except Exception:
            pass

        self._live_running = True
        self.update_live_image()
        
        # ensure histogram redraw immediately for live mode
        try:
            self.last_hist_update_time = 0
            self.update_image_and_histogram(live_mode=True)
        except Exception:
            pass
        
    def snap_image(self, analysis_mode = False):
        """
        Snaps a single image from the microscope and returns it as a numpy array (float32).
        """
        try:
            self.CORE.setExposure(EXPOSURE_TIME_ANALYSIS)
            self.CORE.snapImage()
            img = self.CORE.getImage()

            # Normalize returned type to numpy float32 (raw values)
            if isinstance(img, np.ndarray):
                arr = img.astype(np.float32)
            else:
                try:
                    pil = Image.fromarray(img) if not isinstance(img, Image.Image) else img
                    arr = np.array(pil, dtype=np.float32)
                except Exception as e:
                    arr = np.array(img, dtype=np.float32)

            return arr
        except Exception as e:
            self.context_error = log_error(e, f"Snap image failed")
            return None

    def snap_image_mode(self):
        """
        Snaps a single image from the microscope and displays it.

        This function stops the live image loop, captures a single image with a
        specific exposure time, and then displays it in the UI. It also opens
        a separate window for contrast and brightness adjustment.
        """
        self.live_image = False
        self._live_running = False
        self.clear_enhanced_preview()

        if hasattr(self, "_live_after_id") and self._live_after_id:
            self.root.after_cancel(self._live_after_id)
            self._live_after_id = None

        self.snap_img = self.snap_image()

        # store a raw copy for histogram / contrast logic
        if isinstance(self.snap_img, np.ndarray):
            self.original_snap_array = self.snap_img.copy().astype(np.float32)
        else:
            try:
                self.original_snap_array = np.array(self.snap_img, dtype=np.float32)
            except Exception as e:
                self.original_snap_array = np.zeros((256,256), dtype=np.float32)

        # Now show the page, which creates the display widgets
        self.show_load_position_page()

        # Then delay the display of the image so widgets have time to exist
        try:
            self.root.after(100, self.display_snap_image)
        except:
            pass
        try:
            if not hasattr(self, "hist_canvas") or not self.hist_canvas.get_tk_widget().winfo_exists():
                self.open_contrast_histogram_window()
        except Exception:
            pass
        
        # ensure histogram redraw immediately for snapshot
        try:
            # allow immediate histogram draw
            self.last_hist_update_time = 0
            # prefer to pass the snapshot explicitly
            self.update_image_and_histogram(img_array=getattr(self, "original_snap_array", None), live_mode=False)
        except Exception:
            pass
        
    def update_live_image(self):
        """
        Live update loop (safe). Keeps original behavior but:
        - saves raw frame into self.last_live_frame (float32 / original range),
        - if contrast controls exist, delegates display + throttled histogram to
            update_image_and_histogram(img_array=..., live_mode=True),
        - otherwise falls back to original 16->8bit conversion & display.
        Additionally: updates self.automatic_live_image_label (if present) with
        a square live preview centered in its container.
        """
        try:
            # stop quickly if requested or if root was destroyed
            if not getattr(self, "_live_running", False) or not self.root.winfo_exists():
                return

            # only try to update if at least one live-preview label exists
            live_label_ok = hasattr(self, "live_image_label") and self.live_image_label.winfo_exists()
            auto_label_ok = hasattr(self, "automatic_live_image_label") and self.automatic_live_image_label.winfo_exists()
            if not (live_label_ok or auto_label_ok):
                # no preview targets remain — stop the loop to avoid repeated errors
                self._live_running = False
                return
            
            # --- handle enhanced preview override: display enhanced_image instead of acquiring camera ---
            if getattr(self, "_show_enhanced_preview", False):
                try:
                    # Just display the enhanced image and skip camera acquisition
                    if hasattr(self, "live_image_label") and self.live_image_label.winfo_exists():
                        self.display_enhanced_image()

                    # also update automatic preview if present (optional)
                    if getattr(self, "automatic_live_image_label", None) and self.automatic_live_image_label.winfo_exists():
                        try:
                            # create small square preview from enhanced image
                            en_img = self.enhanced_image
                            en_pil = en_img if isinstance(en_img, Image.Image) else Image.fromarray(np.array(en_img))
                            aw = self.automatic_live_image_label.winfo_width()
                            ah = self.automatic_live_image_label.winfo_height()
                            size = int(min(max(aw, 0), max(ah, 0))) or None
                            if size:
                                mini = en_pil.resize((size, size), Image.Resampling.LANCZOS)
                                tk_auto = ImageTk.PhotoImage(mini)
                                self.automatic_live_image_label.image = tk_auto
                                self.automatic_live_image_label.config(image=tk_auto)
                        except Exception:
                            pass

                    # schedule next iteration (keep loop alive so user can toggle back)
                    try:
                        if getattr(self, "_live_running", False) and self.root.winfo_exists():
                            if hasattr(self, "_live_after_id"):
                                try:
                                    self.root.after_cancel(self._live_after_id)
                                except Exception:
                                    pass
                            self._live_after_id = self.root.after(200, self.update_live_image)
                            if not hasattr(self, "_after_ids"):
                                self._after_ids = []
                            self._after_ids.append(self._live_after_id)
                    except Exception:
                        pass

                    return  # skip the normal camera acquisition/display
                except Exception as e:
                    # If enhanced display fails, log and fall back to camera logic below
                    if self.context_error != "Update live image failed (enhanced preview)":
                        self.context_error = log_error(e, "Update live image failed (enhanced preview)")



            # Safe camera acquisition (guard for CORE missing/failing)
            image_data = None
            try:
                if self.CORE is None:
                    raise RuntimeError("CORE is None")
                # These calls may raise; catch below
                try:
                    exp = int(self.exposure_time.get())
                    self.CORE.setExposure(exp)
                    self.CORE.setProperty(NAME_CAMERA, "Binning", self.binning.get())
                except Exception:
                    pass
                self.CORE.snapImage()
                image_data = self.CORE.getImage()
            except Exception as e:
                # log, but keep running (don't crash the app)
                if self.context_error != "Update live image failed (camera)":
                    self.context_error = log_error(e, "Update live image failed (camera)")
                image_data = None

            # If we got image data, convert and display
            if image_data is not None:
                try:
                    # image_data might already be a numpy array or some other type.
                    # convert to numpy float32 preserving original dynamic range.
                    if isinstance(image_data, np.ndarray):
                        arr = image_data.astype(np.float32)  # keep original values (float32)
                    else:
                        # fallback: use PIL then to numpy
                        pil = Image.fromarray(image_data)
                        arr = np.array(pil, dtype=np.float32)

                    # store raw frame for histogram/contrast code
                    self.last_live_frame = arr.copy()

                    if self.current_page == "load_position" or self.current_page == "length_analysis":
                        # If contrast controls are present (user opened contrast window),
                        # delegate display + histogram drawing to update_image_and_histogram.
                        if getattr(self, "vmin_var", None) is not None or getattr(self, "hist_canvas", None) is not None:
                            try:
                                # prefer passing the raw array (not scaled to 8-bit)
                                if not getattr(self, "_contrast_slider_active", False):
                                    self.update_image_and_histogram(img_array=self.last_live_frame, live_mode=True)
                            except TypeError:
                                # If your update_image_and_histogram signature hasn't been changed
                                # to accept arguments, call without args and let it use self.last_live_frame
                                self.update_image_and_histogram()
                        else:
                            # 1) If length_analysis: use auto-scaling (min-max) to ensure visibility
                            if self.current_page == "length_analysis":
                                arr_16 = arr
                                min_val = arr_16.min() if arr_16.size else 0.0
                                max_val = arr_16.max() if arr_16.size else 0.0
                                denom = (max_val - min_val)
                                if denom == 0:
                                    denom = 1.0
                                arr_8bit = (((arr_16 - min_val) / denom) * 255.0).clip(0, 255).astype(np.uint8)
                            
                            # 2) Otherwise (e.g. load_position without manual contrast): use fixed absolute scaling
                            #    so user sees true dark/bright levels (exposure setting)
                            else:
                                arr_16 = arr
                                max_val = arr_16.max() if arr_16.size else 255.0
                                if max_val > 255:
                                    arr_8bit = (arr_16 / 256.0).clip(0, 255).astype(np.uint8)
                                else:
                                    # unlikely fallback if camera somehow returned 8-bit range
                                    min_val = arr_16.min() if arr_16.size else 0.0
                                    denom = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
                                    arr_8bit = (((arr_16 - min_val) / denom) * 255.0).clip(0, 255).astype(np.uint8)

                            image = Image.fromarray(arr_8bit, mode="L")

                            # resize to label size if available
                            try:
                                label_width = self.live_image_label.winfo_width()
                                label_height = self.live_image_label.winfo_height()
                                if label_width > 0 and label_height > 0:
                                    image = image.resize((label_width, label_height), Image.Resampling.LANCZOS)
                            except Exception:
                                pass

                            tk_image = ImageTk.PhotoImage(image)
                            # keep reference to avoid GC
                            self.live_image_label.image = tk_image
                            self.live_image_label.config(image=tk_image)

                    # ----------------------------
                    # Update the automatic live preview (square) if present
                    # ----------------------------
                    if self.current_page == "automatic_scan":
                        if hasattr(self, "automatic_live_image_label") and self.automatic_live_image_label.winfo_exists():
                            try:
                                # Use self.last_live_frame (float32) to create a 8-bit PIL image
                                arr_auto = getattr(self, "last_live_frame", None)
                                if arr_auto is not None and arr_auto.size:
                                    # Use percentile scaling (0.5% - 99.5%) to be robust against outliers and ensure good contrast
                                    # similar to auto_adjust_contrast logic
                                    vmin_a = np.percentile(arr_auto, 0.5)
                                    vmax_a = np.percentile(arr_auto, 99.5)
                                    
                                    # Safety check
                                    if vmax_a <= vmin_a:
                                        vmax_a = vmin_a + 1.0

                                    denom_a = (vmax_a - vmin_a)
                                    arr_auto_8 = (((arr_auto - vmin_a) / denom_a) * 255.0).clip(0, 255).astype(np.uint8)

                                    pil_auto = Image.fromarray(arr_auto_8.astype(np.uint8), mode="L")

                                    # determine square size from the placed label geometry
                                    try:
                                        aw = self.automatic_live_image_label.winfo_width()
                                        ah = self.automatic_live_image_label.winfo_height()
                                        size = int(min(max(aw, 0), max(ah, 0)))
                                    except Exception:
                                        size = None

                                    if size and size > 0:
                                        pil_auto = pil_auto.resize((size, size), Image.Resampling.LANCZOS)

                                    tk_auto = ImageTk.PhotoImage(pil_auto)
                                    # keep reference to avoid GC
                                    self.automatic_live_image_label.image = tk_auto
                                    self.automatic_live_image_label.config(image=tk_auto)
                            except Exception:
                                # don't break the loop for the automatic preview
                                pass

                except Exception as e:
                    # conversion/display error should not break the loop
                    if self.context_error != "Update live image failed":
                        self.context_error = log_error(e, "Update live image failed (display)")
        except Exception as e:
            # top-level safety net
            if self.context_error != "Update live image failed":
                self.context_error = log_error(e, "Update live image failed")
        finally:
            # schedule next iteration only if still running and root alive
            try:
                if getattr(self, "_live_running", False) and self.root.winfo_exists():
                    # cancel previous scheduled id to avoid stacking
                    try:
                        if hasattr(self, "_live_after_id"):
                            self.root.after_cancel(self._live_after_id)
                    except Exception:
                        pass

                    # adjust delay to tune CPU/fps (200 ms is what you had)
                    self._live_after_id = self.root.after(200, self.update_live_image)

                    # track it for global cleanup
                    if not hasattr(self, "_after_ids"):
                        self._after_ids = []
                    self._after_ids.append(self._live_after_id)
            except Exception:
                pass
    

    def open_contrast_histogram_window(self, parent=None):
        """
        Opens (or updates) a window with a histogram and contrast sliders.
        If parent is provided, embeds the histogram in that widget.
        Otherwise creates a Toplevel window.
        """
        try:
            # Decide which image to use
            if not getattr(self, "live_image", False):
                # Snapshot mode
                img_array = getattr(self, "original_snap_array", None)
                if img_array is None:
                    img_array = getattr(self, "snap_img", None)
            else:
                # Live mode
                img_array = getattr(self, "last_live_frame", None)

            # Fallback to a small black image if none available
            if img_array is None:
                img_array = np.zeros((256, 256), dtype=np.float32)

            # Ensure numpy array with float dtype for percentile/min/max calculations
            try:
                img_array = np.array(img_array, dtype=np.float32)
            except Exception:
                img_array = img_array.copy().astype(np.float32)

            # Compute sensible bounds
            img_min = float(np.min(img_array))
            img_max = float(np.max(img_array))
            if img_max == img_min:
                img_max = img_min + 1.0  # avoid identical bounds

            # Create or update vmin/vmax variables.
            # If user is currently dragging sliders (_contrast_slider_active), preserve their values.
            if not getattr(self, "vmin_var", None):
                self.vmin_var = tk.DoubleVar(value=img_min)
            else:
                if not getattr(self, "_contrast_slider_active", False):
                    self.vmin_var.set(img_min)

            if not getattr(self, "vmax_var", None):
                self.vmax_var = tk.DoubleVar(value=img_max)
            else:
                if not getattr(self, "_contrast_slider_active", False):
                    self.vmax_var.set(img_max)

            # Create window or use parent
            if parent:
                self.contrast_win = parent
            else:
                self.contrast_win = tk.Toplevel()
                self.contrast_win.title("Adjust Brightness / Contrast")
                self.contrast_win.geometry("+1100+450")
                self.contrast_win.protocol("WM_DELETE_WINDOW", lambda: self.close_histogram_window())

            # Clear existing content if re-using a Frame/Window that might have old stuff
            for widget in self.contrast_win.winfo_children():
                widget.destroy()

            def close_histogram_window():
                try:
                    if hasattr(self, "hist_fig"):
                        plt.close(self.hist_fig)
                    
                    if isinstance(self.contrast_win, tk.Toplevel):
                         self.contrast_win.destroy()
                         self.contrast_win = None
                    else:
                        # For embedded, we don't automatically destroy the parent frame here
                        pass
                except:
                    pass
                finally:
                    if isinstance(self.contrast_win, tk.Toplevel):
                        self.contrast_win = None
                    self.hist_canvas = None
                    self.hist_fig = None
                    self.hist_ax = None
            
            self.close_histogram_window = close_histogram_window

            # Histogram
            self.hist_fig, self.hist_ax = plt.subplots(figsize=(5, 2))
            
            # --- Styling: Match app theme ---
            bg_color = self.colors.theme["primary_background"]
            text_color = self.colors.theme["secondary_text"]
            
            self.hist_fig.patch.set_facecolor(bg_color)
            self.hist_ax.set_facecolor(bg_color)
            
            self.hist_ax.spines['bottom'].set_color(text_color)
            self.hist_ax.spines['top'].set_color(text_color) 
            self.hist_ax.spines['right'].set_color(text_color)
            self.hist_ax.spines['left'].set_color(text_color)
            
            self.hist_ax.tick_params(axis='x', colors=text_color)
            self.hist_ax.tick_params(axis='y', colors=text_color)
            self.hist_ax.yaxis.label.set_color(text_color)
            self.hist_ax.xaxis.label.set_color(text_color)
            self.hist_ax.title.set_color(text_color)
            # --------------------------------

            self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=self.contrast_win)
            self.hist_canvas.get_tk_widget().pack(pady=2)
            self.hist_canvas.get_tk_widget().configure(bg=bg_color) # Ensure canvas widget bg matches too

            # Sliders frame
            slider_frame = tk.Frame(self.contrast_win, bg=bg_color) # Add bg
            slider_frame.pack(pady=2)

            # Buttons (Auto / Full)
            button_frame = tk.Frame(self.contrast_win, bg=bg_color) # Add bg
            button_frame.pack(pady=2)

            auto_btn = tk.Button(button_frame, text="Auto", command=self.auto_adjust_contrast,
                                 highlightbackground=bg_color) # Try to blend button
            auto_btn.pack(side=tk.LEFT, padx=10)

            full_btn = tk.Button(button_frame, text="Full", command=self.full_range_contrast,
                                 highlightbackground=bg_color) # Try to blend button
            full_btn.pack(side=tk.LEFT, padx=10)

            # Labels and scales: set from_/to to the image's integer bounds
            vmin_label = tk.Label(slider_frame, text="vmin", bg=bg_color, fg=text_color) # Add colors
            vmin_label.grid(row=0, column=0, padx=5)

            from_val = int(np.floor(img_min))
            to_val = int(np.ceil(img_max))

            self.vmin_slider = tk.Scale(
                slider_frame, from_=from_val, to=to_val, variable=self.vmin_var,
                orient=tk.HORIZONTAL, length=400, resolution=1,
                command=lambda val: self.on_contrast_slider_change(),
                bg=bg_color, fg=text_color, highlightbackground=bg_color # Add colors
            )
            self.vmin_slider.grid(row=0, column=1, padx=5)

            vmax_label = tk.Label(slider_frame, text="vmax", bg=bg_color, fg=text_color) # Add colors
            vmax_label.grid(row=1, column=0, padx=5, pady=(0, 0))

            self.vmax_slider = tk.Scale(
                slider_frame, from_=from_val, to=to_val, variable=self.vmax_var,
                orient=tk.HORIZONTAL, length=400, resolution=1,
                command=lambda val: self.on_contrast_slider_change(),
                bg=bg_color, fg=text_color, highlightbackground=bg_color # Add colors
            )
            self.vmax_slider.grid(row=1, column=1, padx=5, pady=(0, 0))

            # Ensure the visual slider positions match the variable values
            try:
                self.vmin_slider.set(int(round(float(self.vmin_var.get()))))
                self.vmax_slider.set(int(round(float(self.vmax_var.get()))))
            except Exception:
                pass

            # Prevent overwriting user changes while they drag the slider
            def _on_slider_press(event):
                self._contrast_slider_active = True

            def _on_slider_release(event):
                self._contrast_slider_active = False
                try:
                    # choose the correct source image for update after release
                    if getattr(self, "live_image", False):
                        src = getattr(self, "last_live_frame", None)
                        self.last_hist_update_time = 0
                        self.update_image_and_histogram(img_array=src, live_mode=True)
                    else:
                        src = getattr(self, "original_snap_array", None) or getattr(self, "snap_img", None)
                        self.last_hist_update_time = 0
                        self.update_image_and_histogram(img_array=src, live_mode=False)
                except Exception:
                    pass

            # bind press/release to both sliders
            self.vmin_slider.bind("<ButtonPress-1>", _on_slider_press)
            self.vmin_slider.bind("<ButtonRelease-1>", _on_slider_release)
            self.vmax_slider.bind("<ButtonPress-1>", _on_slider_press)
            self.vmax_slider.bind("<ButtonRelease-1>", _on_slider_release)

            # Initial draw using the selected mode (pass the explicit img_array)
            self.update_image_and_histogram(img_array=img_array, live_mode=bool(getattr(self, "live_image", False)))

        except Exception as e:
            self.context_error = log_error(e, f"Open contrast histogram window failed")

    def _try_open_histogram(self):
        """Helper to open histogram window if not already open"""
        try:
            window_exists = False
            if hasattr(self, "contrast_win") and self.contrast_win is not None:
                try:
                    window_exists = self.contrast_win.winfo_exists()
                except:
                    window_exists = False
            
            if not window_exists:
                self.open_contrast_histogram_window()
        except Exception as e:
            self.context_error = log_error(e, f"Error when trying to open the histogram window")
                        
    def on_contrast_slider_change(self):
        """
        Called continuously when sliders move.
        Use live_mode according to current self.live_image so sliders affect the right image.
        """
        try:
            mode = bool(getattr(self, "live_image", False))
            self.update_image_and_histogram(live_mode=mode)
        except Exception as e:
            self.context_error = log_error(e, f"Slider change handler failed")

    def _maybe_update_slider_range(self, img_min, img_max, expand_ratio=0.05):
        """
        Ensure the sliders' from_/to cover the image min/max.
        We only expand range when user is NOT interacting with sliders.
        expand_ratio adds a little padding to avoid frequent small changes.
        """
        if getattr(self, "_contrast_slider_active", False):
            return  # user is interacting, do not auto-change

        if not getattr(self, "vmin_slider", None) or not getattr(self, "vmax_slider", None):
            return

        # Convert to ints for slider ranges
        img_min_i = int(np.floor(float(img_min)))
        img_max_i = int(np.ceil(float(img_max)))
        if img_max_i == img_min_i:
            img_max_i = img_min_i + 1

        # add padding
        pad = max(1, int((img_max_i - img_min_i) * expand_ratio))
        new_from = max(int(self._sensor_min_possible), img_min_i - pad)
        new_to   = min(int(self._sensor_max_possible), img_max_i + pad)

        # read current slider config (they share the same from/to in our UI)
        cur_from = int(float(self.vmin_slider.cget("from")))
        cur_to   = int(float(self.vmin_slider.cget("to")))

        # Only update if image min/max outside current bounds (avoid jitter)
        if img_min_i < cur_from or img_max_i > cur_to:
            # keep user values but clamp into new range
            vmin_val = int(float(self.vmin_var.get()))
            vmax_val = int(float(self.vmax_var.get()))

            # clamp
            vmin_val = max(new_from, min(vmin_val, new_to - 1))
            vmax_val = max(new_from + 1, min(vmax_val, new_to))

            # apply new range to both sliders (same from/to)
            self.vmin_slider.config(from_=new_from, to=new_to)
            self.vmax_slider.config(from_=new_from, to=new_to)

            # update variables (no sudden jump if within range; clamped otherwise)
            self.vmin_var.set(vmin_val)
            self.vmax_var.set(vmax_val)

    def update_image_and_histogram(self, img_array=None, live_mode=False):
        """
        Updates the displayed image and the contrast window's histogram.

        This function is called by the `vmin` and `vmax` sliders. It clips and
        rescales the `original_snap_array` based on the slider values, updates
        the image in the main UI, and redraws the histogram with vertical lines
        indicating the current `vmin` and `vmax`.
        Args:
            img_array: optional numpy array to use for histogram and display.
                    if None and live_mode True, uses self.last_live_frame.
            live_mode: if True behaves with throttled histogram update suitable for live mode.
        """
        if not hasattr(self, "last_hist_update_time"):
            self.last_hist_update_time = 0.0

        # Add lock to prevent simultaneous updates
        if getattr(self, '_image_update_lock', False):
            return
        
        self._image_update_lock = True
        try:
            # Choose source image
            if img_array is None:
                if live_mode:
                    img_array = getattr(self, "last_live_frame", None)
                else:
                    img_array = getattr(self, "original_snap_array", None)

            if img_array is None:
                return

            # get current vmin/vmax (if not set, compute from array)
            if getattr(self, "vmin_var", None) is None or getattr(self, "vmax_var", None) is None:
                vmin_val = float(np.min(img_array))
                vmax_val = float(np.max(img_array))
                self.vmin_var = tk.DoubleVar(value=vmin_val)
                self.vmax_var = tk.DoubleVar(value=vmax_val)
            else:
                vmin_val = float(self.vmin_var.get())
                vmax_val = float(self.vmax_var.get())

            # Safety: ensure vmin < vmax
            if vmax_val <= vmin_val:
                vmax_val = vmin_val + 1.0
                self.vmax_var.set(vmax_val)
                
            img_min = float(np.min(img_array))
            img_max = float(np.max(img_array))
            # Update slider range if needed
            self._maybe_update_slider_range(img_min, img_max)

            # Clip and scale to 0..255
            clipped = np.clip(img_array, vmin_val, vmax_val)
            scaled = ((clipped - vmin_val) / (vmax_val - vmin_val + 1e-8) * 255.0).astype(np.uint8)

            # Convert to PIL and resize to label
            image = Image.fromarray(scaled)
            try:
                self.live_image_label.update_idletasks()
                label_width = self.live_image_label.winfo_width()
                label_height = self.live_image_label.winfo_height()
                if label_width > 0 and label_height > 0:
                    resample_method = Image.Resampling.BILINEAR if live_mode else Image.Resampling.LANCZOS
                    image = image.resize((label_width, label_height), resample_method)
            except Exception:
                pass

            tk_image = ImageTk.PhotoImage(image)
            self.live_image_label.configure(image=tk_image)
            self.live_image_label.image = tk_image

            # Throttle histogram redraws when in live mode
            now = time.time()
            if not live_mode or (now - getattr(self, "last_hist_update_time", 0) >= self.hist_update_interval):
                # update histogram using full-range raw image data (not scaled)
                self.hist_ax.clear()
                # use numpy histogram for speed
                try:
                    self.hist_ax.hist(img_array.ravel(), bins=256, alpha=0.8)
                except Exception:
                    # fall back when array dtype unexpected
                    self.hist_ax.hist(img_array.flatten(), bins=256, alpha=0.8)
                self.hist_ax.axvline(vmin_val, color='red', linestyle='--', linewidth=1.5, label='vmin')
                self.hist_ax.axvline(vmax_val, color='blue', linestyle='--', linewidth=1.5, label='vmax')
                self.hist_ax.set_title("Pixel Intensity Histogram", color=self.colors.theme["primary_text"])
                try:
                    self.hist_ax.set_xlim(np.min(img_array), np.max(img_array))
                except Exception:
                    pass
                self.hist_ax.legend()
                if getattr(self, "hist_canvas", None):
                    self.hist_canvas.draw_idle()
                self.last_hist_update_time = now
        except Exception as e:
            if self.context_error != "Update image histogram failed":
                self.context_error = log_error(e, f"Update image histogram failed")
        finally:
            self._image_update_lock = False

    def auto_adjust_contrast(self, percentile_low=0.5, percentile_high=99.5):
        """
        Compute percentile-based vmin/vmax, update sliders and refresh display.
        """
        try:
            # pick image depending on current live flag
            img = self._get_contrast_image(live_mode=bool(getattr(self, "live_image", False)))
            if img is None:
                return

            img = np.array(img, dtype=np.float32)  # ensure numpy float
            if img.size == 0:
                return

            v_auto_min = float(np.percentile(img.ravel(), percentile_low))
            v_auto_max = float(np.percentile(img.ravel(), percentile_high))
            if v_auto_max <= v_auto_min:
                v_auto_max = v_auto_min + 1.0

            # ensure slider ranges contain the new values (expand if needed)
            try:
                # compute image min/max for slider bounds update
                img_min, img_max = float(np.min(img)), float(np.max(img))
                # Try to update the sliders' from_/to so widget accepts the set values
                if getattr(self, "vmin_slider", None) and getattr(self, "vmax_slider", None):
                    new_from = int(np.floor(img_min))
                    new_to   = int(np.ceil(img_max))
                    # prevent degenerate ranges
                    if new_to == new_from:
                        new_to = new_from + 1
                    self.vmin_slider.config(from_=new_from, to=new_to)
                    self.vmax_slider.config(from_=new_from, to=new_to)
            except Exception as e:
                print("[auto_adjust] failed to adjust slider bounds:", e)

            # Set variables and force the Scale widgets to move
            try:
                if getattr(self, "vmin_var", None):
                    self.vmin_var.set(v_auto_min)
                if getattr(self, "vmax_var", None):
                    self.vmax_var.set(v_auto_max)
                if getattr(self, "vmin_slider", None):
                    self.vmin_slider.set(int(round(v_auto_min)))
                if getattr(self, "vmax_slider", None):
                    self.vmax_slider.set(int(round(v_auto_max)))
            except Exception as e:
                print("[auto_adjust] failed to set sliders/vars:", e)

            # Force update/hist redraw (use non-throttled update)
            try:
                self.last_hist_update_time = 0
                self.update_image_and_histogram(img_array=img, live_mode=False)
            except Exception as e:
                print("[auto_adjust] update_image_and_histogram failed:", e)

        except Exception as e:
            print("[auto_adjust] unexpected error:", e)

    def full_range_contrast(self):
        """
        Set vmin/vmax to the full image range and update display.
        """
        try:
            img = self._get_contrast_image(live_mode=bool(getattr(self, "live_image", False)))
            if img is None:
                return

            img = np.array(img, dtype=np.float32)
            if img.size == 0:
                return

            v_full_min = float(np.min(img))
            v_full_max = float(np.max(img))
            if v_full_max == v_full_min:
                v_full_max = v_full_min + 1.0

            # Update slider bounds first so values fit
            try:
                new_from = int(np.floor(v_full_min))
                new_to   = int(np.ceil(v_full_max))
                if new_to == new_from:
                    new_to = new_from + 1
                if getattr(self, "vmin_slider", None) and getattr(self, "vmax_slider", None):
                    self.vmin_slider.config(from_=new_from, to=new_to)
                    self.vmax_slider.config(from_=new_from, to=new_to)
            except Exception as e:
                print("[full_range] failed to set slider bounds:", e)

            # Set the vars and force the Scale thumbs to move
            try:
                if getattr(self, "vmin_var", None):
                    self.vmin_var.set(v_full_min)
                if getattr(self, "vmax_var", None):
                    self.vmax_var.set(v_full_max)
                if getattr(self, "vmin_slider", None):
                    self.vmin_slider.set(int(round(v_full_min)))
                if getattr(self, "vmax_slider", None):
                    self.vmax_slider.set(int(round(v_full_max)))
            except Exception as e:
                print("[full_range] failed to set sliders/vars:", e)

            # Force update/hist redraw
            try:
                self.last_hist_update_time = 0
                self.update_image_and_histogram(img_array=img, live_mode=False)
            except Exception as e:
                print("[full_range] update_image_and_histogram failed:", e)


        except Exception as e:
            print("[full_range] unexpected error:", e)
        
    def display_snap_image(self):
        """
        Displays the most recently snapped image.

        This function is called after a short delay following `snap_image`. It
        takes the snapped image, normalizes its pixel intensity range, converts
        it to a Tkinter-compatible format, and displays it in the `live_image_label`.
        """
        try:
            if isinstance(self.snap_img, np.ndarray):
                # store raw values (float32)
                self.original_snap_array = self.snap_img.copy().astype(np.float32)

                # produce displayable normalized image
                img = self.snap_img.copy().astype(np.float32)
                mn, mx = img.min(), img.max()
                if mx == mn:
                    img = np.zeros_like(img, dtype=np.uint8)
                else:
                    img = (img - mn) / (mx - mn + 1e-12) * 255.0
                    img = img.astype(np.uint8)

                image = Image.fromarray(img)
                label_width = self.live_image_label.winfo_width()
                label_height = self.live_image_label.winfo_height()
                if label_width > 0 and label_height > 0:
                    image = image.resize((label_width, label_height), Image.Resampling.LANCZOS)

                tk_image = ImageTk.PhotoImage(image)
                self.live_image_label.image = tk_image
                self.live_image_label.config(image=tk_image)
        except Exception as e:
            self.context_error = log_error(e, f"Display snap image failed")

        # Force histogram window to update using snapshot as source
        try:
            if getattr(self, "hist_canvas", None) is not None:
                self.update_image_and_histogram(img_array=getattr(self, "original_snap_array", None), live_mode=False)
        except Exception as e:
            print("[display_snap] histogram update failed:", e)
     
    def save_snap_image(self):
        """
        Saves the currently displayed snapped image to a user-specified directory.

        This function only operates when not in live mode. It displays a "Saved"
        message, saves the `snap_img` to a file with a timestamped filename
        inside the user's chosen directory, and then removes the "Saved" message
        after a short delay.
        """
        try:
            if self.live_image == False or self.save_in_live_mode == True: 
                self.save_button_label_ref.configure(text="Saved")
                self.root.update_idletasks()
                
                CURRENT_DATE = datetime.datetime.now().strftime(DATE_FORMAT) 
                filename = f"{CURRENT_DATE}.tif"
                desktop_path = Path.home() / "Desktop"
                user_directory = desktop_path / str(self.user_directory.get())
                path = user_directory / filename
                if not user_directory.exists():
                    user_directory.mkdir(parents=True, exist_ok=True)
                
                img_to_save = getattr(self, "original_snap_array", None)
                if img_to_save is None:
                    print("Is None")
                    img_to_save = self.snap_img

                img_to_save = np.clip(img_to_save, 0, 65535).astype(np.uint16)

                imwrite(str(path), img_to_save) 
                self.save_in_live_mode = False
                
                self.root.after(2000, lambda: self.save_button_label_ref.configure(text=""))
            else:
                self.original_snap_array = self.snap_image()
                self.snap_img = self.original_snap_array
                self.save_in_live_mode = True
                self.save_snap_image()
                
        except Exception as e:
            self.context_error = log_error(e, f"Save snap image failed")
          
    def _get_contrast_image(self, live_mode):
        """
        Return the numpy array to use for histogram/contrast depending on mode.
        Prefer original_snap_array for snap mode and last_live_frame for live.
        """
        if live_mode:
            return getattr(self, "last_live_frame", None)
        
        img_1 = getattr(self, "original_snap_array", None)
        if img_1 is not None:
            return img_1
        
        img_2 = getattr(self, "snap_img", None)
        if img_2 is not None:
            return img_2

        return None
    
    def display_enhanced_image(self):
        """
        Display self.enhanced_image in the main live preview label (self.live_image_label).
        Handles PIL Image or numpy array inputs and resizes to the label.
        """
        try:
            if getattr(self, "enhanced_image", None) is None:
                return

            # Convert to PIL if needed
            img = self.enhanced_image
            if not isinstance(img, Image.Image):
                try:
                    img = Image.fromarray(np.array(img))
                except Exception:
                    # fallback: try to coerce to uint8 first
                    arr = np.array(img, dtype=np.uint8)
                    img = Image.fromarray(arr)

            # Resize to label geometry if available
            try:
                if hasattr(self, "live_image_label") and self.live_image_label.winfo_exists():
                    w = self.live_image_label.winfo_width()
                    h = self.live_image_label.winfo_height()
                    if w > 0 and h > 0:
                        img = img.resize((w, h), Image.Resampling.LANCZOS)
            except Exception:
                pass

            tk_img = ImageTk.PhotoImage(img)
            # keep reference to avoid GC
            self.live_image_label.image = tk_img
            self.live_image_label.config(image=tk_img)
        except Exception as e:
            # Use your existing logging helper
            self.context_error = log_error(e, "Display enhanced image failed")

    def clear_enhanced_preview(self):
        """
        Disable enhanced preview and clear enhanced_image reference if desired.
        """
        self._show_enhanced_preview = False
        self._hide_enhanced_return_button() 

    def _create_enhanced_return_button(self):
        """
        Create the 'Return to Live' button if it doesn't exist.
        We don't pack it here — packing is handled by show/hide functions.
        """
        if hasattr(self, "enhanced_return_live_button") and self.enhanced_return_live_button.winfo_exists():
            return

        # parent: try to put it inside the bottom buttons container if exists, else inside live container
        parent = getattr(self, "left_live_analysis_container_ref", None) or getattr(self, "live_analysis_container_ref", None) or self.main_content

        # create a small frame for proper placement if needed
        try:
            self._enhanced_button_container = tk.Frame(parent, bg=self.colors.theme["primary_background"])
            # place it at bottom-right of the live container using place() or pack/grid depending on layout
            # here we will use place to overlay it on the live image container
            # but if you'd rather place it in the button row, insert it there instead.
        except Exception:
            self._enhanced_button_container = parent

        self.enhanced_return_live_button = self.create_rounded_button(
            parent=self._enhanced_button_container,
            text="Return to Live",
            icon=None,
            icon_hover=None,
            command=lambda: self._on_enhanced_return_clicked(),
            bg_color=self.colors.theme["tertiary_background"],
            text_color=self.colors.theme["primary_text"],
            hover_color=self.colors.theme["secondary_background"],
            font=(self.font, 10),
            width_pixels=160,
            height_pixels=36,
            corner_radius=12,
            side=tk.TOP,
            pady=6,
            padx_text=0,
            border_width=1,
            border_color=self.colors.theme["stroke_button"]
        )

    def _show_enhanced_return_button(self):
        """Display the return button near the live image (or in the button row)."""
        try:
            self._create_enhanced_return_button()
            # If we created a special container (overlay), place it at bottom-right
            if hasattr(self, "_enhanced_button_container") and self._enhanced_button_container is not None and getattr(self, "_enhanced_button_container", None) is not self.main_content:
                # Put container over the live area (adjust offsets as needed)
                try:
                    # place inside the live analysis container relative coordinates
                    self._enhanced_button_container.place(in_=self.live_analysis_container_ref, relx=0.98, rely=0.98, anchor="se")
                    # pack the button inside the container (create_rounded_button already packed by your helper)
                except Exception:
                    # fallback: pack under the live preview
                    self._enhanced_button_container.pack(side=tk.BOTTOM, anchor="e", padx=8, pady=8)
            else:
                # fallback: pack into the bottom buttons area if exists
                try:
                    self._enhanced_button_container.pack(side=tk.BOTTOM, anchor="e", padx=8, pady=8)
                except Exception:
                    pass

            # ensure the widget is visible
            try:
                self.enhanced_return_live_button.lift()
                self.enhanced_return_live_button.update_idletasks()
            except Exception:
                pass
        except Exception as e:
            # non-fatal
            print("Could not show enhanced return button:", e)

    def _hide_enhanced_return_button(self):
        """Hide/remove the return-to-live button."""
        try:
            if hasattr(self, "enhanced_return_live_button") and self.enhanced_return_live_button.winfo_exists():
                try:
                    self.enhanced_return_live_button.pack_forget()
                except Exception:
                    pass
            if hasattr(self, "_enhanced_button_container") and getattr(self, "_enhanced_button_container", None) is not None:
                try:
                    self._enhanced_button_container.place_forget()
                except Exception:
                    try:
                        self._enhanced_button_container.pack_forget()
                    except Exception:
                        pass
        except Exception:
            pass

    def _on_enhanced_return_clicked(self):
        """
        Called when user clicks the 'Return to Live' button.
        Restores normal live acquisition and hides the button.
        """
        try:
            # Disable the enhanced preview
            self._show_enhanced_preview = False

            # hide the button
            self._hide_enhanced_return_button()

            # ensure live flags are correct
            self.live_image = True
            if not getattr(self, "_live_running", False):
                self._live_running = True

            # Start or resume the live loop in a safe way
            try:
                # cancel previous if present
                if hasattr(self, "_live_after_id") and self._live_after_id:
                    try:
                        self.root.after_cancel(self._live_after_id)
                    except Exception:
                        pass
                # call update_live_image to immediately refresh
                self.update_live_image()
            except Exception:
                # fallback: call start_live() which does full setup
                try:
                    self.start_live()
                except Exception:
                    pass
        except Exception as e:
            self.context_error = log_error(e, "_on_enhanced_return_clicked failed")

    # --- Pages ---  
    def show_assist_acquisition_page(self): # UNUSED and DEPRECATED
        """
        Constructs the UI for the assisted worm acquisition page.

        This page is designed to help users manually find and save the positions
        of worms. It features a live image feed, a button to save the current
        position, and a small map to visualize the saved positions. Users can
        then proceed to the analysis stage once they have acquired all their
        worm positions.
        """
        # Clear previous widgets
        for widget in self.main_content.winfo_children():
            widget.destroy()
          
        # Disable some paramaters buttons   
        self.update_parameter_widgets_state(disabled_widgets=["scan_shape"]) 
        self.refresh_parameters_interface()
        self.update_parameter_widgets_state(disabled_widgets=["scan_shape"])

        # Configure grid layout for main_content
        self.main_content.grid_columnconfigure(0, weight=60)
        self.main_content.grid_columnconfigure(1, weight=30)
        self.main_content.grid_rowconfigure(0, weight=1)

        # ----- LEFT CONTAINER -----
        left_live_assist_container = tk.Frame(self.main_content, bg=self.colors.theme["primary_background"]) 
        left_live_assist_container.grid(row=0, column=0, sticky="nsew", padx=(0, 10))  
        self.left_live_assist_container_ref = left_live_assist_container

        # Use grid in left container to stack content
        left_live_assist_container.grid_rowconfigure(0, weight=1)  # for live_assist_container
        left_live_assist_container.grid_rowconfigure(1, weight=0)  # for bottom_assist_container
        left_live_assist_container.grid_columnconfigure(0, weight=1)

        # Top: Live assist square
        live_assist_container = tk.Frame(left_live_assist_container, bg=self.colors.theme["secondary_background"], relief=tk.RAISED, bd=1)
        live_assist_container.grid(row=0, column=0, sticky="nsew")
        self.live_assist_container_ref = live_assist_container

        # Bind resize for square behavior
        self.left_live_assist_container_ref.bind("<Configure>", self.resize_live_image)
        
        # Placeholder for live image
        if not self.live_image_label.winfo_exists():
            self.live_image_label = tk.Label(live_assist_container, bg="black")
            self.live_image_label.pack(expand=True, fill=tk.BOTH)

        # Bottom: Buttons + label
        bottom_assist_container = tk.Frame(left_live_assist_container, bg=self.colors.theme["primary_background"])
        bottom_assist_container.grid(row=1, column=0, sticky="ew", pady=(10, 10))
        
        self.create_rounded_button(
            parent=bottom_assist_container,
            text="",
            icon=self.play_icon,
            icon_hover=self.play_icon_hover,
            command=lambda: self.switch_page("load_position"),
            bg_color=self.colors.theme["primary_background"],
            text_color=self.colors.theme["primary_text"],
            hover_color=self.colors.theme["secondary_background"],
            font=(self.font, 16),
            width_pixels=200,
            height_pixels=60,
            corner_radius=20,
            side=tk.TOP,
            pady=5,
            padx_text=-10,
            border_width=2,
            border_color=self.colors.theme["stroke_button"]
        )

        # Info label with tooltip
        launch_label_assist_container = tk.Frame(bottom_assist_container, bg=self.colors.theme["primary_background"])
        launch_label_assist_container.pack()

        tk.Label(
            launch_label_assist_container, text="Start analysis",
            bg=self.colors.theme["primary_background"], fg=self.colors.theme["tertiary_text"],
            font=(self.font, 10)
        ).pack(side=tk.LEFT)

        tk.Label(
            launch_label_assist_container, image=self.info_icon,
            bg=self.colors.theme["primary_background"]
        ).pack(side=tk.LEFT, padx=(5, 0))

        Tooltip(launch_label_assist_container, "Be sure to use the L camera.", posx=160, posy=-60)

        # ----- RIGHT CONTAINER -----
        right_map_assist_container = tk.Frame(self.main_content, bg=self.colors.theme["primary_background"])
        right_map_assist_container.grid(row=0, column=1, sticky="nsew", padx=(0, 20))
        self.right_map_assist_container_ref = right_map_assist_container

        # Make right container expand vertically
        right_map_assist_container.grid_rowconfigure(0, weight=1)
        right_map_assist_container.grid_columnconfigure(0, weight=1)

        # Container for button + text
        top_button_assist_container = tk.Frame(right_map_assist_container, bg=self.colors.theme["primary_background"])
        top_button_assist_container.pack(pady=(70, 0))  # adjust padding as needed

        # Button at the top
        self.create_rounded_button(
            parent=top_button_assist_container,
            text="",
            icon=self.plus_icon,
            icon_hover=self.plus_icon_hover,
            command=lambda: self.add_worm_assist_acquisition,
            bg_color=self.colors.theme["secondary_background"],
            text_color=self.colors.theme["primary_text"],
            hover_color=self.colors.theme["tertiary_background"],
            font=(self.font, 14),
            width_pixels=150,
            height_pixels=120,
            corner_radius=20,
            side=tk.TOP,
            pady=5,
            padx_text=-5,
            border_width=0
        )

        # Two lines of text under the button
        tk.Label(
            top_button_assist_container,
            text="Save position",
            bg=self.colors.theme["primary_background"],
            fg=self.colors.theme["tertiary_text"],
            font=(self.font, 10)
        ).pack()

        tk.Label(
            top_button_assist_container,
            text="(you can use the press bar)",
            bg=self.colors.theme["primary_background"],
            fg=self.colors.theme["tertiary_text"],
            font=(self.font, 7)
        ).pack()

        # Bottom: Black square (map)
        map_assist_container = tk.Frame(right_map_assist_container, bg="black")
        map_assist_container.place(x=0, y=0, width=0, height=0)
        self.map_assist_containter_ref = map_assist_container
        self.right_map_assist_container_ref.bind("<Configure>", self.resize_map_assist)
        
        # Update the live image
        if self.live_image:
            self._live_running = True
            self.update_live_image()
    
    def show_automatic_scan_page(self):
        """
        Constructs the UI for the automatic scanning page.

        This function sets up a user interface for launching an automated
        microscope scan. It creates a main content area for displaying
        the scan, a status label to provide feedback to the user during the
        scan process, and a launch button. It also configures the layout
        to be responsive, ensuring the content area resizes correctly when
        the window is resized.
        """
        try:
            x, y = self.CORE.getXYPosition()
            if hasattr(self, "init_pos_x"):
                if x != self.init_pos_x or y != self.init_pos_y:
                    self.CORE.setXYPosition(self.CORE.getXYStageDevice(), self.init_pos_x, self.init_pos_y)
        except Exception as e:
            self.context_error = log_error(e, "Go to start position failed")
        
        # Clear previous widgets if needed
        for widget in self.main_content.winfo_children():
            widget.destroy()
            
        # Disable some paramaters buttons
        self.update_parameter_widgets_state(disabled_widgets=[])  # Everything enabled

        # Middle container that will hold the content_area and expand to max space
        middle_container = tk.Frame(self.main_content, bg=self.colors.theme["primary_background"])
        middle_container.pack(fill=tk.BOTH, expand=True)
        self.middle_container_ref = middle_container

        # Content area inside the middle container
        content_area = tk.Frame(middle_container, bg=self.colors.theme["secondary_background"], relief=tk.RAISED, bd=1)
        content_area.place(x=0, y=0, width=0, height=0)  # Temporary, real size set later
        self.content_area_ref = content_area
        
        self.automatic_image_wrapper = tk.Frame(content_area, bg=self.colors.theme["secondary_background"])
        self.automatic_image_wrapper.pack(expand=True, fill=tk.BOTH, padx=12, pady=12)

        # Bind configure so that when the wrapper resizes we keep a centered square
        self.automatic_image_wrapper.bind("<Configure>", self.resize_automatic_live_image)
           
        # The label that will display the live image. We use place() so we can
        # set exact x,y,width,height to maintain a square.
        if not hasattr(self, "automatic_live_image_label") or not self.automatic_live_image_label.winfo_exists():
            self.automatic_live_image_label = tk.Label(self.automatic_image_wrapper, bg="black")
            # start with small zero geometry; resize handler will place it correctly
            self.automatic_live_image_label.place(x=0, y=0, width=0, height=0)
            # reuse same click handler you already have for the live image page
            try:
                self.automatic_live_image_label.bind("<Button-1>", self.on_live_image_click)
            except Exception:
                pass        
            
        # ensure the live loop is running so the automatic preview is updated
        # prefer calling your start_live() function if it handles camera startup, otherwise:
        try:
            if not getattr(self, "_live_running", False):
                # if you have a start_live() that does extra setup, call it instead:
                # self.start_live()
                self._live_running = True
                self.update_live_image()
        except Exception:
            pass

        # Bottom section with launch button
        bottom_frame = tk.Frame(self.main_content, bg=self.colors.theme["primary_background"])
        bottom_frame.pack(fill=tk.X, pady=(20,5))
        
        # Status message for scan steps
        self.scan_status_label = tk.Label(
            bottom_frame,
            text="",
            fg=self.colors.theme["secondary_text"],
            bg=self.colors.theme["primary_background"],
            font=(self.font, 10),
            relief=tk.SOLID,
            border=0,
            padx=0,
            pady=0
        )
        self.scan_status_label.pack(pady=(10, 0))

        # Launch scan button
        self.create_rounded_button(
            parent=bottom_frame,
            text="",
            icon=self.play_icon,
            icon_hover=self.play_icon_hover,
            command=lambda: self.launch_scan(),
            bg_color=self.colors.theme["primary_background"],
            text_color=self.colors.theme["primary_text"],
            hover_color=self.colors.theme["secondary_background"],
            font=(self.font, self.screen_height // 60), # 16
            width_pixels=self.screen_height // 5, # old 200, new 192
            height_pixels=self.screen_height // 16, # 60
            corner_radius=self.screen_height // 48, # 20
            side=tk.TOP,
            pady=5,
            padx_text=-10,
            border_width=2,
            border_color=self.colors.theme["stroke_button"]
        )
        
        # Container to hold label + info icon
        launch_label_frame = tk.Frame(bottom_frame, bg=self.colors.theme["primary_background"])
        launch_label_frame.pack()

        # Text label
        title_launch_scan = tk.Label(
            launch_label_frame, text="Launch scan",
            bg=self.colors.theme["primary_background"], fg=self.colors.theme["tertiary_text"],
            font=(self.font, self.screen_height // 96) # 10
        )
        title_launch_scan.pack(side=tk.LEFT)

        # Info icon
        info_label = tk.Label(
            launch_label_frame, image=self.info_icon,
            bg=self.colors.theme["primary_background"]
        )
        info_label.pack(side=tk.LEFT, padx=(5, 0))  # small gap between text and icon

        # Tooltip on hover
        if MICROSCOPE == "Nikon":
            Tooltip(info_label, "Be sure to have the objective in the lower right corner and to use the L camera, and to focus the microscope.", posx=70, posy=-70)
        elif MICROSCOPE == "Macrozoom":
            Tooltip(info_label, "Set the objective to 2x and adjust the zoom to the 3rd detent (click-stop) before focusing the image.", posx=70, posy=-70)

        # Trigger resizing after layout completes with error handling
        try:
            if hasattr(self, 'middle_container_ref') and self.middle_container_ref.winfo_exists():
                self.middle_container_ref.bind("<Configure>", self.resize_scan_content_area)
        except:
            pass
    
    def show_result_scan_page(self):
        """
        Constructs the UI for the scan results page.

        This page displays the stitched image from a completed scan, showing
        bounding boxes around detected worms. It provides controls for users
        to manually add or remove worms from the results and to initiate the
        next step, which is the analysis of each worm. All scan-related
        parameters are disabled on this page.
        """
        # Clear previous widgets if needed
        for widget in self.main_content.winfo_children():
            widget.destroy()
            
        # Disable some paramaters buttons 
        self.update_parameter_widgets_state(disabled_widgets=["exposure_time","binning","shutter","dual_view","display_mode","scan_objective","fluo_objective","scan_shape", "model_name", "fov_size_um"]) 
        self.refresh_parameters_interface()
        self.update_parameter_widgets_state(disabled_widgets=["exposure_time","binning","shutter","dual_view","display_mode","scan_objective","fluo_objective","scan_shape", "model_name", "fov_size_um"])
        
        # Middle container that will hold the content_area and expand to max space
        middle_result_container = tk.Frame(self.main_content, bg=self.colors.theme["primary_background"])
        middle_result_container.pack(fill=tk.BOTH, expand=True)
        self.middle_result_container_ref = middle_result_container

        # Content area inside the middle container
        content_area_result_container = tk.Frame(middle_result_container, bg=self.colors.theme["secondary_background"], relief=tk.RAISED, bd=1)
        content_area_result_container.place(x=0, y=0, width=0, height=0)  # Temporary, real size set later
        self.content_area_result_container_ref = content_area_result_container        

        # Bottom section with launch button
        bottom_frame_result_container = tk.Frame(self.main_content, bg=self.colors.theme["primary_background"])
        bottom_frame_result_container.pack(fill=tk.X, pady=(20,5))
        
        # Row container for aligned buttons
        button_row_frame = tk.Frame(bottom_frame_result_container, bg=self.colors.theme["primary_background"])
        button_row_frame.pack(pady=5, anchor="center")  
        
        # Choose which button appears "selected"
        add_bg = self.colors.theme["tertiary_background"] if self.worm_scan_result_mode == "add" else self.colors.theme["primary_background"]
        remove_bg = self.colors.theme["tertiary_background"] if self.worm_scan_result_mode == "delete" else self.colors.theme["primary_background"]
        move_bg = self.colors.theme["tertiary_background"] if self.worm_scan_result_mode == "move" else self.colors.theme["primary_background"]
        add_icon = self.add_worm_icon_hover if self.worm_scan_result_mode == "add" else self.add_worm_icon
        remove_icon = self.remove_worm_icon_hover if self.worm_scan_result_mode == "delete" else self.remove_worm_icon
        move_icon = self.move_worm_icon_hover if self.worm_scan_result_mode == "move" else self.move_worm_icon
        
        # -- Add worm button --
        add_button_frame = tk.Frame(button_row_frame, bg=self.colors.theme["primary_background"])
        add_button_frame.pack(side=tk.LEFT, pady=5, padx=30)

        self.create_rounded_button(
            parent=add_button_frame,
            text="",
            icon=add_icon,
            icon_hover=self.add_worm_icon_hover,
            command=lambda: self.toggle_mode_worm_scan_result(mode="add"),
            bg_color=add_bg,
            text_color=self.colors.theme["primary_text"],
            hover_color=self.colors.theme["tertiary_background"],
            font=(self.font, self.screen_height // 60), # 16
            width_pixels=self.screen_height // 10, # old 100, new 96
            height_pixels= self.screen_height // 16, # 60
            corner_radius=self.screen_height // 48, # 20
            side=tk.TOP,
            pady=0,
            padx=0,
            padx_text=-6,
            border_width=2,
            border_color=self.colors.theme["stroke_button"]
        )

        add_info_icon = tk.Label(add_button_frame, image=self.info_icon, bg=self.colors.theme["primary_background"])
        add_info_icon.pack(side=tk.TOP, pady=(4, 0))  # slight spacing above icon
        Tooltip(add_info_icon, "Click on the image to add a new worm position", title="Info", theme="info", posx=70, posy=-80)
        
        # -- Move worm button --
        move_button_frame = tk.Frame(button_row_frame, bg=self.colors.theme["primary_background"])
        move_button_frame.pack(side=tk.LEFT, pady=5, padx=30)

        self.create_rounded_button(
            parent=move_button_frame,
            text="",
            icon=move_icon,           
            icon_hover=self.move_worm_icon_hover,
            command=lambda: self.toggle_mode_worm_scan_result(mode="move"),
            bg_color=move_bg,
            text_color=self.colors.theme["primary_text"],
            hover_color=self.colors.theme["tertiary_background"],
            font=(self.font, self.screen_height // 60), # 16
            width_pixels=self.screen_height // 10, # old 100, new 96
            height_pixels= self.screen_height // 16, # 60
            corner_radius=self.screen_height // 48, # 20
            side=tk.TOP,
            pady=0,
            padx=0,
            padx_text=-6,
            border_width=2,
            border_color=self.colors.theme["stroke_button"]
        )

        move_info_icon = tk.Label(move_button_frame, image=self.info_icon, bg=self.colors.theme["primary_background"])
        move_info_icon.pack(side=tk.TOP, pady=(4, 0))  # slight spacing above icon
        Tooltip(move_info_icon, "Click on a worm box and drag to move its position", title="Info", theme="info", posx=70, posy=-80)

        
        # -- Start analysis button --
        """start_button_frame = tk.Frame(button_row_frame, bg=self.colors.theme["primary_background"])
        start_button_frame.pack(side=tk.LEFT, pady=5, padx=30)

        self.create_rounded_button(
            parent=start_button_frame,
            text="",
            icon=self.play_icon,
            icon_hover=self.play_icon_hover,
            command=lambda: self.switch_page("load_position"),
            bg_color=self.colors.theme["primary_background"],
            text_color=self.colors.theme["primary_text"],
            hover_color=self.colors.theme["secondary_background"],
            font=(self.font, 16),
            width_pixels=200,
            height_pixels=60,
            corner_radius=20,
            side=tk.TOP,
            pady=0,
            padx=0,
            padx_text=-10,
            border_width=2,
            border_color=self.colors.theme["stroke_button"]
        )

        # Create sub-frame for label + info icon (centered below button)
        label_info_frame = tk.Frame(start_button_frame, bg=self.colors.theme["primary_background"])
        label_info_frame.pack(side=tk.TOP, pady=(4, 0), anchor="center")

        # Text label
        title_launch_analysis_result_container = tk.Label(
            label_info_frame, text="Start analysis",
            bg=self.colors.theme["primary_background"], fg=self.colors.theme["tertiary_text"],
            font=(self.font, 10)
        )
        title_launch_analysis_result_container.pack(side=tk.LEFT)

        # Info icon
        start_info_icon = tk.Label(label_info_frame, image=self.info_icon, bg=self.colors.theme["primary_background"])
        start_info_icon.pack(side=tk.LEFT, padx=(5, 0))  # slight space between label and icon

        # Tooltip
        Tooltip(start_info_icon, "Be sure to use the L camera.", posx=70, posy=-70)      
        """
        
        # -- Remove worm button --
        remove_button_frame = tk.Frame(button_row_frame, bg=self.colors.theme["primary_background"])
        remove_button_frame.pack(side=tk.LEFT, pady=5, padx=30)

        self.create_rounded_button(
            parent=remove_button_frame,
            text="",
            icon=remove_icon,
            icon_hover=self.remove_worm_icon_hover,
            command=lambda: self.toggle_mode_worm_scan_result(mode="delete"),
            bg_color=remove_bg,
            text_color=self.colors.theme["primary_text"],
            hover_color=self.colors.theme["tertiary_background"],
            font=(self.font, self.screen_height // 60), # 16
            width_pixels=self.screen_height // 10, # old 100, new 96
            height_pixels= self.screen_height // 16, # 60
            corner_radius=self.screen_height // 48, # 20
            side=tk.TOP,
            pady=0,
            padx=0,
            padx_text=-6,
            border_width=2,
            border_color=self.colors.theme["stroke_button"]
        )

        remove_info_icon = tk.Label(remove_button_frame, image=self.info_icon, bg=self.colors.theme["primary_background"])
        remove_info_icon.pack(side=tk.TOP, pady=(4, 0))
        Tooltip(remove_info_icon, "Remove the worms by clicking on them or by dragging to select them. Press the E key to clear all of them.", title="Info", theme="info", posx=70, posy=-80)
        

        # Trigger resizing after layout completes with error handling
        try:
            if hasattr(self, 'middle_result_container_ref') and self.middle_result_container_ref.winfo_exists():
                self.middle_result_container_ref.bind("<Configure>", self.resize_scan_content_area)
        except:
            pass
        
        # when e is pressed on the keyboard, apply the function self.worm_position.delete_all_worms()
        self.main_content.focus_set()  # Make sure the frame has focus to capture key events
        self.main_content.bind('e', lambda event: self.delete_all_worms())
            
        # ----- IMAGE DISPLAY -----        
        # Create canvas and store reference
        self.img_canvas = tk.Canvas(content_area_result_container, bg="black", highlightthickness=0)
        self.img_canvas.pack(fill=tk.BOTH, expand=True)

        # Initialize base image references
        self.base_stitched_image = None
        self.displayed_image = None
        self.image_scale = 1.0

        # Load the base image
        self.load_base_image()
        
        # Bind events to the canvas
        self.img_canvas.bind("<Button-1>", self.on_stitching_image_click)
        self.img_canvas.bind("<B1-Motion>", self.on_stitching_image_drag)    
        self.img_canvas.bind("<ButtonRelease-1>", self.on_stitching_image_release)
 
    def show_load_position_page(self):
        """
        Constructs the UI for the worm analysis and classification page.

        This page allows users to review individual worms from the scan results,
        capture snapshots, manually classify them as Wild-Type or Mutant, and
        view model predictions. It features a live/snapped image display,
        navigation buttons, and a panel for showing prediction results and
        manual classification buttons.
        """

        # Clear previous widgets
        for widget in self.main_content.winfo_children():
            widget.destroy()
         
        self.worms_position = WormPositionManager(new_acquisition=False, id = self.id_worm_seen)
        # Recalculate TSP here because we skipped it during batch delete
        self.worms_position.find_shortest_path()
        
        # move to the 1st worm
        x_microscope, y_microscope = self.CORE.getXYPosition()
        if [int(x_microscope), int(y_microscope)] not in self.worms_position.get_all_worm_microscope_position():
            x_microscope_1st_worm, y_microscope_1st_worm = self.worms_position.get_worm_microscope_position(self.worms_position.get_id_worm_seen())
            log_debug_coordinate(f"[Load Position] Init: Moving from current ({x_microscope}, {y_microscope}) to 1st worm ({x_microscope_1st_worm}, {y_microscope_1st_worm})")
            time.sleep(0.01)
            self.CORE.setXYPosition(self.CORE.getXYStageDevice(), x_microscope_1st_worm, y_microscope_1st_worm)

            
        # Disable some paramaters buttons   
        self.update_parameter_widgets_state(disabled_widgets=["scan_shape", "scan_objective"]) 
        self.refresh_parameters_interface()
        self.update_parameter_widgets_state(disabled_widgets=["scan_shape", "scan_objective"])
        
        # Configure grid layout for main_content
        self.main_content.grid_columnconfigure(0, weight=75)
        self.main_content.grid_columnconfigure(1, weight=25)
        self.main_content.grid_rowconfigure(0, weight=1)

        # ----- LEFT CONTAINER -----
        left_live_analysis_container = tk.Frame(self.main_content, bg=self.colors.theme["primary_background"]) 
        left_live_analysis_container.grid(row=0, column=0, sticky="nsew", padx=(0, 10))  
        self.left_live_analysis_container_ref = left_live_analysis_container

        left_live_analysis_container.grid_rowconfigure(0, weight=1)
        left_live_analysis_container.grid_rowconfigure(1, weight=0)
        left_live_analysis_container.grid_columnconfigure(0, weight=1)

        # Top: Live analysis square
        live_analysis_container = tk.Frame(
            left_live_analysis_container,
            bg=self.colors.theme["secondary_background"],
            relief=tk.RAISED,
            bd=1
        )
        live_analysis_container.grid(row=0, column=0, sticky="nsew")
        self.live_analysis_container_ref = live_analysis_container
        self.left_live_analysis_container_ref.bind("<Configure>", self.resize_live_image)
        
        # Placeholder for live image
        if not hasattr(self, "live_image_label") or not self.live_image_label.winfo_exists():
            self.live_image_label = tk.Label(live_analysis_container, bg="black", takefocus=0)
            self.live_image_label.pack(expand=True, fill=tk.BOTH)
            
            self.live_image_label.bind("<Button-1>", self.on_live_image_press)
            self.live_image_label.bind("<B1-Motion>", self.on_live_image_drag)
            self.live_image_label.bind("<ButtonRelease-1>", self.on_live_image_release)

        # Bottom: Buttons + labels
        bottom_analysis_container = tk.Frame(left_live_analysis_container, bg=self.colors.theme["primary_background"])
        bottom_analysis_container.grid(row=1, column=0, sticky="ew", pady=(10, 10))

        # --- Row that holds both button + label groups ---
        button_label_row_analysis_container = tk.Frame(bottom_analysis_container, bg=self.colors.theme["primary_background"])
        button_label_row_analysis_container.pack()

        # --- First button + label ---
        button1_analysis_container = tk.Frame(button_label_row_analysis_container, bg=self.colors.theme["primary_background"])
        button1_analysis_container.pack(side=tk.LEFT, padx=10)

        bg_live_button = self.colors.theme["tertiary_background"] if self.live_image else self.colors.theme["primary_background"]
        icon_live_button = self.live_icon_hover if self.live_image else self.live_icon
        self.live_button_ref = self.create_rounded_button(
            parent=button1_analysis_container,
            text="",
            icon=icon_live_button,
            icon_hover=self.live_icon_hover,
            command=lambda: self.start_live(),
            bg_color=bg_live_button,
            text_color=self.colors.theme["primary_text"],
            hover_color=self.colors.theme["tertiary_background"],
            font=(self.font, self.screen_height // 60), # 16
            width_pixels=self.screen_height // 5, # old 180, new 192
            height_pixels=self.screen_height // 16, # 60
            corner_radius=self.screen_height // 48, # 20
            side=tk.TOP,
            pady=5,
            padx_text=-7,
            border_width=2,
            border_color=self.colors.theme["stroke_button"]
        )

        tk.Label(
            button1_analysis_container,
            text="Live",
            bg=self.colors.theme["primary_background"],
            fg=self.colors.theme["secondary_text"],
            font=(self.font, self.screen_height // 96), # 10
            takefocus=0
        ).pack()

        # --- Second button + label ---
        button2_analysis_container = tk.Frame(button_label_row_analysis_container, bg=self.colors.theme["primary_background"])
        button2_analysis_container.pack(side=tk.LEFT, padx=10)

        bg_snap_button = self.colors.theme["primary_background"] if self.live_image else self.colors.theme["tertiary_background"]
        icon_snap_button = self.snap_icon if self.live_image else self.snap_icon_hover
        self.snap_button_ref = self.create_rounded_button(
            parent=button2_analysis_container,
            text="",
            icon=icon_snap_button,
            icon_hover=self.snap_icon_hover,
            command=lambda: self.snap_image_mode(),
            bg_color=bg_snap_button,
            text_color=self.colors.theme["primary_text"],
            hover_color=self.colors.theme["tertiary_background"],
            font=(self.font, self.screen_height // 60), # 16
            width_pixels=self.screen_height // 5, # old 180, new 192
            height_pixels=self.screen_height // 16, # 60
            corner_radius=self.screen_height // 48, # 20
            side=tk.TOP,
            pady=5,
            padx_text=-7,
            border_width=2,
            border_color=self.colors.theme["stroke_button"]
        )

        tk.Label(
            button2_analysis_container,
            text="Snap image",
            bg=self.colors.theme["primary_background"],
            fg=self.colors.theme["secondary_text"],
            font=(self.font, self.screen_height // 96), # 10
            takefocus=0
        ).pack()
        
        # --- Third button + label ---
        button3_analysis_container = tk.Frame(button_label_row_analysis_container, bg=self.colors.theme["primary_background"])
        button3_analysis_container.pack(side=tk.LEFT, padx=(20,10))

        self.save_snap_button_ref = self.create_rounded_button(
            parent=button3_analysis_container,
            text="Save image",
            command=lambda: self.save_snap_image(), 
            bg_color=self.colors.theme["primary_background"],
            text_color=self.colors.theme["secondary_text"],
            hover_color=self.colors.theme["tertiary_background"],
            font=(self.font, self.screen_height // 96), # 10
            width_pixels=self.screen_height // 8, # 120
            height_pixels=self.screen_height // 32, # 30
            corner_radius=self.screen_height // 48, # 20
            side=tk.TOP,
            pady=(0,0),
            padx_text=0,
            border_width=2,
            border_color=self.colors.theme["stroke_button"]
        )
        self.save_button_label_ref = tk.Label(
            button3_analysis_container,
            text="",
            bg=self.colors.theme["primary_background"],
            fg=self.colors.theme["tertiary_text"],
            font=(self.font, self.screen_height // 96), # 10
            takefocus=0
        )
        self.save_button_label_ref.pack()



        # ----- RIGHT CONTAINER -----
        right_map_analysis_container = tk.Frame(self.main_content, bg=self.colors.theme["primary_background"])
        right_map_analysis_container.grid(row=0, column=1, sticky="nsew", padx=(0, 0))
        self.right_map_analysis_container_ref = right_map_analysis_container

        # Make it expand vertically
        # Make it expand vertically (Remove row 3 from weights to collapse gap)
        right_map_analysis_container.grid_rowconfigure((0, 1, 2, 4, 5), weight=1)
        right_map_analysis_container.grid_columnconfigure(0, weight=1)  # left spacer
        right_map_analysis_container.grid_columnconfigure(1, weight=0)  # the container column
        right_map_analysis_container.grid_columnconfigure(2, weight=1)  # right spacer


        # 1. Histogram Container (Embedded)
        self.histogram_container = tk.Frame(right_map_analysis_container, bg=self.colors.theme["primary_background"])
        self.histogram_container.grid(row=0, column=1, sticky="nsew", pady=(10, 5))
        
        # Embed the histogram window here
        self.open_contrast_histogram_window(parent=self.histogram_container)

        # 2. Two Buttons with Text Below (Side by Side)
        mid_buttons_2_analysis_container = tk.Frame(right_map_analysis_container, bg=self.colors.theme["primary_background"], height=int(self.screen_height * 0.04))
        mid_buttons_2_analysis_container.grid(row=1, column=1, sticky="nsew")
        mid_buttons_2_analysis_container.pack_propagate(False)

        # 1st - classify as wild-type
        sub1_2_analysis_container = tk.Frame(mid_buttons_2_analysis_container, bg=self.colors.theme["primary_background"])
        sub1_2_analysis_container.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10)

        self.create_rounded_button(
            parent=sub1_2_analysis_container,
            text="",
            icon=self.wildtype_icon,
            icon_hover=self.wildtype_icon_hover,
            command=lambda: self.classify_as_wt(),
            bg_color=self.colors.theme["primary_background"],
            text_color=self.colors.theme["primary_text"],
            hover_color=self.colors.theme["secondary_background"],
            font=(self.font, self.screen_height // 69), # 14
            width_pixels=self.screen_height // 9, # old 104, new 107
            height_pixels=self.screen_height // 30, # reduced from 14 to 20
            corner_radius=self.screen_height // 48, # 20
            side=tk.TOP,
            padx_text=-5,
            pady=2,
            border_width=2,
            border_color=self.colors.theme["stroke_button"],
            autoresize=True,
            expand=True,
            fill=tk.BOTH,
            icon_path=Path(RESSOURCES_DIR) / "icon" / "wildtype.png", 
            icon_hover_path=Path(RESSOURCES_DIR) / "icon" / "wildtype.png"
        )

        tk.Label(sub1_2_analysis_container, text="Wild-Type", bg=self.colors.theme["primary_background"],
                fg=self.colors.theme["secondary_text"], font=(self.font, self.screen_height // 130), takefocus=0).pack()
        self.proportion_wt_label_ref = tk.Label(sub1_2_analysis_container, text=f"{int(100*(1-self.worms_position.get_mutant_proportion()))}%", bg=self.colors.theme["primary_background"],
                fg=self.colors.theme["secondary_text"], font=(self.font, self.screen_height // 137), takefocus=0)
        self.proportion_wt_label_ref.pack()
        
        # 2nd - classify as mutant
        sub2_2_analysis_container = tk.Frame(mid_buttons_2_analysis_container, bg=self.colors.theme["primary_background"])
        sub2_2_analysis_container.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10)

        self.create_rounded_button(
            parent=sub2_2_analysis_container,
            text="",
            icon=self.mutant_icon,
            icon_hover=self.mutant_icon_hover,
            command=lambda: self.classify_as_mutant(), 
            bg_color=self.colors.theme["primary_background"],
            text_color=self.colors.theme["primary_text"],
            hover_color=self.colors.theme["secondary_background"],
            font=(self.font, self.screen_height // 69), # 14
            width_pixels=self.screen_height // 9, # old 104, new 107
            height_pixels=self.screen_height // 30, # reduced from 14 to 20
            corner_radius=self.screen_height // 48, # 20
            side=tk.TOP,
            padx_text=-5,
            pady=2,
            border_width=2,
            border_color=self.colors.theme["stroke_button"],
            autoresize=True,
            expand=True,
            fill=tk.BOTH,
            icon_path=Path(RESSOURCES_DIR) / "icon" / "mutant.png",
            icon_hover_path=Path(RESSOURCES_DIR) / "icon" / "mutant.png"
        )

        # Mutation Label + Info Icon
        mutant_label_frame = tk.Frame(sub2_2_analysis_container, bg=self.colors.theme["primary_background"])
        mutant_label_frame.pack(side=tk.TOP, anchor="center")

        tk.Label(mutant_label_frame, text="Mutation", bg=self.colors.theme["primary_background"],
                fg=self.colors.theme["secondary_text"], font=(self.font, self.screen_height // 130), takefocus=0).pack(side=tk.LEFT)
        
        mutant_info_icon = tk.Label(mutant_label_frame, image=self.info_icon, bg=self.colors.theme["primary_background"])
        mutant_info_icon.pack(side=tk.LEFT, padx=(5, 0))
        
        Tooltip(mutant_info_icon, lambda: self.get_formatted_mutant_list(), title="Mutant List", theme="info", posx=30, posy=-50)
        self.proportion_mutant_label_ref = tk.Label(sub2_2_analysis_container, text=f"{int(100*(self.worms_position.get_mutant_proportion()))}%", bg=self.colors.theme["primary_background"],
                fg=self.colors.theme["secondary_text"], font=(self.font, self.screen_height // 137), takefocus=0)
        self.proportion_mutant_label_ref.pack()

        # 3. Text Container
        text_3_analysis_container = tk.Frame(right_map_analysis_container, bg=self.colors.theme["primary_background"])
        text_3_analysis_container.grid(row=2, column=1, sticky="ew", pady=0, ipady=0)
        
        # Container for text and icon side-by-side
        worm_count_container = tk.Frame(text_3_analysis_container, bg=self.colors.theme["primary_background"])
        worm_count_container.pack(side=tk.TOP, pady=(4, 5))

        self.id_worm_seen_label = tk.Label(
            worm_count_container,
            text=f"{int(self.id_worm_seen+1)}/{self.worms_position.get_number_of_worms()}",
            bg=self.colors.theme["primary_background"],
            fg=self.colors.theme["tertiary_text"],
            font=(self.font, self.screen_height // 120),
            takefocus=0
        )
        self.id_worm_seen_label.pack(side=tk.LEFT)

        order_worms_icon = tk.Label(worm_count_container, image=self.info_icon, bg=self.colors.theme["primary_background"])
        order_worms_icon.pack(side=tk.LEFT, padx=(5, 0))
        Tooltip(order_worms_icon, "Be aware, the display order may change if you leave this page.", title="Info", theme="info", posx=70, posy=-80)
        
        
        # Add number of worm to the statistic file
        update_user_statistics('nb_vers_final', self.worms_position.get_number_of_worms())


        # 4. Two Buttons Side by Side - Use same row to eliminate gap
        bottom_buttons_4_analysis_container = tk.Frame(text_3_analysis_container, bg=self.colors.theme["primary_background"], height=int(self.screen_height * 0.18))
        bottom_buttons_4_analysis_container.pack(side=tk.BOTTOM, pady=(0, 0), fill=tk.X)
        bottom_buttons_4_analysis_container.pack_propagate(False)

        # Label "Worm"
        tk.Label(bottom_buttons_4_analysis_container, text="Next/Last worm", bg=self.colors.theme["primary_background"],
                 fg=self.colors.theme["secondary_text"], font=(self.font, self.screen_height // 120)).pack(pady=(0, 0))

        # Create a single container for both buttons without expansion
        buttons_wrapper = tk.Frame(bottom_buttons_4_analysis_container, bg=self.colors.theme["primary_background"])
        buttons_wrapper.pack(expand=True, fill=tk.BOTH)

        # 1st - next worm
        sub1_4_analysis_container = tk.Frame(buttons_wrapper, bg=self.colors.theme["primary_background"])
        sub1_4_analysis_container.pack(side=tk.LEFT, padx=(0, 1), expand=True, fill=tk.BOTH)  
        self.create_rounded_button(
            parent=sub1_4_analysis_container,
            text="",
            icon=self.last_icon,
            icon_hover=self.last_icon_hover,
            command=lambda: self.go_to_last_worm(), 
            bg_color=self.colors.theme["primary_background"],
            text_color=self.colors.theme["primary_text"],
            hover_color=self.colors.theme["secondary_background"],
            font=(self.font, self.screen_height // 80), # 12
            width_pixels=self.screen_height // 11, 
            height_pixels=self.screen_height // 30,
            corner_radius=self.screen_height // 96, # 10
            side=tk.TOP,
            padx=10,  
            pady=2,
            padx_text=-5,
            border_width=2,
            border_color=self.colors.theme["stroke_button"],
            autoresize=True,
            expand=True,
            fill=tk.X,
            icon_path=Path(RESSOURCES_DIR) / "icon" / "last.png", 
            icon_hover_path=Path(RESSOURCES_DIR) / "icon" / "last.png"
        )

        # 2nd - last worm
        sub2_4_analysis_container = tk.Frame(buttons_wrapper, bg=self.colors.theme["primary_background"])
        sub2_4_analysis_container.pack(side=tk.LEFT, padx=(1, 0), expand=True, fill=tk.BOTH)  
        self.create_rounded_button(
            parent=sub2_4_analysis_container,
            text="",
            icon=self.next_icon,
            icon_hover=self.next_icon_hover,
            command=lambda: self.go_to_next_worm(), 
            bg_color=self.colors.theme["primary_background"],
            text_color=self.colors.theme["primary_text"],
            hover_color=self.colors.theme["secondary_background"],
            font=(self.font, self.screen_height // 80), # 12
            width_pixels=self.screen_height // 11, 
            height_pixels=self.screen_height // 30,
            corner_radius=self.screen_height // 96, # 10
            side=tk.TOP,
            padx=10, 
            pady=2,
            padx_text=-5,
            border_width=2,
            border_color=self.colors.theme["stroke_button"],
            autoresize=True,
            expand=True,
            fill=tk.X,
            icon_path=Path(RESSOURCES_DIR) / "icon" / "next.png",
            icon_hover_path=Path(RESSOURCES_DIR) / "icon" / "next.png"
        )

        warning_label = tk.Label(
            bottom_buttons_4_analysis_container,
            text="Use the space bar to move to the next worm.",
            bg=self.colors.theme["primary_background"],
            fg=self.colors.theme["secondary_text"],
            font=(self.font, self.screen_height // 130, "bold"),
            justify="center",
            takefocus=0
        )
        warning_label.pack(pady=(0, 0))
        
        # Label "Mutant"
        tk.Label(bottom_buttons_4_analysis_container, text="Next/Last mutant", bg=self.colors.theme["primary_background"],
                 fg=self.colors.theme["secondary_text"], font=(self.font, self.screen_height // 120)).pack(pady=(5, 0))

        # Create a single container for both mutant buttons without expansion
        buttons_wrapper_mutant = tk.Frame(bottom_buttons_4_analysis_container, bg=self.colors.theme["primary_background"])
        buttons_wrapper_mutant.pack(pady=(0, 0), expand=True, fill=tk.BOTH)

        # 1st - previous mutant
        sub1_mutant_analysis_container = tk.Frame(buttons_wrapper_mutant, bg=self.colors.theme["primary_background"])
        sub1_mutant_analysis_container.pack(side=tk.LEFT, padx=(0, 1), expand=True, fill=tk.BOTH)  
        self.create_rounded_button(
            parent=sub1_mutant_analysis_container,
            text="",
            icon=self.last_icon,
            icon_hover=self.last_icon_hover,
            command=lambda: self.go_to_last_mutant(), 
            bg_color=self.colors.theme["primary_background"],
            text_color=self.colors.theme["primary_text"],
            hover_color=self.colors.theme["secondary_background"],
            font=(self.font, self.screen_height // 80), 
            width_pixels=self.screen_height // 11, 
            height_pixels=self.screen_height // 30, 
            corner_radius=self.screen_height // 96, 
            side=tk.TOP,
            padx=10,  
            pady=2,
            padx_text=-5,
            border_width=2,
            border_color=self.colors.theme["stroke_button"],
            autoresize=True,
            expand=True,
            fill=tk.X,
            icon_path=Path(RESSOURCES_DIR) / "icon" / "last.png",
            icon_hover_path=Path(RESSOURCES_DIR) / "icon" / "last.png"
        )

        # 2nd - next mutant
        sub2_mutant_analysis_container = tk.Frame(buttons_wrapper_mutant, bg=self.colors.theme["primary_background"])
        sub2_mutant_analysis_container.pack(side=tk.LEFT, padx=(1, 0), expand=True, fill=tk.BOTH) 
        self.create_rounded_button(
            parent=sub2_mutant_analysis_container,
            text="",
            icon=self.next_icon,
            icon_hover=self.next_icon_hover,
            command=lambda: self.go_to_next_mutant(), 
            bg_color=self.colors.theme["primary_background"],
            text_color=self.colors.theme["primary_text"],
            hover_color=self.colors.theme["secondary_background"],
            font=(self.font, self.screen_height // 80), 
            width_pixels=self.screen_height // 11, 
            height_pixels=self.screen_height // 30, 
            corner_radius=self.screen_height // 96, 
            side=tk.TOP,
            padx=10, 
            pady=2,
            padx_text=-5,
            border_width=2,
            border_color=self.colors.theme["stroke_button"],
            autoresize=True,
            expand=True,
            fill=tk.X,
            icon_path=Path(RESSOURCES_DIR) / "icon" / "next.png",
            icon_hover_path=Path(RESSOURCES_DIR) / "icon" / "next.png"
        )

        # 5. Button + Text with Padding
        final_5_analysis_container = tk.Frame(right_map_analysis_container, bg=self.colors.theme["primary_background"], height=int(self.screen_height * 0.06))
        # Increase top padding strongly to separate from spacer label, decrease bottom padding slightly
        final_5_analysis_container.grid(row=4, column=1, sticky="ew", pady=(0, 2))
        final_5_analysis_container.pack_propagate(False)

        self.create_rounded_button(
            parent=final_5_analysis_container,
            text="",
            icon=self.play_icon,
            icon_hover=self.play_icon_hover,
            command=lambda: self.analyse_worm(),
            bg_color=self.colors.theme["primary_background"],  
            text_color=self.colors.theme["primary_text"],
            hover_color=self.colors.theme["secondary_background"],
            icon_path=Path(RESSOURCES_DIR) / "icon" / "play.png",
            icon_hover_path=Path(RESSOURCES_DIR) / "icon" / "play.png",
            font=(self.font, self.screen_height // 80), # 12
            width_pixels=self.screen_height // 4, # old 250, new 240
            height_pixels=self.screen_height // 30, # reduced from 16 to 20
            corner_radius=self.screen_height // 96, # 10
            side=tk.TOP,
            pady=5,
            padx_text=-10,
            border_width=2,
            border_color=self.colors.theme["stroke_button"],
            autoresize=True,
            expand=True,
            fill=tk.BOTH
        )

        tk.Label(
            final_5_analysis_container,
            text="Launch analysis", 
            bg=self.colors.theme["primary_background"],
            fg=self.colors.theme["secondary_text"],
            font=(self.font, self.screen_height // 120),
            takefocus=0
        ).pack()

        # 6. Label Container (Moved from top)
        self.top_label_1_analysis_container = tk.Frame(right_map_analysis_container)
        self.top_label_1_analysis_container.grid(row=5, column=1, sticky="ew", pady=(0, 30)) 
        self.top_label_1_analysis_container.grid_columnconfigure(0, weight=1)
        self.top_label_1_analysis_container.config(bg=self.colors.theme["primary_background"])

        # Create a canvas inside this frame for drawing the rounded rectangle
        self.top_label_canvas = tk.Canvas(
            self.top_label_1_analysis_container, 
            height=self.screen_height // 10, # old 100, new 96 
            highlightthickness=0,
            bg=self.colors.theme["primary_background"]
        )
        self.top_label_canvas.grid(row=0, column=0, sticky="ns")

        # Create a frame on top of the canvas to hold the labels
        self.top_label_frame = tk.Frame(self.top_label_canvas, bg=self.colors.theme["primary_background"], bd=0)
    
        # Now add labels inside self.top_label_frame
        tk.Label(
            self.top_label_frame,
            text="Synaptic profiling prediction",
            bg=self.colors.theme["primary_background"],
            fg=self.colors.theme["secondary_text"],
            font=(self.font, self.screen_height // 130, "bold"), # 10
            takefocus=0
        ).pack(pady=(0, 0), anchor="center")  

        self.prediction_label = tk.Label(
            self.top_label_frame,
            text=f"The analysed worm is a mutant",
            bg=self.colors.theme["primary_background"],
            fg=self.colors.theme["secondary_text"],
            justify="center",  
            font=(self.font, self.screen_height // 150), # 8
            takefocus=0
        )
        self.prediction_label.pack(pady=(5, 0), anchor="center")

        self.prediction_label_2 = tk.Label(
            self.top_label_frame,
            text=f"with a probability of {self.prediction} %",
            bg=self.colors.theme["primary_background"],
            fg=self.colors.theme["secondary_text"],
            justify="center",  
            font=(self.font, self.screen_height // 150), # 8
            takefocus=0
        )
        self.prediction_label_2.pack(pady=(0, 0), anchor="center")

        
        self.top_label_frame_window = self.top_label_canvas.create_window(
            (0, 0), window=self.top_label_frame, anchor="center"
        )

        self.top_label_canvas.bind("<Configure>", self.resize_prediction_result_box)
        
        self.root.lift()
        self.root.after(50, lambda: (self.root.focus_force(), self.root.update_idletasks()))
        
        # Bind arrow keys to microscope movement
        self.root.bind("<Left>", lambda event: self.move_microscope_relative('left'))
        self.root.bind("<Right>", lambda event: self.move_microscope_relative('right'))
        self.root.bind("<Up>", lambda event: self.move_microscope_relative('up'))
        self.root.bind("<Down>", lambda event: self.move_microscope_relative('down'))
        # Bind spacebar to next worm (using root to be consistent with arrow keys)
        self.root.bind("<space>", lambda event: self.go_to_next_worm())

        """self.main_content.focus_set()  # Make sure the frame has focus to capture key events
        self.main_content.bind("<Left>", lambda event: self.go_to_last_worm())
        self.main_content.bind("<Right>", lambda event: self.go_to_next_worm())"""

        # Update the live image
        if self.live_image:
            if not getattr(self, "_live_running", False):
                self._live_running = True
                self.update_live_image()
            # self.root.after(300, self._try_open_histogram) # Disabled as we embed it now

    def get_formatted_mutant_list(self):
        """
        Retrieves the list of mutant worm IDs and formats them for the tooltip.
        """
        mutant_ids = self.worms_position.get_mutant_worm_ids()
        if not mutant_ids:
            return "No mutants identified."
        
        # Sort ids just in case
        mutant_ids.sort()
        
        # Format as a vertical list string
        formatted_list = "\n".join([f"• Worm {worm_id + 1}" for worm_id in mutant_ids])
        return f"Mutant Worms:\n{formatted_list}"
    
    def show_length_analysis_page(self):
        """
        Constructs the UI for the Length Analysis page.
        """
        # Clear previous widgets
        for widget in self.main_content.winfo_children():
            widget.destroy()

        # Disable inappropriate parameters
        self.update_parameter_widgets_state(disabled_widgets=["scan_shape", "scan_objective"])
        self.refresh_parameters_interface()
        self.update_parameter_widgets_state(disabled_widgets=["scan_shape", "scan_objective"])

        # Configure grid layout for main_content
        self.main_content.grid_columnconfigure(0, weight=75)
        self.main_content.grid_columnconfigure(1, weight=25)
        self.main_content.grid_rowconfigure(0, weight=1)

        # ----- LEFT CONTAINER (Live Image) -----
        left_container = tk.Frame(self.main_content, bg=self.colors.theme["primary_background"])
        left_container.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.left_length_analysis_container_ref = left_container # Store ref for resizing

        left_container.grid_rowconfigure(0, weight=1)
        left_container.grid_columnconfigure(0, weight=1)

        # Live image container
        live_container = tk.Frame(
            left_container,
            bg=self.colors.theme["secondary_background"],
            relief=tk.RAISED,
            bd=1
        )
        live_container.grid(row=0, column=0, sticky="nsew")
        
        # Placeholder for live image
        if not hasattr(self, "live_image_label") or not self.live_image_label.winfo_exists():
            self.live_image_label = tk.Label(live_container, bg="black", takefocus=0)
            self.live_image_label.pack(expand=True, fill=tk.BOTH)
            
        # Hook up resize event if needed, reusing existing method if compatible
        # For now just simple pack, could reuse resize_live_image if self.live_analysis_container_ref is set
        self.live_analysis_container_ref = live_container # Reuse this ref name for resize_live_image compatibility
        left_container.bind("<Configure>", self.resize_live_image)


        # ----- RIGHT CONTAINER (Controls & Results) -----
        right_container = tk.Frame(self.main_content, bg=self.colors.theme["primary_background"])
        right_container.grid(row=0, column=1, sticky="nsew")
        right_container.grid_columnconfigure(0, weight=1)
        right_container.grid_rowconfigure(1, weight=1) # Table expands

        # 1. Launch Button
        button_frame = tk.Frame(right_container, bg=self.colors.theme["primary_background"])
        button_frame.grid(row=0, column=0, pady=20, sticky="ew")

        self.create_rounded_button(
            parent=button_frame,
            text="Launch length analysis",
            icon=self.play_icon,
            icon_hover=self.play_icon_hover,
            command=self.launch_length_analysis,
            bg_color=self.colors.theme["secondary_background"],
            text_color=self.colors.theme["primary_text"],
            hover_color=self.colors.theme["tertiary_background"],
            font=(self.font, 14),
            width_pixels=280,
            height_pixels=60,
            corner_radius=20,
            side=tk.TOP,
            pady=0,
            border_width=2,
            border_color=self.colors.theme["stroke_button"]
        )

        # 2. Results Table
        table_frame = tk.Frame(right_container, bg=self.colors.theme["primary_background"])
        table_frame.grid(row=1, column=0, sticky="nsew", padx=20)

        columns = ("worm_id", "length_px")
        self.length_tree = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse")
        self.length_tree.heading("worm_id", text="Worm ID")
        self.length_tree.heading("length_px", text="Length (px)")
        self.length_tree.column("worm_id", width=100, anchor="center")
        self.length_tree.column("length_px", width=150, anchor="center")
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.length_tree.yview)
        self.length_tree.configure(yscroll=scrollbar.set)
        
        self.length_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 3. Statistics
        stats_frame = tk.Frame(right_container, bg=self.colors.theme["primary_background"])
        stats_frame.grid(row=2, column=0, sticky="ew", pady=20, padx=20)
        
        self.stats_labels = {}
        for i, stat in enumerate(["Mean", "Variance", "Min", "Max", "Errors"]):
            lbl = tk.Label(
                stats_frame, 
                text=f"{stat}: --", 
                bg=self.colors.theme["primary_background"], 
                fg=self.colors.theme["primary_text"],
                font=(self.font, 12)
            )
            lbl.grid(row=i//2, column=i%2, sticky="w", padx=20, pady=5)
            self.stats_labels[stat] = lbl

        # Update the live image
        self.live_image = True
        if self.live_image:
            if not getattr(self, "_live_running", False):
                self._live_running = True
                self.update_live_image()

    def launch_length_analysis(self):
        """
        Executes the length analysis on all detected worms.
        """
        # Stop live updates during analysis to control display
        if hasattr(self, "_live_running") and self._live_running:
            self.live_image = False
            self._live_running = False
            # Cancel any pending after callbacks for live loop
            if hasattr(self, "_live_after_id") and self._live_after_id:
                try:
                    self.root.after_cancel(self._live_after_id)
                except Exception:
                    pass

        # Initialize WormPositionManager if not present
        if self.worms_position is None:
             self.worms_position = WormPositionManager(new_acquisition=False)

        all_worms = self.worms_position.get_all_worm_microscope_position()
        if not all_worms:
            log_error(Exception("No worms found"), "No worms to analyze")
            return

        # Clear table
        # Check window existence before clearing table as well
        if not self.root.winfo_exists():
            return
            
        for item in self.length_tree.get_children():
            self.length_tree.delete(item)

        lengths = []
        errors_count = 0
        
        # Iterate over worms
        for i, (x, y) in enumerate(all_worms):
            worm_start_time = time.time()
            print(f"--- Analyzing Worm {i+1} ---")

            # Safe check at start of iteration
            if not self.root.winfo_exists():
                return
                
            worm_id = i + 1 # Assuming 1-based ID for display
            
            # 1. Move Microscope
            t0 = time.time()
            try:
                log_debug_coordinate(f"[Analysis] Moving to worm {worm_id} at ({x}, {y})")
                self.CORE.setXYPosition(self.CORE.getXYStageDevice(), x, y)

                self.CORE.waitForDevice(self.CORE.getXYStageDevice())
                #time.sleep(0.5) # Settle time
            except Exception as e:
                log_error(e, f"Failed to move to worm {worm_id}")
                length = 0
                lengths.append(length)
                
                if self.root.winfo_exists():
                    self.length_tree.insert("", "end", values=(worm_id, length))
                    
                errors_count += 1
                continue
            print(f"  [Time] Move microscope: {time.time() - t0:.4f}s")
            
            # 2. Snap Image
            t0 = time.time()
            img = None
            try:
                print(f"  [Analysis] Snapping image for worm {worm_id}")
                # Use snap_image helper which handles exposure and format conversion
                img = self.snap_image(analysis_mode=True)
                
                if img is None:
                     raise Exception("Snap image returned None")

            except Exception as e:
                log_error(e, f"Failed to snap image for worm {worm_id}")
                length = 0
                lengths.append(length)
                
                if self.root.winfo_exists():
                    self.length_tree.insert("", "end", values=(worm_id, length))
                    
                errors_count += 1
                continue
            print(f"  [Time] Snap image: {time.time() - t0:.4f}s")

            # 3. Process
            try:
                # Segment
                if img is None:
                    raise Exception("Image is None")

                t0 = time.time()
                # Use self.find_worm_segmentation (uses cached model + auto_contrast)
                worm_mask = self.find_worm_segmentation(img, verbose=False)
                print(f"  [Time] Segmentation: {time.time() - t0:.4f}s")

                if worm_mask is None or np.sum(worm_mask) == 0:
                    # No worm found - try fallback or just fail
                    raise Exception("No worm detected in segmentation")

                # Skeletonize
                # Use simplified length calculation (only backbone)
                t0 = time.time()
                
                # worm_mask from find_worm_segmentation is boolean, convert to uint8 if needed by get_worm_length
                # get_worm_length uses get_backbone_graph which usually expects bool or uint8, checking usages...
                # Preprocessing.get_worm_length calls get_backbone_graph(worm_mask)
                # Let's ensure it's bool as likely expected, or whatever find_worm_segmentation returns (it returns bool mask)
                # But for visualization (find contours) we need uint8.
                
                if worm_mask.dtype == bool:
                     worm_mask_uint8 = worm_mask.astype(np.uint8)
                else:
                     worm_mask_uint8 = worm_mask
                     
                length, G = self.preprocessing.get_worm_length(img, worm_mask_uint8)
                print(f"  [Time] Skeletonization: {time.time() - t0:.4f}s")
                
                # Calculate Length (number of nodes in skeleton graph)
                if G is not None and length > 0:
                     # length is already calculated
                     lengths.append(length)
                     
                     if self.root.winfo_exists():
                         # Update Table
                         self.length_tree.insert("", "end", values=(worm_id, length))
                     
                     # --- VISUALIZATION ---
                     t0 = time.time()
                     # Convert to 8-bit if currently 16-bit (for display compatibility and visibility)
                     if img.dtype == np.uint16 or img.dtype == np.float32:
                         img_display_base = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                     else:
                         img_display_base = img.astype(np.uint8)

                     # Convert to BGR for coloring
                     display_img = cv2.cvtColor(img_display_base, cv2.COLOR_GRAY2BGR)
                     
                     # 1. Draw Segmentation (Green)
                     # Find contours requires uint8
                     contours, _ = cv2.findContours(worm_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                     cv2.drawContours(display_img, contours, -1, (0, 255, 0), 1)
                     
                     # 2. Draw Skeleton (Red)
                     # Iterate over graph edges
                     for edge in G.edges:
                         pt1 = (edge[0][1], edge[0][0]) # (x, y) for cv2, nodes are (y, x)
                         pt2 = (edge[1][1], edge[1][0])
                         cv2.line(display_img, pt1, pt2, (0, 0, 255), 1)
                         
                     # Display image
                     # Convert BGR (OpenCV) to RGB (PIL)
                     display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                     pil_img = Image.fromarray(display_img_rgb)
                     
                     # Use display_enhanced_image logic or direct update
                     if self.root.winfo_exists() and hasattr(self, "live_image_label") and self.live_image_label.winfo_exists():
                         # Resize to label
                         w = self.live_image_label.winfo_width()
                         h = self.live_image_label.winfo_height()
                         if w > 0 and h > 0:
                            pil_img = pil_img.resize((w, h), Image.Resampling.LANCZOS)
                            
                         tk_img = ImageTk.PhotoImage(pil_img)
                         self.live_image_label.configure(image=tk_img)
                         self.live_image_label.image = tk_img # Keep reference
                         
                         self.main_content.update_idletasks() # Refresh UI
                    
                     print(f"  [Time] Visualization update: {time.time() - t0:.4f}s")
                else:
                    raise Exception("Skeletonization failed (G is None or length is 0)")
                
            except Exception as e:
                log_error(e, f"Analysis failed for worm {worm_id}")
                length = 0
                lengths.append(length)
                
                if self.root.winfo_exists():
                    self.length_tree.insert("", "end", values=(worm_id, length))
                    
                errors_count += 1

            # Update stats live
            if self.root.winfo_exists() and lengths:
                # Filter out 0 lengths for statistics
                valid_lengths = [l for l in lengths if l > 0]
                
                if valid_lengths:
                    self.stats_labels["Mean"].config(text=f"Mean: {np.mean(valid_lengths):.2f}")
                    self.stats_labels["Variance"].config(text=f"Variance: {np.var(valid_lengths):.2f}")
                    self.stats_labels["Min"].config(text=f"Min: {np.min(valid_lengths)}")
                    self.stats_labels["Max"].config(text=f"Max: {np.max(valid_lengths)}")
                else:
                    self.stats_labels["Mean"].config(text=f"Mean: --")
                    self.stats_labels["Variance"].config(text=f"Variance: --")
                    self.stats_labels["Min"].config(text=f"Min: --")
                    self.stats_labels["Max"].config(text=f"Max: --")
                
                self.stats_labels["Errors"].config(text=f"Errors: {errors_count}")
                self.main_content.update_idletasks()
            
            print(f"  [Time] Total worm analysis: {time.time() - worm_start_time:.4f}s")
        
        # Restart live view after analysis
        if self.root.winfo_exists():
            self.live_image = True
            self.start_live(switch_to_load_position=False)

    def show_training_model_page(self):
        """
        Page: Training a model
        """

        # --- helpers ---
        def refresh_model_list(): 
            """Scan base dir for subdirectories and populate dropdown."""
            base = TRAINING_DIR
            os.makedirs(base, exist_ok=True)
            # list only directories
            dirs = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])
            # put into combobox values
            self.model_combobox['values'] = dirs
            # if there is at least one and none selected, set first
            if dirs and not self.selected_model_var.get():
                self.selected_model_var.set(dirs[0])
            update_instruction_label()

        def create_model_directory():  
            """Create model folder and the Mutant / WT subfolders from entry content."""
            name = new_model_var.get().strip()
            if not name:
                append_status("❗ Please enter a model name to create.")
                return
            base = TRAINING_DIR
            model_dir = os.path.join(base, name)
            try:
                os.makedirs(os.path.join(model_dir, "Mutant"), exist_ok=True)
                os.makedirs(os.path.join(model_dir, "WT"), exist_ok=True)
                append_status(f"✅ Created model directory: {model_dir} (Mutant/ and WT/).")
            except Exception as e:
                append_status(f"❌ Failed to create directories: {e}")
                return
            # refresh dropdown and select new model
            refresh_model_list()
            self.selected_model_var.set(name)

        def get_selected_model_name(): 
            """Take priority: entry (if non-empty) when creating; otherwise combobox selection."""
            sel = self.selected_model_var.get().strip()
            return sel

        def update_instruction_label(*_): 
            """Update the text instructing where to put images according to the selected model."""
            model = get_selected_model_name()
            if not model:
                text = 'Select a model or create a new one to see the directories.'
            else:
                text = f'Add your images to the directories: "{model}/Mutant" and "{model}/WT"'
            self.instruction_label.config(text=text)

        def count_images_in_dir(folder): 
            """Count image files in folder; looks for common image extensions (non-recursive)."""
            if not os.path.isdir(folder):
                return 0
            exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
            count = 0
            for e in exts:
                count += len(glob.glob(os.path.join(folder, e)))
            return count

        def append_status(text): 
            """Append a line to the status box."""
            self.status_text.configure(state='normal')
            self.status_text.insert("end", text + "\n")
            self.status_text.see("end")
            self.status_text.configure(state='disabled')
            # Force immediate update of the specific widget and root
            self.status_text.update_idletasks()
            if self.root:
                 self.root.update_idletasks()

        def clear_status(): 
            self.status_text.configure(state='normal')
            self.status_text.delete("1.0", "end")
            self.status_text.configure(state='disabled')

        def on_train_clicked():
            """Validate folders & images, then either call a real train function or show placeholder status."""
            clear_status()
            model = get_selected_model_name()
            if not model:
                append_status("❗ No model selected. Please select a model from the dropdown.")
                return
            base = TRAINING_DIR
            model_dir = os.path.join(base, model)
            mutant_dir = os.path.join(model_dir, "Mutant")
            wt_dir = os.path.join(model_dir, "WT")

            # Check directories
            if not os.path.isdir(model_dir):
                append_status(f"❗ Model directory not found: {model_dir}")
                append_status("You can create it by typing a name in the 'New model name' box and pressing 'Create directory'.")
                return

            mutant_count = count_images_in_dir(mutant_dir)
            wt_count = count_images_in_dir(wt_dir)

            append_status(f"Model: {model}")
            append_status(f"Found {mutant_count} image(s) in: {mutant_dir}")
            append_status(f"Found {wt_count} image(s) in: {wt_dir}")

            if mutant_count == 0 and wt_count == 0:
                append_status("⚠️ No images found in either directory. Please add images and try again.")
                return
            if mutant_count == 0:
                append_status("⚠️ No images found in Mutant. Please add images to proceed.")
                return
            if wt_count == 0:
                append_status("⚠️ No images found in WT. Please add images to proceed.")
                return
            if mutant_count < 50 or wt_count < 50:
                append_status("⚠️ Warning: You must have at least 50 images in each category to train a new model.")
                return

            # At this point, the minimal checks are passed.
            append_status("🚀 Starting training")

            try:
                dataset_training = Dataset_Manager()
                append_status("Load images... (~2s/image)")
                model = get_selected_model_name()
                _, mutants_filename, wt_filename, _ = dataset_training.load_images(training = True, model_name = str(model))
                append_status("✅ Images loaded")
            except Exception as e:
                self.context_error = log_error(e, f"[TRAINING MODEL] - Loading Images")
            
            try:
                append_status("Compute features... (~1.5s/image)")
                dataset_training.set_features()
                append_status("✅ Features computed")
            except Exception as e:
                self.context_error = log_error(e, f"[TRAINING MODEL] - Compute Features")

            try:
                append_status("Compute model...")
                self.root.update_idletasks() # Force UI update before heavy op
                _, score = dataset_training.get_model(compute = True, verbose_plot = False, model_name = str(model))
                append_status("✅ Model computed")
                append_status("The accuracy score for this model is {:.2f}%".format(score*100))
                append_status("You can now choose to use this model when analysing a worm")
            except Exception as e:
                self.context_error = log_error(e, f"[TRAINING MODEL] - Compute Model")

        # --- Clear previous widgets in main_content ---
        for widget in self.main_content.winfo_children():
            widget.destroy()

        # Layout configuration: provide rows for title + steps + 3 parts
        self.main_content.grid_columnconfigure(0, weight=1) 
        self.main_content.grid_rowconfigure(0, weight=0) # title
        self.main_content.grid_rowconfigure(1, weight=0) # step 1
        self.main_content.grid_rowconfigure(2, weight=0) # part 1
        self.main_content.grid_rowconfigure(3, weight=0) # step 2
        self.main_content.grid_rowconfigure(4, weight=0) # part 2
        self.main_content.grid_rowconfigure(5, weight=0) # step 3
        self.main_content.grid_rowconfigure(6, weight=0) # part 3

        # Title (centered)
        title_label = tk.Label(
            self.main_content,
            text="Training a model",
            bg=self.colors.theme["primary_background"],
            fg=self.colors.theme["primary_text"],
            font=(self.font, 18, "bold"),
            justify="center",
            anchor="center"
        )
        title_label.grid(row=0, column=0, pady=(20, 40))

        # Steps text (centered, small)
        steps_label_1 = tk.Label(
            self.main_content,
            text="Step 1 : Choose or create a model",
            bg=self.colors.theme["primary_background"],
            fg=self.colors.theme["secondary_text"],
            font=(self.font, 16),
            justify="center",
            anchor="center"
        )
        steps_label_1.grid(row=1, column=0, pady=(0, 12))

        # --- PART 1: Model selection / creation (centered contents) ---
        part1 = tk.Frame(self.main_content, bg=self.colors.theme["primary_background"])
        part1.grid(row=2, column=0, padx=20, pady=(0, 10))  # removed sticky to avoid full-width stretching

        # Inner centered container for part1
        inner1 = tk.Frame(part1, bg=self.colors.theme["primary_background"])
        inner1.pack(anchor="center")

        # Existing models label + combobox
        tk.Label(inner1, text="Select existing model:", bg=self.colors.theme["primary_background"],
                fg=self.colors.theme["secondary_text"], font=(self.font, 10)).pack(anchor="center", pady=(0, 4))

        self.selected_model_var = tk.StringVar()
        self.model_combobox = ttk.Combobox(inner1, textvariable=self.selected_model_var, state="readonly")
        # make combobox shorter and centered text
        self.model_combobox.config(width=30)
        self.model_combobox.pack(anchor="center", pady=(0, 8))

        self.model_combobox.bind("<<ComboboxSelected>>", lambda e: update_instruction_label())

        # New model entry + create directory button
        tk.Label(inner1, text="New model name:", bg=self.colors.theme["primary_background"],
                fg=self.colors.theme["secondary_text"], font=(self.font, 10)).pack(anchor="center", pady=(6, 4))

        new_model_var = tk.StringVar()
        entry_new = tk.Entry(inner1, textvariable=new_model_var)
        entry_new.config(width=30, justify='center')  # shorter entry
        entry_new.pack(anchor="center", pady=(0, 8))

        # Create directory button (centered)
        create_btn_container = tk.Frame(inner1, bg=self.colors.theme["primary_background"])
        create_btn_container.pack(anchor="center", pady=(4, 0))
        self.create_rounded_button(
            parent=create_btn_container,
            text="Create directory",
            command=create_model_directory,
            bg_color=self.colors.theme["primary_background"],
            text_color=self.colors.theme["primary_text"],
            hover_color=self.colors.theme["tertiary_background"],
            font=(self.font, 10),
            width_pixels=160,
            height_pixels=36,
            corner_radius=12,
            side=tk.TOP,
            padx_text=0,
            border_width=2,
            border_color=self.colors.theme["stroke_button"]
        )

        # Steps text (centered, small)
        steps_label_2 = tk.Label(
            self.main_content,
            text="Step 2 : Add images",
            bg=self.colors.theme["primary_background"],
            fg=self.colors.theme["secondary_text"],
            font=(self.font, 16),
            justify="center",
            anchor="center"
        )
        steps_label_2.grid(row=3, column=0, pady=(40, 12))
        
        # --- PART 2: Instruction text for where to add images (centered) ---
        part2 = tk.Frame(self.main_content, bg=self.colors.theme["primary_background"])
        part2.grid(row=4, column=0, padx=20, pady=(6, 10))

        self.instruction_label = tk.Label(
            part2,
            text="",  # will be set by update_instruction_label()
            bg=self.colors.theme["primary_background"],
            fg=self.colors.theme["secondary_text"],
            font=(self.font, 10),
            wraplength=600,
            justify="center",
            anchor="center"
        )
        self.instruction_label.pack(anchor="center")

        # Update as soon as the page is shown
        refresh_model_list()
        
        # ---- add AFTER: self.instruction_label.pack(anchor="center") ----

        # frame for the two "Open folder" buttons and counters
        folders_frame = tk.Frame(part2, bg=self.colors.theme["primary_background"])
        folders_frame.pack(anchor="center", pady=(12, 0))

        # small vars to display counts
        mutant_count_var = tk.StringVar(value="Mutant: 0 images")
        wt_count_var = tk.StringVar(value="WT: 0 images")
        
        def open_folder_in_explorer(path):
            """Create folder if missing and open it in file explorer (Windows primary)."""
            try:
                os.makedirs(path, exist_ok=True)
            except Exception as e:
                append_status(f"❌ Could not create folder {path}: {e}")
                return
            try:
                if sys.platform.startswith("win"):
                    os.startfile(path)      # Windows
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", path])
                else:
                    subprocess.Popen(["xdg-open", path])
                append_status(f"📁 Opened: {path}")
            except Exception as e:
                append_status(f"❌ Failed to open folder {path}: {e}")

        def open_mutant_folder():
            model = get_selected_model_name()
            if not model:
                append_status("❗ No model selected. Please select or create a model first.")
                return
            mutant_dir = os.path.join(TRAINING_DIR, model, "Mutant")
            open_folder_in_explorer(mutant_dir)
            update_counts()

        def open_wt_folder():
            model = get_selected_model_name()
            if not model:
                append_status("❗ No model selected. Please select or create a model first.")
                return
            wt_dir = os.path.join(TRAINING_DIR, model, "WT")
            open_folder_in_explorer(wt_dir)
            update_counts()

        def update_counts(*_):
            """Refresh the small labels showing how many images are in each folder."""
            model = get_selected_model_name()
            if not model:
                mutant_count_var.set("Mutant: -")
                wt_count_var.set("WT: -")
                return
            base = os.path.join(TRAINING_DIR, model)
            mutant_count = count_images_in_dir(os.path.join(base, "Mutant"))
            wt_count = count_images_in_dir(os.path.join(base, "WT"))
            mutant_count_var.set(f"Mutant: {mutant_count} image(s)")
            wt_count_var.set(f"WT: {wt_count} image(s)")

        # Buttons and counters layout (centered)
        btns_container = tk.Frame(folders_frame, bg=self.colors.theme["primary_background"])
        btns_container.pack(anchor="center")

        # Mutant button + counter
        mutant_container = tk.Frame(btns_container, bg=self.colors.theme["primary_background"])
        mutant_container.grid(row=0, column=0, padx=10)
        self.create_rounded_button(
            parent=mutant_container,
            text="Open Mutant folder",
            command=open_mutant_folder,
            bg_color=self.colors.theme["primary_background"],
            text_color=self.colors.theme["primary_text"],
            hover_color=self.colors.theme["tertiary_background"],
            font=(self.font, 10),
            width_pixels=160,
            height_pixels=36,
            corner_radius=12,
            side=tk.TOP,
            padx_text=0,
            border_width=2,
            border_color=self.colors.theme["stroke_button"]
        )
        tk.Label(mutant_container, textvariable=mutant_count_var,
                bg=self.colors.theme["primary_background"],
                fg=self.colors.theme["secondary_text"],
                font=(self.font, 9)).pack(anchor="center", pady=(6,0))

        # WT button + counter
        wt_container = tk.Frame(btns_container, bg=self.colors.theme["primary_background"])
        wt_container.grid(row=0, column=1, padx=10)
        self.create_rounded_button(
            parent=wt_container,
            text="Open WT folder",
            command=open_wt_folder,
            bg_color=self.colors.theme["primary_background"],
            text_color=self.colors.theme["primary_text"],
            hover_color=self.colors.theme["tertiary_background"],
            font=(self.font, 10),
            width_pixels=160,
            height_pixels=36,
            corner_radius=12,
            side=tk.TOP,
            padx_text=0,
            border_width=2,
            border_color=self.colors.theme["stroke_button"]
        )
        tk.Label(wt_container, textvariable=wt_count_var,
                bg=self.colors.theme["primary_background"],
                fg=self.colors.theme["secondary_text"],
                font=(self.font, 9)).pack(anchor="center", pady=(6,0))

        # Make sure counts update when model selection changes
        self.model_combobox.bind("<<ComboboxSelected>>", lambda e: update_counts())
        # Also update right away
        update_counts()

        # ----------------------------------------------------------------

        
        
        
        
        


        # Steps text (centered, small)
        steps_label_3 = tk.Label(
            self.main_content,
            text="Step 3 : Train the model",
            bg=self.colors.theme["primary_background"],
            fg=self.colors.theme["secondary_text"],
            font=(self.font, 16),
            justify="center",
            anchor="center"
        )
        steps_label_3.grid(row=5, column=0, pady=(40, 12))
        
        # --- PART 3: Train button + status area (centered) ---
        part3 = tk.Frame(self.main_content, bg=self.colors.theme["primary_background"])
        part3.grid(row=6, column=0, padx=20, pady=(10, 20))  # removed sticky; bottom area

        inner3 = tk.Frame(part3, bg=self.colors.theme["primary_background"])
        inner3.pack(anchor="center")

        # Train button (centered)
        train_btn_container = tk.Frame(inner3, bg=self.colors.theme["primary_background"])
        train_btn_container.pack(anchor="center", pady=(0, 8))
        self.create_rounded_button(
            parent=train_btn_container,
            text="Train model",
            command=on_train_clicked,
            bg_color=self.colors.theme["primary_background"],
            text_color=self.colors.theme["primary_text"],
            hover_color=self.colors.theme["secondary_background"],
            font=(self.font, 14),
            width_pixels=220,
            height_pixels=48,
            corner_radius=12,
            side=tk.TOP,
            padx_text=0,
            border_width=2,
            border_color=self.colors.theme["stroke_button"]
        )

        # Status / message area (Text, no scrollbar, same background as page)
        self.status_text = tk.Text(
            inner3,
            height=16,
            state='disabled',
            wrap='word',
            bg=self.colors.theme["primary_background"],  # same bg as page
            fg=self.colors.theme["primary_text"],
            font=(self.font, 10),
            bd=0,
            relief='flat',
            highlightthickness=0,
            insertbackground=self.colors.theme["primary_text"]
        )
        self.status_text.pack(anchor="center", fill="x", padx=40, pady=(6, 0))

        # focus and key binds (optional)
        self.main_content.focus_set()

        # Ensure instruction label updates when combobox changes or when entry changes
        new_model_var.trace_add("write", lambda *args: None)  # no-op; entry reserved for create action
        self.model_combobox.bind("<<ComboboxSelected>>", update_instruction_label)
        update_instruction_label()

    def show_documentation_page(self):
        """
        Constructs the UI for a Documentation page that explains the three main pages:
        - Automatic Scan (show_automatic_scan_page)
        - Scan Results (show_result_scan_page)
        - Worm Analysis (show_load_position_page)

        The doc page provides:
        - A scrollable area containing one section per page with description, steps,
            important controls, tips, and warnings.
        - A search box to filter sections.
        - Collapsible sections for compactness.
        """
        # Clear previous widgets
        for widget in self.main_content.winfo_children():
            widget.destroy()

        # Top area: Title + search
        top_frame = tk.Frame(self.main_content, bg=self.colors.theme["primary_background"])
        top_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        title_label = tk.Label(
            top_frame,
            text="Documentation",
            bg=self.colors.theme["primary_background"],
            fg=self.colors.theme["secondary_text"],
            font=(self.font, 16, "bold")
        )
        title_label.pack(side=tk.LEFT, anchor="w")

        # Search entry to filter sections
        search_frame = tk.Frame(top_frame, bg=self.colors.theme["primary_background"])
        search_frame.pack(side=tk.RIGHT, anchor="e")

        search_var = tk.StringVar()

        search_entry = tk.Entry(
            search_frame,
            textvariable=search_var,
            bg=self.colors.theme["secondary_background"],
            fg=self.colors.theme["primary_text"],
            relief=tk.FLAT,
            font=(self.font, 10),
            width=28
        )
        search_entry.pack(side=tk.LEFT, padx=(0,8))

        search_icon_label = tk.Label(search_frame, image=getattr(self, "info_icon", None),
                                    bg=self.colors.theme["primary_background"])
        # safe: if info_icon missing it will show nothing
        search_icon_label.pack(side=tk.LEFT)
        Tooltip(search_icon_label, "Type to filter documentation sections", theme="info", title="Info", posx=70, posy=-40)

        # Middle: Scrollable area
        scroll_container = tk.Frame(self.main_content, bg=self.colors.theme["primary_background"])
        scroll_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5,10))

        doc_canvas = tk.Canvas(
            scroll_container,
            bg=self.colors.theme["primary_background"],
            highlightthickness=0
        )
        v_scroll = tk.Scrollbar(scroll_container, orient=tk.VERTICAL, command=doc_canvas.yview)
        doc_canvas.configure(yscrollcommand=v_scroll.set)

        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        doc_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner_frame = tk.Frame(doc_canvas, bg=self.colors.theme["primary_background"])
        inner_id = doc_canvas.create_window((0, 0), window=inner_frame, anchor="nw")

        # Ensure scrolling region updates
        def _on_configure(event):
            doc_canvas.configure(scrollregion=doc_canvas.bbox("all"))
        inner_frame.bind("<Configure>", _on_configure)

        # Resize inner window width on canvas resize
        def _on_canvas_resize(event):
            canvas_width = event.width
            doc_canvas.itemconfig(inner_id, width=canvas_width)
        doc_canvas.bind("<Configure>", _on_canvas_resize)

        # Helper: create a collapsible section for each page
        section_frames = []  # store tuples (frame, text) to allow search filtering

        def make_section(title, content_lines, quick_action=None, shortcuts=None, tips=None, warning=None):
            """
            title: str
            content_lines: list[str] - paragraphs / bullet points
            quick_action: dict with keys {'label': 'Open page', 'command': lambda: ...}
            shortcuts: list[str]
            tips: list[str]
            warning: str
            """
            container = tk.Frame(inner_frame, bg=self.colors.theme["primary_background"], bd=0)
            container.pack(fill=tk.X, pady=(8, 8), padx=4)

            # Header (clickable)
            header_frame = tk.Frame(container, bg=self.colors.theme["secondary_background"])
            header_frame.pack(fill=tk.X, ipady=6)

            header_lbl = tk.Label(
                header_frame,
                text=title,
                bg=self.colors.theme["secondary_background"],
                fg=self.colors.theme["primary_text"],
                font=(self.font, 12, "bold"),
                anchor="w"
            )
            header_lbl.pack(side=tk.LEFT, padx=10)

            # Body (collapsible)
            body = tk.Frame(container, bg=self.colors.theme["primary_background"], padx=10, pady=8)
            body.pack(fill=tk.X)

            # fill content
            for line in content_lines:
                tk.Label(body, text=line, wraplength=860,
                        justify="left",
                        bg=self.colors.theme["primary_background"],
                        fg=self.colors.theme["secondary_text"],
                        font=(self.font, 10)).pack(anchor="w", pady=(2,2))

            if shortcuts:
                tk.Label(body, text="Shortcuts:", bg=self.colors.theme["primary_background"],
                        fg=self.colors.theme["tertiary_text"], font=(self.font, 9, "bold")).pack(anchor="w", pady=(6,0))
                for s in shortcuts:
                    tk.Label(body, text=f"• {s}", wraplength=860, justify="left",
                            bg=self.colors.theme["primary_background"], fg=self.colors.theme["secondary_text"],
                            font=(self.font, 9)).pack(anchor="w")

            if tips:
                tk.Label(body, text="Tips:", bg=self.colors.theme["primary_background"],
                        fg=self.colors.theme["tertiary_text"], font=(self.font, 9, "bold")).pack(anchor="w", pady=(6,0))
                for t in tips:
                    tk.Label(body, text=f"• {t}", wraplength=860, justify="left",
                            bg=self.colors.theme["primary_background"], fg=self.colors.theme["secondary_text"],
                            font=(self.font, 9)).pack(anchor="w")

            if warning:
                tk.Label(body, text="Warning:", bg=self.colors.theme["primary_background"],
                        fg=self.colors.theme["tertiary_text"], font=(self.font, 9, "bold")).pack(anchor="w", pady=(6,0))
                tk.Label(body, text=warning, wraplength=860, justify="left",
                        bg=self.colors.theme["primary_background"], fg=self.colors.theme["secondary_text"],
                        font=(self.font, 9, "bold")).pack(anchor="w", pady=(2,0))

            # Collapsible behavior
            body.visible = True
            def toggle_body(event=None):
                if body.visible:
                    body.pack_forget()
                    body.visible = False
                else:
                    body.pack(fill=tk.X)
                    body.visible = True

            # clicking header toggles
            header_frame.bind("<Button-1>", toggle_body)
            header_lbl.bind("<Button-1>", toggle_body)

            # record for search
            section_frames.append((container, " ".join([title] + content_lines + (tips or []) + (shortcuts or []))))
            return container

        # Prepare content for each of your three pages
        # Prepare content for each of your three pages
        
        # ----------------- Getting Started / Interface -----------------
        intro_content = [
            "Welcome to the Worm Analysis App! This application is designed for automated scanning, detection, and analysis of C. elegans worms.",
            "Use the sidebar menu on the left to navigate between the main modules of the application.",
            "Top Bar Controls:",
            "- Dark/Light Mode: Toggle the application theme.",
            "- '...' Button: Opens/Closes the Parameters Panel on the right side.",
            "Parameters Panel:",
            "- Allows you to configure exposure time, binning, objectives, and other hardware settings.",
            "- Changes here apply immediately to the connected hardware."
        ]
        intro_shortcuts = [
            "Use the sidebar to quickly switch between tasks.",
            "Click parameters (...) to adjust camera settings on the fly."
        ]
        
        # ----------------- Automatic Scan Page -----------------
        automatic_content = [
            "Purpose: Start and run an automated microscope scan that stitches images to produce a full field view.",
            "Process:",
            "1. The microscope scans the slide based on the configured 'Scan shape' and 'Search area'.",
            "2. It automatically detects worms in the field of view.",
            "3. Bounding boxes are drawn around detected worms, and their positions are saved.",
            "Main UI elements:",
            "- Large resizable content area: Shows the live feed (before scan) or scan progress.",
            "- 'Launch scan' button: Starts the automated process.",
            "- Status label: Displays the current action (e.g., 'Moving to position...', 'Acquiring image').",
            "Behavior notes: Many scan parameters get disabled on this page during acquisition to prevent inconsistencies.",
            "Performance: The scan preview might pause updates during high-speed movement to prioritize data capture.",
            "Completion: After the scan finishes, you will be automatically redirected to the Scan Results page."
        ]
        automatic_shortcuts = [
            "Make sure the objective is in the lower-right corner (default home) before launching if required by your setup.",
        ]
        automatic_tips = [
            "Use the 'Scan Objective' (e.g., 4x) for faster scans covering larger areas.",
            "Ensure the stage is homed and the objective turret is in the correct position before starting."
        ]
        automatic_warning = "The default detection model is optimized for 4x magnification with high white light intensity. Using other settings may miss worms."

        # ----------------- Scan Results Page -----------------
        result_content = [
            "Purpose: Review the stitched scan image and manually correct worm detections.",
            "Workflow:",
            "1. Inspect the stitched map. Red boxes indicate detected worms.",
            "2. Use 'Add worm' to mark missed worms.",
            "3. Use 'Remove worm' to delete false positives or debris.",
            "4. Use 'Move worm' to move a box to the wished location.",
            "Main UI elements:",
            "- Add / Move / Remove toggle buttons: Switch between correction modes."
        ]
        result_shortcuts = [
            "Click on the stitched image to add a worm (when in Add mode).",
            "Drag over worms to remove them (when in Remove mode).",
            "Press 'E' to clear ALL detected worms from the scan (Use with caution!)."
        ]
        result_tips = [
            "The active mode (Add/Move/Remove) is highlighted visually.",
            "You cannot change scan geometry or objectives here; this page is for data validation only."
        ]
        result_warning = "Once you leave this page by starting analysis, the list of worms is finalized for the next step."

        # ----------------- Worm Analysis Page -----------------
        load_content = [
            "Purpose: Analyze individual worms one by one. Features live preview, snapping, AI prediction, and manual classification.",
            "Interface Structure:",
            "- Left Panel: Live camera feed or snapped image. Use the histogram tool to adjust brightness/contrast.",
            "- Right Panel: Prediction results, manual classification buttons, and navigation controls.",
            "Workflow:",
            "1. Navigate between worms using the Space bar or the Next/Last worm buttons.",
            "2. The stage automatically moves to center the selected worm.",
            "3. 'Launch Analysis' button: Runs the specific activity/synapse classification model on the current view.",
            "4. Review the AI prediction (Synaptic profiling prediction) appearing at the bottom right.",
            "5. Manually classify as 'Wild-Type' or 'Mutant' if needed.",
            "6. Navigate specifically between mutants using 'Next/Last mutant' buttons if revisiting data.",
            "7. 'Save image': Save a snapshot of the current view to disk."
        ]
        load_shortcuts = [
            "Space bar: Navigate to the next worm.",
            "Arrow keys (Up/Down/Left/Right): Move the microscope stage manually relative to the current position.",
            "Do NOT use the joystick to move between worms; the app tracks coordinates internally."
        ]
        load_tips = [
            "Click on the live image to recenter the stage on that specific point.",
            "Use 'Next/Last mutant' to quickly review all worms classified as mutants.",
            "Adjust the histogram (brightness/contrast) to see faint features better."
        ]
        load_warning = "Avoid manually moving the stage with the joystick, as it may desynchronize the app's coordinate system."


        # ----------------- Training Model Page -----------------
        training_content = [
            "Purpose: Train a new AI model for worm classification (e.g., Mutant vs WT) using your own images.",
            "Prerequisites:",
            "- A folder structure will be created: TRAINING_DIR/<model_name>/Mutant/ and .../WT/.",
            "- You need at least 50 images in EACH category.",
            "- Images should be .tiff format.",
            "Process:",
            "1. Create a new model directory or select an existing one.",
            "2. Use the 'Open folder' buttons to place your training images into the respective 'Mutant' and 'WT' folders.",
            "3. Click 'Train model'.",
            "4. The app validates folders, loads images (~2s/image), computes features, and trains the classifier.",
            "5. Upon success, the accuracy score is displayed, and the model becomes available for selection in the Parameters panel."
        ]
        training_shortcuts = [
            "Type a name and click 'Create directory' to set up the workspace.",
        ]
        training_tips = [
            "Use simple names for models (no spaces/special chars specific to your OS).",
            "The training process moves 'validation' images automatically; keep backups of your raw data."
        ]
        training_warning = "Training is CPU-intensive. Ensure you have enough disk space and do not close the app during training."

        # ----------------- Parameters & Configuration -----------------
        params_content = [
            "The Parameters Panel (right side) allows you to control the microscope and camera:",
            "- Scan Objective: Selects the objective lens used for the automated scan (usually low mag, e.g., 4x).",
            "- scan shape: Defines the geometry of the scan area (Square or Rectangle).",
            "- Exposure time (ms): Sets how long the camera sensor collects light. Increase for brighter images, decrease for less motion blur.",
            "- Binning: Combines pixels to increase sensitivity (e.g., 2x2), at the cost of resolution.",
            "- Live objective: Selects the lens used for the detailed 'Analyse Worms' phase (usually high mag, e.g., 40x or 100x).",
            "- Model name: Choose which AI model to use for the 'Analyse' button in the Worm Analysis page."
        ]
        params_tips = [
            "Parameters are disabled during active scans to prevent hardware conflicts.",
            "Dual View options appear here if enabled in Machine Configuration."
        ]

        # ----------------- Machine Configuration -----------------
        config_content = [
            "Found in 'Menu > Help > Machine Config'.",
            "Purpose: Set global hardware constants. THESE SHOULD RARELY BE CHANGED.",
            "Settings:",
            "- Dual View Mode: Toggle this if your system uses a dual-view splitter (allows simultaneous dual-channel imaging).",
            "- Scan Area Dimensions: Defines the physical travel limits of the scan in microscope units (default 26000 x 45000).",
            "- Objective Offsets/Parcentricity: Calibrate the position difference between objectives."
        ]
        config_warning = "Incorrectly changing scan dimensions or offsets can cause the stage to crash or coordinates to be wrong. Only change if you know the physical limits."

        # ----------------- Troubleshooting -----------------
        trouble_content = [
            "Q: The live image is black.",
            "A: Check if the light source is on, the shutter is open, and exposure time is high enough. Also check 'Binning' matches the camera capabilities.",
            "",
            "Q: 'No images found' during training.",
            "A: Ensure you put the images inside the specific 'Mutant' and 'WT' subfolders created by the app, not just the root model folder.",
            "",
            "Q: Stage is moving to the wrong place.",
            "A: Did you move the joystick manually? Try re-homing the stage or restarting the app to reset coordinates."
        ]

        # Create sections with quick navigation buttons that try to call your page switcher
        make_section("Introduction", intro_content, shortcuts=intro_shortcuts)
        
        make_section(
            "Automatic Scan",
            automatic_content,
            quick_action={"label": "Open Automatic Scan", "command": lambda: self.switch_page("automatic_scan")},
            shortcuts=automatic_shortcuts, tips=automatic_tips, warning=automatic_warning
        )

        make_section(
            "Scan Results",
            result_content,
            quick_action={"label": "Open Scan Results", "command": lambda: self.switch_page("result_scan")},
            shortcuts=result_shortcuts, tips=result_tips, warning=result_warning
        )

        make_section(
            "Analyse Worms",
            load_content,
            quick_action={"label": "Open Worm Analysis", "command": lambda: self.switch_page("load_position")},
            shortcuts=load_shortcuts, tips=load_tips, warning=load_warning
        )
        
        make_section(
            "Training a model",
            training_content,
            quick_action={"label": "Open Training Page", "command": lambda: self.switch_page("training_model")},
            shortcuts=training_shortcuts, tips=training_tips, warning=training_warning
        )

        make_section("Parameters & Settings", params_content, tips=params_tips)

        make_section(
            "Machine Configuration",
            config_content,
            quick_action={"label": "Open Config", "command": lambda: self.switch_page("configuration")},
            warning=config_warning
        )

        make_section("Troubleshooting", trouble_content)

        # Footer: general notes and close/back button
        footer_frame = tk.Frame(self.main_content, bg=self.colors.theme["primary_background"])
        footer_frame.pack(fill=tk.X, padx=10, pady=(6,12))

        notes_label = tk.Label(
            footer_frame,
            text="Notes: This documentation page gives quick operational tips. For implementation details, check the source methods in the codebase.",
            bg=self.colors.theme["primary_background"],
            fg=self.colors.theme["tertiary_text"],
            font=(self.font, 9),
            justify="left"
        )
        notes_label.pack(side=tk.LEFT, anchor="w")


        # Live search functionality
        def apply_search(*args):
            q = search_var.get().strip().lower()
            for frame, text in section_frames:
                if q == "" or q in text.lower():
                    frame.pack(fill=tk.X, pady=(8,8), padx=4)
                else:
                    frame.pack_forget()

            # scroll to top after filtering
            self.main_content.update_idletasks()
            doc_canvas.yview_moveto(0.0)

        search_var.trace_add("write", apply_search)

        # Set focus to main_content for key handling if needed
        try:
            self.main_content.focus_set()
        except:
            pass

        # Trigger layout resize helper similar to other pages
        try:
            if hasattr(self, 'main_content') and self.main_content.winfo_exists():
                after_id = self.main_content.after(100, lambda: doc_canvas.configure(scrollregion=doc_canvas.bbox("all")))
                if not hasattr(self, '_after_ids'):
                    self._after_ids = []
                self._after_ids.append(after_id)
        except:
            pass
       
    def show_machine_configuration_page(self):
        """
        Displays the machine configuration page in the main content area.

        This method clears any existing widgets, sets up the UI for configuring machine parameters,
        and adds a toggle for dual view mode with an info tooltip. It also triggers resizing of the
        scan content area after layout completion.
        """
        # Clear previous widgets if needed
        for widget in self.main_content.winfo_children():
            widget.destroy()
            
        # Disable some paramaters buttons
        self.update_parameter_widgets_state(disabled_widgets=[])  # Everything enabled
        
        # -------------------------------------------------------------------------------------------------- #
        
        # section with buttons dual view
        title_frame = tk.Frame(self.main_content, bg=self.colors.theme["primary_background"])
        title_frame.pack(fill=tk.BOTH, expand=True, pady=(40,10))

        # Text label
        title_label = tk.Label(
            title_frame, text="Be careful when you change these values.",
            bg=self.colors.theme["primary_background"], fg=self.colors.theme["danger_stroke"],
            font=(self.font, 12)
        )
        title_label.pack()
        
        title_label_2 = tk.Label(
            title_frame, text="Once you have configured them for your system's operation, do not change them again.",
            bg=self.colors.theme["primary_background"], fg=self.colors.theme["danger_stroke"],
            font=(self.font, 12)
        )
        title_label_2.pack()
        
        title_label_3 = tk.Label(
            title_frame, text="Incorrectly configured values will prevent the program from working properly.",
            bg=self.colors.theme["primary_background"], fg=self.colors.theme["danger_stroke"],
            font=(self.font, 12)
        )
        title_label_3.pack()

        # -------------------------------------------------------------------------------------------------- #

        """# section with buttons dual view
        buttons_dual_view_frame = tk.Frame(self.main_content, bg=self.colors.theme["primary_background"])
        buttons_dual_view_frame.pack(fill=tk.BOTH, expand=True, pady=(50,10))

        # Machine_has_dual_view button
        self.Machine_has_dual_view_toggle = self.create_custom_toggle(buttons_dual_view_frame, "", self.machine_has_dual_view, size="big", bg="primary_background")
        self.Machine_has_dual_view_toggle.pack(expand=True)
        
        # Container to hold label + info icon
        launch_label_frame = tk.Frame(buttons_dual_view_frame, bg=self.colors.theme["primary_background"])
        launch_label_frame.pack()

        # Text label
        title_launch_scan = tk.Label(
            launch_label_frame, text="Does it have a dual-view mode ?",
            bg=self.colors.theme["primary_background"], fg=self.colors.theme["tertiary_text"],
            font=(self.font, 10)
        )
        title_launch_scan.pack(side=tk.LEFT)

        # Info icon
        info_label = tk.Label(
            launch_label_frame, image=self.info_icon,
            bg=self.colors.theme["primary_background"]
        )
        info_label.pack(side=tk.LEFT, padx=(5, 0))  # small gap between text and icon

        # Tooltip on hover
        Tooltip(info_label, "If your microscope has a dual-view mode, you can active this button to enable the use of the scan with it. (It will appear in the parameters panel)", title="Info", theme="info", posx=70, posy=-70)
        """
        # -------------------------------------------------------------------------------------------------- #
        # section with button scan length
        buttons_scan_length_frame = tk.Frame(self.main_content, bg=self.colors.theme["primary_background"])
        buttons_scan_length_frame.pack(fill=tk.BOTH, expand=True, pady=(10,10))

        # Create an inner frame to hold the two canvases side-by-side
        input_container_frame = tk.Frame(buttons_scan_length_frame, bg=self.colors.theme["primary_background"])
        # Center this inner frame horizontally
        input_container_frame.pack(pady=0)

        # Create the first canvas (size_scan_height) and pack it to the left
        _, entry = self.create_rounded_input(
            input_container_frame, self.scan_height_length, bg="machine_config_button", width=100
        )
        self.size_scan_height_canvas = entry.master
        self.size_scan_height_canvas.pack(side=tk.LEFT, padx=10) # Use side=tk.LEFT and padx for spacing

        # Create the second canvas (size_scan_width) and pack it to the left
        _, entry = self.create_rounded_input(
            input_container_frame, self.scan_width_length, bg="machine_config_button", width=100
        )
        self.size_scan_width_canvas = entry.master
        self.size_scan_width_canvas.pack(side=tk.LEFT, padx=10) # It will appear to the right of the first


        # Container to hold label + info icon
        scan_length_label_frame = tk.Frame(buttons_scan_length_frame, bg=self.colors.theme["primary_background"])
        scan_length_label_frame.pack()

        # Text label
        title_scan = tk.Label(
            scan_length_label_frame, text="Manage the scan area (default values : 26 000 ; 45 000)",
            bg=self.colors.theme["primary_background"], fg=self.colors.theme["tertiary_text"],
            font=(self.font, 10)
        )
        title_scan.pack(side=tk.LEFT)

        # Info icon
        info_scan_label = tk.Label(
            scan_length_label_frame, image=self.info_icon,
            bg=self.colors.theme["primary_background"]
        )
        info_scan_label.pack(side=tk.LEFT, padx=(5, 0))  # small gap between text and icon

        # Tooltip on hover
        Tooltip(info_scan_label, "When you let the microscope scan the entire slide, it will use the above value to determine the width and height of the scan (in the microscope's system unit). When the 'Square' option is used, its edge lengthis the width. When the 'Rectangle' option is used, the 2 values define the rectangle's shape.", title="Info", theme="info", posx=70, posy=-70)




        # -------------------------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------- #
        # section with button size of the objective on the microscope
        if MICROSCOPE != "Macrozoom":
            buttons_objective_size = tk.Frame(self.main_content, bg=self.colors.theme["primary_background"])
            buttons_objective_size.pack(fill=tk.BOTH, expand=True, pady=(10,50))

            # Create an inner frame to hold the canvas
            input_container_size_objective_frame = tk.Frame(buttons_objective_size, bg=self.colors.theme["primary_background"])
            # Center this inner frame horizontally
            input_container_size_objective_frame.pack(pady=0)

            # Create the first canvas (size_scan_height) and pack it to the left
            _, entry = self.create_rounded_input(
                input_container_size_objective_frame, self.microscope_objective_size_1, bg="machine_config_button", width=80
            )
            entry.master.pack(side=tk.LEFT, padx=5) 
            
            # Create the 2nd canvas (size_scan_height) and pack it to the left
            _, entry = self.create_rounded_input(
                input_container_size_objective_frame, self.microscope_objective_size_2, bg="machine_config_button", width=80
            )
            entry.master.pack(side=tk.LEFT, padx=5) 
            
            # Create the 3th canvas (size_scan_height) and pack it to the left
            _, entry = self.create_rounded_input(
                input_container_size_objective_frame, self.microscope_objective_size_3, bg="machine_config_button", width=80
            )
            entry.master.pack(side=tk.LEFT, padx=5) 
            
            # Create the 4th canvas (size_scan_height) and pack it to the left
            _, entry = self.create_rounded_input(
                input_container_size_objective_frame, self.microscope_objective_size_4, bg="machine_config_button", width=80
            )
            entry.master.pack(side=tk.LEFT, padx=5) 
            
            # Create the 5th canvas (size_scan_height) and pack it to the left
            _, entry = self.create_rounded_input(
                input_container_size_objective_frame, self.microscope_objective_size_5, bg="machine_config_button", width=80
            )
            entry.master.pack(side=tk.LEFT, padx=5) 
            
            # Create the 6th canvas (size_scan_height) and pack it to the left
            _, entry = self.create_rounded_input(
                input_container_size_objective_frame, self.microscope_objective_size_6, bg="machine_config_button", width=80
            )
            entry.master.pack(side=tk.LEFT, padx=5)


            # Container to hold label + info icon
            microscope_objective_size_label_frame = tk.Frame(buttons_objective_size, bg=self.colors.theme["primary_background"])
            microscope_objective_size_label_frame.pack()

            # Text label
            title_objective_size = tk.Label(
                microscope_objective_size_label_frame, text="Manage the magnifications on your microscope",
                bg=self.colors.theme["primary_background"], fg=self.colors.theme["tertiary_text"],
                font=(self.font, 10)
            )
            title_objective_size.pack(side=tk.LEFT)

            # Info icon
            info_objective_size_label = tk.Label(
                microscope_objective_size_label_frame, image=self.info_icon,
                bg=self.colors.theme["primary_background"]
            )
            info_objective_size_label.pack(side=tk.LEFT, padx=(5, 0))  # small gap between text and icon

            # Tooltip on hover
            Tooltip(info_objective_size_label, "Enter the magnification of each objective you have on your microscope.", title="Info", theme="info", posx=70, posy=-70)
        
    def show_loading_page(self):
        """
        Page displayed when the microscope is still initializing and the application is not ready.
        """
        # Clear previous widgets if needed
        for widget in self.main_content.winfo_children():
            widget.destroy()

        # Text section
        bottom_frame = tk.Frame(self.main_content, bg=self.colors.theme["primary_background"])
        bottom_frame.pack(fill=tk.X, pady=(330,5))
        
        # Container to hold message
        launch_label_frame = tk.Frame(bottom_frame, bg=self.colors.theme["primary_background"])
        launch_label_frame.pack()

        # Text label
        title_launch_scan = tk.Label(
            launch_label_frame, text="Microscope is initializing, please wait and restart the program.",
            bg=self.colors.theme["primary_background"], fg=self.colors.theme["primary_text"],
            font=(self.font, 20)
        )
        title_launch_scan.pack(side=tk.LEFT)

        # Trigger resizing after layout completes with error handling
        try:
            if hasattr(self, 'main_content') and self.main_content.winfo_exists():
                after_id = self.main_content.after(100, self.resize_scan_content_area)
                if not hasattr(self, '_after_ids'):
                    self._after_ids = []
                self._after_ids.append(after_id)
        except:
            pass
    
    
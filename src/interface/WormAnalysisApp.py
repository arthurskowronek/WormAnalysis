
import cv2
import yaml
import numpy as np
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageColor

from config import RESSOURCES_DIR, PARAMETERS_FILE, load_config_file

from src.interface.Tooltip import Tooltip
from src.interface.colorTheme import ColorTheme
from src.system.ScanSlice import ScanSlice
from src.system.Worm_Position_Manager import WormPositionManager

class WormAnalysisApp:
    def __init__(self, root, mmc = None, initial_dark_mode=False, first_page = "automatic_scan", initial_show_parameters = True):
        """
        Initializes the Worm Analysis Application interface.

        Parameters:
            root (tk.Tk): The root window of the Tkinter application.
            initial_dark_mode (bool): If True, enables dark mode by default. Defaults to False.
            first_page (str): The initial page to display. Options include:
                              "automatic_scan", "scan_result", "assist_acquisition", "load_position", "documentation", "tutorial", "machine_config".
                              Defaults to "automatic_scan".
            initial_show_parameters (bool): If True, shows parameters section on startup. Defaults to True.

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
        self.root.geometry("1440x960")
        self.PARAMS_FILE = PARAMETERS_FILE

        # Initialize variables
        self.show_parameters = initial_show_parameters
        self.current_page = first_page
        self.dark_mode = initial_dark_mode
        self.worms_position = None
        self.prediction = 85 # TODO
        self.proportion_mutation = 10 # TODO
        self.id_worm_seen = 1 # TODO
        self.nb_of_worm = 26 # TODO
        self.add_worm_scan_result = True
        self.bounding_box_size = 15 # Size of the bounding box around worms in pixels
        self.loaded_params = load_config_file()
        self.set_parameters()
        self.enable_parameters_buttons = ["exposure_time","binning","shutter","dual_view","display_mode","scan_objective","fluo_objective","scan_shape"]

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
        elif self.current_page == "assist_acquisition":
            self.show_assist_acquisition_page()
        elif self.current_page == "load_position":
            self.show_load_position_page()
        elif self.current_page == "documentation":
            self.show_placeholder_page(self.current_page.replace('_', ' ').title())
        elif self.current_page == "tutorial":
            self.show_placeholder_page(self.current_page.replace('_', ' ').title())
        elif self.current_page == "configuration":
            self.show_placeholder_page(self.current_page.replace('_', ' ').title())
    
    # Initalization helper function
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
        self.shape = tk.StringVar(value=self.loaded_params.get("shape", "square"))
        self.shape.trace_add("write", lambda *args: self.resize_scan_content_area())
        self.shape.trace_add("write", lambda *args: self.save_parameters())
        
        self.exposure_time = tk.StringVar(value=self.loaded_params.get("exposure_time", 100))
        self.exposure_time.trace_add("write", lambda *args: self.save_parameters())
        
        self.binning = tk.StringVar(value=self.loaded_params.get("binning", "2x2"))
        self.binning.trace_add("write", lambda *args: self.save_parameters())
        
        self.shutter = tk.BooleanVar(value=self.loaded_params.get("shutter", False))
        self.shutter.trace_add("write", lambda *args: self.save_parameters())
        
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
                
    def save_parameters(self):
        """
        Updates only the first 9 lines of the parameters YAML file
        with current application parameters.
        """
        # New parameters to update
        params = {
            "exposure_time": self.exposure_time.get(),
            "binning": self.binning.get(),
            "shutter": self.shutter.get(),
            "dual_view": self.dual_view.get(),
            "display_mode": self.display_mode.get(),
            "scan_objective": self.scan_objective.get(),
            "fluo_objective": self.fluo_objective.get(),
            "shape": self.shape.get(),
            "user_directory": self.user_directory.get()
        }

        # Convert new parameters to YAML lines
        new_lines = yaml.dump(params, default_flow_style=False).splitlines(keepends=True)

        # Read existing file lines
        try:
            with open(self.PARAMS_FILE, "r") as f:
                old_lines = f.readlines()
        except FileNotFoundError:
            old_lines = []

        # Replace first 9 lines with new ones, preserving the rest
        updated_lines = new_lines[:9] + old_lines[9:]

        # Write updated lines back
        with open(self.PARAMS_FILE, "w") as f:
            f.writelines(updated_lines)
 
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
        scan_path = Path(RESSOURCES_DIR) / "icon" / "scan.png" 
        self.scan_icon = self.flatten_and_resize_icon(scan_path, 18, 18, self.colors.theme["primary_background"], self.colors.theme["icon"])
        self.scan_icon_hover = self.flatten_and_resize_icon(scan_path, 18, 18, self.colors.theme["secondary_background"], self.colors.theme["icon"])
        
        # Process validation.png
        validation_path = Path(RESSOURCES_DIR) / "icon" / "validation.png" 
        self.validation_icon = self.flatten_and_resize_icon(validation_path, 18, 18, self.colors.theme["primary_background"], self.colors.theme["icon"])
        self.validation_icon_hover = self.flatten_and_resize_icon(validation_path, 18, 18, self.colors.theme["secondary_background"], self.colors.theme["icon"])
        
        # Process load.png
        load_path = Path(RESSOURCES_DIR) / "icon" / "load.png" 
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
           
        # Process plus.png
        plus_path = Path(RESSOURCES_DIR) / "icon" / "plus.png" 
        self.plus_icon = self.flatten_and_resize_icon(plus_path, 60, 60, self.colors.theme["secondary_background"], self.colors.theme["stroke_button"])
        self.plus_icon_hover = self.flatten_and_resize_icon(plus_path, 60, 60, self.colors.theme["tertiary_background"], self.colors.theme["stroke_button"])
        
        # Process live.png
        live_path = Path(RESSOURCES_DIR) / "icon" / "live.png" 
        self.live_icon = self.flatten_and_resize_icon(live_path, 40, 40, self.colors.theme["primary_background"], self.colors.theme["stroke_button"])
        self.live_icon_hover = self.flatten_and_resize_icon(live_path, 40, 40, self.colors.theme["secondary_background"], self.colors.theme["stroke_button"])
        
        # Process snap.png
        snap_path = Path(RESSOURCES_DIR) / "icon" / "snap.png" 
        self.snap_icon = self.flatten_and_resize_icon(snap_path, 40, 40, self.colors.theme["primary_background"], self.colors.theme["stroke_button"])
        self.snap_icon_hover = self.flatten_and_resize_icon(snap_path, 40, 40, self.colors.theme["secondary_background"], self.colors.theme["stroke_button"])
        
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
        self.next_icon = self.flatten_and_resize_icon(next_path, 40, 40, self.colors.theme["primary_background"], self.colors.theme["stroke_button"])
        self.next_icon_hover = self.flatten_and_resize_icon(next_path, 40, 40, self.colors.theme["secondary_background"], self.colors.theme["stroke_button"])
        
        # Process last.png
        last_path = Path(RESSOURCES_DIR) / "icon" / "last.png" 
        self.last_icon = self.flatten_and_resize_icon(last_path, 40, 40, self.colors.theme["primary_background"], self.colors.theme["stroke_button"])
        self.last_icon_hover = self.flatten_and_resize_icon(last_path, 40, 40, self.colors.theme["secondary_background"], self.colors.theme["stroke_button"])
        
        # Process add_worm.png
        add_worm_path = Path(RESSOURCES_DIR) / "icon" / "add_worm.png" 
        self.add_worm_icon = self.flatten_and_resize_icon(add_worm_path, 30, 30, self.colors.theme["primary_background"], self.colors.theme["stroke_button"])
        self.add_worm_icon_hover = self.flatten_and_resize_icon(add_worm_path, 30, 30, self.colors.theme["secondary_background"], self.colors.theme["stroke_button"])
        
        # Process remove_worm.png
        remove_worm_path = Path(RESSOURCES_DIR) / "icon" / "remove_worm.png" 
        self.remove_worm_icon = self.flatten_and_resize_icon(remove_worm_path, 30, 30, self.colors.theme["primary_background"], self.colors.theme["stroke_button"])
        self.remove_worm_icon_hover = self.flatten_and_resize_icon(remove_worm_path, 30, 30, self.colors.theme["secondary_background"], self.colors.theme["stroke_button"])
                                                  
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

        # Resize while preserving aspect ratio
        img_pil_resized = img_pil.copy()
        img_pil_resized.thumbnail((width, height), Image.LANCZOS)

        # Separate alpha channel
        r, g, b, alpha = img_pil_resized.split()

        # Create a new solid image with the desired foreground color (primary_text)
        fg_rgb = ImageColor.getrgb(fg_color)  # Converts "#FFFFFF" → (255, 255, 255)
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

        return ImageTk.PhotoImage(background)

    # Create global interface
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
        title_label = tk.Label(top_frame, text="Worm Analysis", bg=self.colors.theme["primary_background"], fg=self.colors.theme["primary_text"],
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
            ("Scan result", "scan_result", self.validation_icon, self.validation_icon_hover),
            ("Assist Acquisition", "assist_acquisition", self.modify_icon, self.modify_icon_hover)
        ])
        
        self.create_menu_section("Analysis", [
            ("Load last position", "load_position", self.loading_icon, self.loading_icon_hover)
        ])
        
        self.create_menu_section("Help", [
            ("Documentation", "documentation", self.page_icon, self.page_icon_hover),
            ("Tutorial", "tutorial", self.question_icon, self.question_icon_hover),
            ("Machine Config", "configuration", self.machine_parameters_icon, self.machine_parameters_icon_hover)
        ])
        
        # Quit button at bottom       
        self.create_rounded_button(
            parent=self.sidebar,
            text="Quit",
            icon=self.quit_icon,
            icon_hover=self.quit_icon_hover,
            command=self.root.quit,
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

        self.name_directory_entry = self.create_rounded_input(
            self.params_frame, self.user_directory
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
        # Exposure time
        bg = "parameters_button_background" if "exposure_time" in self.enable_parameters_buttons else "parameters_button_disabled_background"
        icon = self.clock_icon if "exposure_time" in self.enable_parameters_buttons else self.clock_icon_disabled
        tk.Label(self.params_content_frame, text="Exposure time (ms)", bg=self.colors.theme["secondary_background"], fg=self.colors.theme["secondary_text"], font=(self.font, 10)).pack(anchor='w', pady=(5, 0))
        _, self.exposure_time_entry = self.create_rounded_input_with_icon(
            self.params_content_frame, self.exposure_time, icon, bg
        )
        
        # Binning
        bg = "parameters_button_background" if "binning" in self.enable_parameters_buttons else "parameters_button_disabled_background"
        tk.Label(self.params_content_frame, text="Binning", bg=self.colors.theme["secondary_background"], fg=self.colors.theme["secondary_text"], font=(self.font, 10)).pack(anchor='w', pady=(5, 0))
        _, self.binning_dropdown = self.create_rounded_dropdown(
            self.params_content_frame, ["2x2", "3x3"], self.binning, bg
        )

        # Shutter toggle
        self.shutter_toggle = self.create_custom_toggle(self.params_content_frame, "Shutter", self.shutter)

        # Dual view
        self.dual_view_toggle = self.create_custom_toggle(self.params_content_frame, "Dual view", self.dual_view)

        # Display mode
        bg = "parameters_button_background" if "display_mode" in self.enable_parameters_buttons else "parameters_button_disabled_background"
        tk.Label(self.params_content_frame, text="Display mode", bg=self.colors.theme["secondary_background"], fg=self.colors.theme["secondary_text"], font=(self.font, 10)).pack(anchor='w', pady=(5, 0))
        _, self.display_mode_dropdown = self.create_rounded_dropdown(
            self.params_content_frame, ["Grayscale"], self.display_mode, bg
        )
        
        # Scan Objective
        bg = "parameters_button_background" if "scan_objective" in self.enable_parameters_buttons else "parameters_button_disabled_background"
        tk.Label(self.params_content_frame, text="Scan Objective", bg=self.colors.theme["secondary_background"], fg=self.colors.theme["secondary_text"], font=(self.font, 10)).pack(anchor='w', pady=(5, 0))
        _, self.scan_objective_dropdown = self.create_rounded_dropdown(
            self.params_content_frame, ["4x", "5x", "10x"], self.scan_objective, bg
        )

        # Fluo objective
        bg = "parameters_button_background" if "fluo_objective" in self.enable_parameters_buttons else "parameters_button_disabled_background"
        tk.Label(self.params_content_frame, text="Fluo objective", bg=self.colors.theme["secondary_background"], fg=self.colors.theme["secondary_text"], font=(self.font, 10)).pack(anchor='w', pady=(5, 0))
        _, self.fluo_objective_dropdown = self.create_rounded_dropdown(
            self.params_content_frame, ["10x", "20x", "40x"], self.fluo_objective, bg
        )

        # Scan shape
        bg = "parameters_button_background" if "scan_shape" in self.enable_parameters_buttons else "parameters_button_disabled_background"
        tk.Label(self.params_content_frame, text="Scan shape", bg=self.colors.theme["secondary_background"], fg=self.colors.theme["secondary_text"], font=(self.font, 10)).pack(anchor='w', pady=(5, 0))
        _, self.scan_shape_dropdown = self.create_rounded_dropdown(
            self.params_content_frame, ["square", "rectangle"], self.shape, bg
        )
    
    # Button
    def create_rounded_input(self, parent, variable, bg = "parameters_button_background"): 
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
        canvas_width = 190 
        canvas_height = 35
        radius = 20 

        # Create the canvas
        canvas = tk.Canvas(parent, width=canvas_width, height=canvas_height,
                           bg=parent.cget("bg"), highlightthickness=0) # Use parent's bg for canvas
        canvas.pack(fill=tk.X, pady=(0, 15), padx=20)

        # Draw the rounded background
        self.draw_rounded_rect(canvas, 0, 0, canvas_width, canvas_height,
                               radius, fill=self.colors.theme[bg],
                               outline=self.colors.theme[bg], tag="input_bg")

        # Create the Entry widget        
        entry = tk.Entry(canvas, textvariable=variable, font=(self.font, 10), bd=0, relief="flat", highlightthickness=0,
                 bg=self.colors.theme[bg], fg=self.colors.theme["tertiary_text"],
                 insertbackground=self.colors.theme["primary_text"])

        # Place the entry widget inside the canvas. Adjust x, y for padding.
        entry_width = canvas_width - 2 * radius # Approximate width of the entry part
        entry_height = canvas_height - 10 # Approximate height of the entry part
        canvas.create_window(radius, canvas_height / 2, window=entry, anchor="w",
                             width=entry_width, height=entry_height)
        return variable
    
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
                        bg=parent.cget("bg"), highlightthickness=0)
        canvas.pack(fill=tk.X, pady=(0, 0))

        # Draw the background
        self.draw_rounded_rect(canvas, 0, 0, canvas_width, canvas_height,
                            radius, fill=self.colors.theme[bg],
                            outline=self.colors.theme[bg], tag="input_bg")

        # Add the icon
        if isinstance(icon, str):
            # It's a text/emoji icon
            tk.Label(canvas, text=icon, bg=self.colors.theme[bg],
                    fg=self.colors.theme["secondary_text"], font=(self.font, 12)).place(x=5, rely=0.5, anchor="w")
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
                        bg=parent.cget("bg"), highlightthickness=0)
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
    
    def create_custom_toggle(self, parent, label, boolean_var):
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
        frame = tk.Frame(parent, bg=self.colors.theme["secondary_background"])
        frame.pack(fill=tk.X, pady=(5, 5))

        # Add the label of the toggle
        tk.Label(frame, text=label, bg=self.colors.theme["secondary_background"],
                fg=self.colors.theme["secondary_text"], font=(self.font, 10)).pack(side=tk.LEFT)

        # Create the toggle
        toggle_canvas = tk.Canvas(frame, width=self.toggle_open_icon.width(),
                                height=self.toggle_open_icon.height(),
                                bg=self.colors.theme["primary_background"], highlightthickness=0)
        toggle_canvas.pack(side=tk.RIGHT, padx=3)

        def draw_toggle():
            toggle_canvas.delete("all")
            image = self.toggle_open_icon if not boolean_var.get() else self.toggle_close_icon
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
                          anchor='center', border_width=0, border_color=None, icon=None, icon_hover=None):
        """Creates a stylized button with rounded corners and optional icon.

        The button is rendered on a canvas and supports hover effects, click binding,
        and image swapping when hovered. It can display text only or an icon with text.

        Args:
            parent (tk.Widget): The parent widget for the button.
            text (str): The button label.
            command (Callable): The function to execute on click.
            bg_color (str): The background color of the button.
            text_color (str): The text color.
            hover_color (str): The background color on hover.
            font (Tuple): Font tuple for the button text.
            width_pixels (int): Width of the button in pixels.
            height_pixels (int): Height of the button in pixels.
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
            highlightthickness=0
        )
        canvas.pack(side=side, padx=padx, pady=pady)

        # Get coordinates in which to draw the button
        x1, y1 = 0, 0
        x2, y2 = width_pixels, height_pixels

        # Draw border
        self.draw_rounded_rect(
            canvas,
            x1, y1,
            x2, y2,
            corner_radius,
            fill=border_color,
            outline=border_color,
            tag="button_border"
        )

        # Draw main shape inset by border_width
        inset = border_width        
        self.draw_rounded_rect(
            canvas,
            x1 + inset, y1 + inset,
            x2 - inset, y2 - inset,
            max(corner_radius - inset, 0),
            fill=bg_color,
            outline=bg_color,
            tag="button_shape"
        )

        # Build label (icon + text or text-only)
        if icon:
            label_frame = tk.Frame(canvas, bg=bg_color)

            icon_label = tk.Label(label_frame, image=icon, bg=bg_color)
            icon_label.image = icon
            if icon_hover:
                icon_label.image_normal = icon
                icon_label.image_hover = icon_hover
            icon_label.pack(side=tk.LEFT, padx=(0, 5))

            text_label = tk.Label(label_frame, text=text, bg=bg_color, fg=text_color, font=font)
            text_label.pack(side=tk.LEFT)

            label_widget = label_frame
        else:
            text_label = tk.Label(canvas, text=text, bg=bg_color, fg=text_color, font=font)
            label_widget = text_label

        # Place the label on the canvas
        canvas.create_window(
            width_pixels / 2 - padx_text,
            height_pixels / 2 - pady_text,
            window=label_widget,
            anchor=anchor,
            tags="button_label"
        )

        # Event handlers
        def on_enter(event):
            canvas.itemconfig("button_shape", fill=hover_color, outline=hover_color)
            if icon:
                icon_label.config(bg=hover_color)
                if icon_hover:
                    icon_label.config(image=icon_hover)
                label_frame.config(bg=hover_color)
            text_label.config(bg=hover_color)

        def on_leave(event):
            canvas.itemconfig("button_shape", fill=bg_color, outline=bg_color)
            if icon:
                icon_label.config(bg=bg_color)
                if icon_hover:
                    icon_label.config(image=icon)
                label_frame.config(bg=bg_color)
            text_label.config(bg=bg_color)

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

    # Command
    def refresh_ui(self):        
        self.root.configure(bg=self.colors.theme["primary_background"])
        for widget in self.root.winfo_children():
            widget.destroy()
        self.__init__(self.root, self.CORE, self.dark_mode, self.current_page, self.show_parameters)
  
    def refresh_parameters_interface(self):
        if hasattr(self, "params_frame"):
            self.params_frame.destroy()
        self.create_parameters_panel()
        
    def update_parameter_widgets_state(self, disabled_widgets):
        all_widgets = {
            "exposure_time": self.exposure_time_entry,
            "binning": self.binning_dropdown,
            "shutter": self.shutter_toggle,
            "dual_view": self.dual_view_toggle,
            "display_mode": self.display_mode_dropdown,
            "scan_objective": self.scan_objective_dropdown,
            "fluo_objective": self.fluo_objective_dropdown,
            "scan_shape": self.scan_shape_dropdown
        }
        
        for key, widget in all_widgets.items():
            if key in disabled_widgets:
                if isinstance(widget, tk.Canvas):  # Handle custom toggle
                    widget.unbind("<Button-1>")
                else:
                    widget.configure(state="disabled")
            else:
                if isinstance(widget, ttk.Combobox):
                    widget.configure(state="readonly")
                elif isinstance(widget, tk.Canvas):
                    widget.bind("<Button-1>", widget.toggle_command)
                else:
                    widget.configure(state="normal")
        
        # Store only enabled widget keys
        self.enable_parameters_buttons = [key for key in all_widgets if key not in disabled_widgets]

    def toggle_dark_mode(self):
        self.dark_mode = not self.dark_mode
        self.update_colors()
        self.refresh_ui()

    def toggle_parameters(self):
        self.show_parameters = not self.show_parameters
        if self.show_parameters:
            self.params_frame.pack(side=tk.RIGHT, fill=tk.Y)
        else:
            self.params_frame.pack_forget()
        
        # Store the after_id and schedule resizing with error handling
        if hasattr(self, 'main_content') and self.main_content.winfo_exists() and (self.current_page == "automatic_scan" or self.current_page == "scan_result"):
            after_id = self.main_content.after(50, self.resize_scan_content_area)
            if not hasattr(self, '_after_ids'):
                self._after_ids = []
            self._after_ids.append(after_id)
    
    def toggle_add_worm_scan_result(self):
        self.add_worm_scan_result = not self.add_worm_scan_result   
        self.show_result_scan_page()
    
    def switch_page(self, page_id):
        self.current_page = page_id
        self.refresh_ui()
    
    def resize_scan_content_area(self):
        if self.current_page == "automatic_scan":
            middle_container = self.middle_container_ref
            content_area = self.content_area_ref
        elif self.current_page == "scan_result":
            middle_container = self.middle_result_container_ref
            content_area = self.content_area_result_container_ref

        container_width = middle_container.winfo_width()
        container_height = middle_container.winfo_height()

        if self.shape.get() == 'square':
            side = min(container_width, container_height)
            width = height = side
        elif self.shape.get() == 'rectangle':
            height = min(container_height, container_width / 2)
            width = 2 * height
        else:
            height = min(container_height, container_width)
            width = height

        x = (container_width - width) / 2
        y = (container_height - height) / 2

        content_area.place(x=x, y=y, width=width, height=height)
        
        self.last_scan_area_size = (int(width), int(height))
        
        # --- Resize image accordingly ---
        if hasattr(self, 'original_image') and hasattr(self, 'img_label') and self.img_label.winfo_exists():
            resized_img = self.original_image.resize(self.last_scan_area_size)
            photo = ImageTk.PhotoImage(resized_img)
            self.displayed_image = photo
            self.img_label.configure(image=photo)
        
    def resize_live_image(self, event):
        w, h = event.width, event.height
        size = min(w, h - 80)  # leave space for bottom button
        x = (w - size) // 2
        if self.current_page == "assist_acquisition":
            self.live_assist_container_ref.place(x=x, y=0, width=size, height=size)
        elif self.current_page == "load_position":
            self.live_analysis_container_ref.place(x=x, y=0, width=size, height=size)
    
    def resize_map_assist(self, event):
        w, h = event.width, event.height
        size = min(w, h) 
        x = (w - size) // 2
        y = h - size - 10  # 10 px from bottom
        self.map_assist_containter_ref.place(x=x, y=y, width=size, height=size)
       
    def resize_prediction_result_box(self, event):
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
       
    def launch_scan(self):
        scanner = ScanSlice(self.CORE, self.scan_objective, self.dual_view, self.shape) # create an instance
        worms_microscope_position = scanner.scan() # launch scan and get the microscope position of the worms
        self.worms_position = WormPositionManager(table_worm_position = worms_microscope_position) # save the positions
        scanner.reconstruct_slice() # get the reconstructed slice
        self.switch_page("scan_result") # switch to the result page
      
    def draw_prediction_result_box(self):
        # Load original image
        image = Image.open(Path(RESSOURCES_DIR) / "stitched_final.jpg")
        
        # Convert to numpy array
        img_with_bounding_box_np = np.array(image)
        
        # Convert to color
        img_with_bounding_box_np = cv2.cvtColor(img_with_bounding_box_np, cv2.COLOR_GRAY2BGR)
        
        # Get worms positions
        if self.worms_position is None:
            self.worms_position = WormPositionManager(new_acquisition=False)
            all_worm_data = self.worms_position.get_all_worm_proportion_position()
            list_of_worm_position = [[x, y] for worm_id, x, y in all_worm_data]
        else:
            all_worm_data = self.worms_position.get_all_worm_proportion_position()
            list_of_worm_position = [[x, y] for worm_id, x, y in all_worm_data]
          
        # Draw bounding boxes
        img_height, img_width = img_with_bounding_box_np.shape[:2]      
        for worm in list_of_worm_position:
            x = int(worm[0] * img_width)
            y = int(worm[1] * img_height)
            box = (x - self.bounding_box_size, y - self.bounding_box_size, x + self.bounding_box_size, y + self.bounding_box_size)  # (x1, y1, x2, y2)
            cv2.rectangle(img_with_bounding_box_np, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
            
        # Convert back to PIL Image
        img_with_bounding_box_np = cv2.cvtColor(img_with_bounding_box_np, cv2.COLOR_BGR2RGB)
        self.original_image = Image.fromarray(img_with_bounding_box_np)
        
        # Create placeholder image for display
        placeholder_img = self.original_image.resize((10, 10))
        img_with_bounding_box = ImageTk.PhotoImage(placeholder_img)
        
        return img_with_bounding_box
        
    def on_stitching_image_click(self, event):
        # Get clicked coordinates in displayed image
        x_display, y_display = event.x, event.y

        # Get displayed image size
        display_width = self.img_label.winfo_width()
        display_height = self.img_label.winfo_height()
        
        # Compute relative position
        x_mouse = float(x_display / display_width)
        y_mouse = float(y_display / display_height)
        x_bounding_box_proportion = float(self.bounding_box_size / display_width)
        y_bounding_box_proportion = float(self.bounding_box_size / display_height)

        # Check if click is inside any bounding box
        if not self.add_worm_scan_result:
            for _, (id, x, y) in enumerate(self.worms_position.get_all_worm_proportion_position()):
                if x-x_bounding_box_proportion <= x_mouse <= x+x_bounding_box_proportion and y-y_bounding_box_proportion <= y_mouse <= y+y_bounding_box_proportion:
                    # Remove the worm
                    self.worms_position.delete_worm(id)
                    break
        else:
            # Add a new worm
            x_microscope, y_microscope = self.worms_position.transform_proportion_into_microscope_positions(x_mouse, y_mouse)
            self.worms_position.add_worm_microscope_position(x_microscope, y_microscope)
            
        # Redraw image with updated worm positions
        updated_img = self.draw_prediction_result_box()
        self.displayed_image = updated_img
        self.img_label.configure(image=updated_img)
        self.img_label.image = updated_img  # Prevent image from being garbage collected
        self.resize_scan_content_area()

    def on_stitching_image_drag(self, event):
        if not self.add_worm_scan_result:
            x_display, y_display = event.x, event.y
            display_width = self.img_label.winfo_width()
            display_height = self.img_label.winfo_height()

            x_mouse = float(x_display / display_width)
            y_mouse = float(y_display / display_height)
            x_bounding_box_proportion = float(self.bounding_box_size / display_width)
            y_bounding_box_proportion = float(self.bounding_box_size / display_height)

            removed = False
            for _, (id, x, y) in enumerate(self.worms_position.get_all_worm_proportion_position()):
                if x - x_bounding_box_proportion <= x_mouse <= x + x_bounding_box_proportion and \
                y - y_bounding_box_proportion <= y_mouse <= y + y_bounding_box_proportion:
                    self.worms_position.delete_worm(id)
                    removed = True
                    break

            if removed:
                updated_img = self.draw_prediction_result_box()
                self.displayed_image = updated_img
                self.img_label.configure(image=updated_img)
                self.img_label.image = updated_img
                self.resize_scan_content_area()
         
    # Pages   
    def show_automatic_scan_page(self):
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

        # Bottom section with launch button
        bottom_frame = tk.Frame(self.main_content, bg=self.colors.theme["primary_background"])
        bottom_frame.pack(fill=tk.X, pady=(20,5))

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
        
        # Container to hold label + info icon
        launch_label_frame = tk.Frame(bottom_frame, bg=self.colors.theme["primary_background"])
        launch_label_frame.pack()

        # Text label
        title_launch_scan = tk.Label(
            launch_label_frame, text="Launch scan",
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
        Tooltip(info_label, "Be sure to have the objective in the lower right corner and to use the L camera.", posx=70, posy=-70)

        # Trigger resizing after layout completes with error handling
        if hasattr(self, 'main_content') and self.main_content.winfo_exists():
            after_id = self.main_content.after(100, self.resize_scan_content_area)
            if not hasattr(self, '_after_ids'):
                self._after_ids = []
            self._after_ids.append(after_id)
    
    def show_result_scan_page(self):
        # Clear previous widgets if needed
        for widget in self.main_content.winfo_children():
            widget.destroy()
            
        # Disable some paramaters buttons 
        self.update_parameter_widgets_state(disabled_widgets=["exposure_time","binning","shutter","dual_view","display_mode","scan_objective","fluo_objective","scan_shape"]) 
        self.refresh_parameters_interface()
        self.update_parameter_widgets_state(disabled_widgets=["exposure_time","binning","shutter","dual_view","display_mode","scan_objective","fluo_objective","scan_shape"])
        
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
        add_bg = self.colors.theme["secondary_background"] if self.add_worm_scan_result else self.colors.theme["primary_background"]
        remove_bg = self.colors.theme["secondary_background"] if not self.add_worm_scan_result else self.colors.theme["primary_background"]
        add_icon = self.add_worm_icon if not self.add_worm_scan_result else self.add_worm_icon_hover
        rmeove_icon = self.remove_worm_icon if self.add_worm_scan_result else self.remove_worm_icon_hover
        
        # Add worm button
        self.create_rounded_button(
            parent=button_row_frame,
            text="",
            icon=add_icon,
            icon_hover=self.add_worm_icon_hover,
            command=lambda: self.toggle_add_worm_scan_result(),
            bg_color=add_bg,
            text_color=self.colors.theme["primary_text"],
            hover_color=self.colors.theme["secondary_background"],
            font=(self.font, 16),
            width_pixels=100,
            height_pixels=60,
            corner_radius=20,
            side=tk.LEFT, 
            pady=5,
            padx=30,
            padx_text=-6,
            border_width=2,
            border_color=self.colors.theme["stroke_button"]
        )
        
        # Start analysis button
        self.create_rounded_button(
            parent=button_row_frame,
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
            side=tk.LEFT,
            pady=5,
            padx=30,
            padx_text=-10,
            border_width=2,
            border_color=self.colors.theme["stroke_button"]
        )
        
        # Remove worm button
        self.create_rounded_button(
            parent=button_row_frame,
            text="",
            icon=rmeove_icon,
            icon_hover=self.remove_worm_icon_hover,
            command=lambda: self.toggle_add_worm_scan_result(),
            bg_color=remove_bg,
            text_color=self.colors.theme["primary_text"],
            hover_color=self.colors.theme["secondary_background"],
            font=(self.font, 16),
            width_pixels=100,
            height_pixels=60,
            corner_radius=20,
            side=tk.LEFT, 
            pady=5,
            padx=30,
            padx_text=-6,
            border_width=2,
            border_color=self.colors.theme["stroke_button"]
        )
        
        # Container to hold label + info icon
        launch_label_frame_result_container = tk.Frame(bottom_frame_result_container, bg=self.colors.theme["primary_background"])
        launch_label_frame_result_container.pack()

        # Text label
        title_launch_analysis_result_container = tk.Label(
            launch_label_frame_result_container, text="Start analysis",
            bg=self.colors.theme["primary_background"], fg=self.colors.theme["tertiary_text"],
            font=(self.font, 10)
        )
        title_launch_analysis_result_container.pack(side=tk.LEFT)

        # Info icon
        info_label_result_container = tk.Label(
            launch_label_frame_result_container, image=self.info_icon,
            bg=self.colors.theme["primary_background"]
        )
        info_label_result_container.pack(side=tk.LEFT, padx=(5, 0))  # small gap between text and icon

        # Tooltip on hover
        Tooltip(info_label_result_container, "Be sure to use the L camera.", posx=70, posy=-70)

        # Trigger resizing after layout completes with error handling
        if hasattr(self, 'main_content') and self.main_content.winfo_exists():
            after_id = self.main_content.after(100, self.resize_scan_content_area)
            if not hasattr(self, '_after_ids'):
                self._after_ids = []
            self._after_ids.append(after_id)
            
        # ----- IMAGE DISPLAY -----        
        img_with_bounding_box = self.draw_prediction_result_box()
        self.displayed_image = img_with_bounding_box
        
        # Create image label and store reference
        self.img_label = tk.Label(content_area_result_container, image=img_with_bounding_box, bg=self.colors.theme["secondary_background"])
        self.img_label.pack(expand=True)
        
        # Bind click event to the image label
        self.img_label.bind("<Button-1>", self.on_stitching_image_click)
        self.img_label.bind("<B1-Motion>", self.on_stitching_image_drag)

    def show_assist_acquisition_page(self):
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
            command=lambda: self.add_worm_callback, # TODO
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
    
    def show_load_position_page(self):
        # Clear previous widgets
        for widget in self.main_content.winfo_children():
            widget.destroy()
            
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

        # Bottom: Buttons + labels
        bottom_analysis_container = tk.Frame(left_live_analysis_container, bg=self.colors.theme["primary_background"])
        bottom_analysis_container.grid(row=1, column=0, sticky="ew", pady=(10, 10))

        # --- Row that holds both button + label groups ---
        button_label_row_analysis_container = tk.Frame(bottom_analysis_container, bg=self.colors.theme["primary_background"])
        button_label_row_analysis_container.pack()

        # --- First button + label ---
        button1_analysis_container = tk.Frame(button_label_row_analysis_container, bg=self.colors.theme["primary_background"])
        button1_analysis_container.pack(side=tk.LEFT, padx=10)

        self.create_rounded_button(
            parent=button1_analysis_container,
            text="",
            icon=self.live_icon,
            icon_hover=self.live_icon_hover,
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
            padx_text=-7,
            border_width=2,
            border_color=self.colors.theme["stroke_button"]
        )

        tk.Label(
            button1_analysis_container,
            text="Start analysis",
            bg=self.colors.theme["primary_background"],
            fg=self.colors.theme["secondary_text"],
            font=(self.font, 10)
        ).pack()

        # --- Second button + label ---
        button2_analysis_container = tk.Frame(button_label_row_analysis_container, bg=self.colors.theme["primary_background"])
        button2_analysis_container.pack(side=tk.LEFT, padx=10)

        self.create_rounded_button(
            parent=button2_analysis_container,
            text="",
            icon=self.snap_icon,
            icon_hover=self.snap_icon_hover,
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
            padx_text=-7,
            border_width=2,
            border_color=self.colors.theme["stroke_button"]
        )

        tk.Label(
            button2_analysis_container,
            text="Take snapshot",
            bg=self.colors.theme["primary_background"],
            fg=self.colors.theme["secondary_text"],
            font=(self.font, 10)
        ).pack()




        # ----- RIGHT CONTAINER -----
        right_map_analysis_container = tk.Frame(self.main_content, bg=self.colors.theme["primary_background"])
        right_map_analysis_container.grid(row=0, column=1, sticky="nsew", padx=(0, 0))
        self.right_map_analysis_container_ref = right_map_analysis_container

        # Make it expand vertically
        right_map_analysis_container.grid_rowconfigure((0, 1, 2, 3, 4), weight=1)
        right_map_analysis_container.grid_columnconfigure(0, weight=1)  # left spacer
        right_map_analysis_container.grid_columnconfigure(1, weight=0)  # the container column
        right_map_analysis_container.grid_columnconfigure(2, weight=1)  # right spacer


        # 1. Label Container (fixed space with rounded border)
        self.top_label_1_analysis_container = tk.Frame(right_map_analysis_container)
        self.top_label_1_analysis_container.grid(row=0, column=1, sticky="ew", pady=(30, 10))
        self.top_label_1_analysis_container.grid_columnconfigure(1, weight=1)

        # Create a canvas inside this frame for drawing the rounded rectangle
        self.top_label_canvas = tk.Canvas(
            self.top_label_1_analysis_container, 
            height=100, 
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
            font=(self.font, 10, "bold")
        ).pack(pady=(0, 0), anchor="center")  # changed from "w" to "center"

        self.prediction_label = tk.Label(
            self.top_label_frame,
            text=f"The analysed worm is a mutant",
            bg=self.colors.theme["primary_background"],
            fg=self.colors.theme["secondary_text"],
            justify="center",  # changed from "left" to "center"
            font=(self.font, 8)
        )
        self.prediction_label.pack(pady=(5, 0), anchor="center")

        self.prediction_label_2 = tk.Label(
            self.top_label_frame,
            text=f"with a probability of {self.prediction}%",
            bg=self.colors.theme["primary_background"],
            fg=self.colors.theme["secondary_text"],
            justify="center",  # changed from "left" to "center"
            font=(self.font, 8)
        )
        self.prediction_label_2.pack(pady=(0, 0), anchor="center")

        
        self.top_label_frame_window = self.top_label_canvas.create_window(
            (0, 0), window=self.top_label_frame, anchor="center"
        )

        self.top_label_canvas.bind("<Configure>", self.resize_prediction_result_box)


        # To hide it, you can use :
        #   self.top_label_1_analysis_container.grid_remove()
        # And to show it :
        #   self.top_label_1_analysis_container.grid()

        # 2. Two Buttons with Text Below (Side by Side)
        mid_buttons_2_analysis_container = tk.Frame(right_map_analysis_container, bg=self.colors.theme["primary_background"])
        mid_buttons_2_analysis_container.grid(row=1, column=1, sticky="ew")

        # 1st
        sub1_2_analysis_container = tk.Frame(mid_buttons_2_analysis_container, bg=self.colors.theme["primary_background"])
        sub1_2_analysis_container.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=10)

        self.create_rounded_button(
            parent=sub1_2_analysis_container,
            text="",
            icon=self.wildtype_icon,
            icon_hover=self.wildtype_icon_hover,
            command=lambda: self.add_worm_callback, # TODO
            bg_color=self.colors.theme["primary_background"],
            text_color=self.colors.theme["primary_text"],
            hover_color=self.colors.theme["secondary_background"],
            font=(self.font, 14),
            width_pixels=104,
            height_pixels=70,
            corner_radius=20,
            side=tk.TOP,
            padx_text=-5,
            pady=5,
            border_width=2,
            border_color=self.colors.theme["stroke_button"]
        )

        tk.Label(sub1_2_analysis_container, text="Wild-Type", bg=self.colors.theme["primary_background"],
                fg=self.colors.theme["secondary_text"], font=(self.font, 10)).pack()
        text_proportion_wild_type = f"{100-self.proportion_mutation}%"
        tk.Label(sub1_2_analysis_container, text=text_proportion_wild_type, bg=self.colors.theme["primary_background"],
                fg=self.colors.theme["secondary_text"], font=(self.font, 7)).pack()
        
        # 2nd
        sub2_2_analysis_container = tk.Frame(mid_buttons_2_analysis_container, bg=self.colors.theme["primary_background"])
        sub2_2_analysis_container.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=10)

        self.create_rounded_button(
            parent=sub2_2_analysis_container,
            text="",
            icon=self.mutant_icon,
            icon_hover=self.mutant_icon_hover,
            command=lambda: self.add_worm_callback, # TODO
            bg_color=self.colors.theme["primary_background"],
            text_color=self.colors.theme["primary_text"],
            hover_color=self.colors.theme["secondary_background"],
            font=(self.font, 14),
            width_pixels=104,
            height_pixels=70,
            corner_radius=20,
            side=tk.TOP,
            padx_text=-5,
            pady=5,
            border_width=2,
            border_color=self.colors.theme["stroke_button"]
        )

        tk.Label(sub2_2_analysis_container, text="Mutation", bg=self.colors.theme["primary_background"],
                fg=self.colors.theme["secondary_text"], font=(self.font, 10)).pack()
        text_proportion_mutation = f"{self.proportion_mutation}%"
        tk.Label(sub2_2_analysis_container, text=text_proportion_mutation, bg=self.colors.theme["primary_background"],
                fg=self.colors.theme["secondary_text"], font=(self.font, 7)).pack()

        # 3. Text Container
        text_3_analysis_container = tk.Frame(right_map_analysis_container, bg=self.colors.theme["primary_background"])
        text_3_analysis_container.grid(row=2, column=1, sticky="ew", pady=0, ipady=0)  # Remove all padding
        text_id_worm_seen = f"{self.id_worm_seen}/{self.nb_of_worm}"
        tk.Label(
            text_3_analysis_container,
            text=text_id_worm_seen,
            bg=self.colors.theme["primary_background"],
            fg=self.colors.theme["tertiary_text"],
            font=(self.font, 10)
        ).pack(pady=0)

        # 4. Two Buttons Side by Side - Use same row to eliminate gap
        bottom_buttons_4_analysis_container = tk.Frame(text_3_analysis_container, bg=self.colors.theme["primary_background"])
        bottom_buttons_4_analysis_container.pack(side=tk.BOTTOM, pady=(5, 0))  # Remove fill=tk.X to center content

        # Create a single container for both buttons without expansion
        buttons_wrapper = tk.Frame(bottom_buttons_4_analysis_container, bg=self.colors.theme["primary_background"])
        buttons_wrapper.pack()

        # 1st
        sub1_4_analysis_container = tk.Frame(buttons_wrapper, bg=self.colors.theme["primary_background"])
        sub1_4_analysis_container.pack(side=tk.LEFT, padx=(0, 1))  # Remove expand=True and fill=tk.X
        self.create_rounded_button(
            parent=sub1_4_analysis_container,
            text="",
            icon=self.last_icon,
            icon_hover=self.last_icon_hover,
            command=lambda: print(""), # TODO
            bg_color=self.colors.theme["primary_background"],
            text_color=self.colors.theme["primary_text"],
            hover_color=self.colors.theme["secondary_background"],
            font=(self.font, 12),
            width_pixels=104,
            height_pixels=70,
            corner_radius=10,
            side=tk.TOP,
            padx=10,  
            pady=5,
            padx_text=-5,
            border_width=2,
            border_color=self.colors.theme["stroke_button"]
        )

        # 2nd
        sub2_4_analysis_container = tk.Frame(buttons_wrapper, bg=self.colors.theme["primary_background"])
        sub2_4_analysis_container.pack(side=tk.LEFT, padx=(1, 0))  # Remove expand=True and fill=tk.X
        self.create_rounded_button(
            parent=sub2_4_analysis_container,
            text="",
            icon=self.next_icon,
            icon_hover=self.next_icon_hover,
            command=lambda: print(""), # TODO
            bg_color=self.colors.theme["primary_background"],
            text_color=self.colors.theme["primary_text"],
            hover_color=self.colors.theme["secondary_background"],
            font=(self.font, 12),
            width_pixels=104,
            height_pixels=70,
            corner_radius=10,
            side=tk.TOP,
            padx=10, 
            pady=5,
            padx_text=-5,
            border_width=2,
            border_color=self.colors.theme["stroke_button"]
        )

        # 5. Button + Text with Padding
        final_5_analysis_container = tk.Frame(right_map_analysis_container, bg=self.colors.theme["primary_background"])
        final_5_analysis_container.grid(row=4, column=1, sticky="ew", pady=(10, 50))

        self.create_rounded_button(
            parent=final_5_analysis_container,
            text="",
            icon=self.play_icon,
            icon_hover=self.play_icon_hover,
            command=lambda: print("Final Action"), # TODO
            bg_color=self.colors.theme["primary_background"],
            text_color=self.colors.theme["primary_text"],
            hover_color=self.colors.theme["secondary_background"],
            font=(self.font, 12),
            width_pixels=250,
            height_pixels=60,
            corner_radius=10,
            side=tk.TOP,
            pady=5,
            padx_text=-10,
            border_width=2,
            border_color=self.colors.theme["stroke_button"]
        )

        tk.Label(
            final_5_analysis_container,
            text="Launch analysis",
            bg=self.colors.theme["primary_background"],
            fg=self.colors.theme["secondary_text"],
            font=(self.font, 10)
        ).pack()

    def show_placeholder_page(self, page_name):
        placeholder = tk.Label(self.main_content, text=f"{page_name} Page\n(Coming soon...)",
                             bg=self.colors.theme["primary_background"], fg=self.colors.theme["primary_text"], font=(self.font, 16))
        placeholder.pack(expand=True)
        
        # Disable some paramaters buttons 
        self.update_parameter_widgets_state(disabled_widgets=["exposure_time","binning","shutter","dual_view","display_mode","scan_objective","fluo_objective","scan_shape"]) 
        self.refresh_parameters_interface()
        self.update_parameter_widgets_state(disabled_widgets=["exposure_time","binning","shutter","dual_view","display_mode","scan_objective","fluo_objective","scan_shape"])
        

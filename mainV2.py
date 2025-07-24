
import os
import time
from PIL import Image, ImageTk, ImageColor
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

from config import RESSOURCES_DIR

from Tooltip import Tooltip
from colorTheme import ColorTheme

class WormAnalysisApp:
    def __init__(self, root, initial_dark_mode=False, first_page = "automatic_scan"):
        self.root = root
        self.root.title("Worm Analysis")
        self.root.geometry("1440x960")

        # Variables
        self.show_parameters = True
        self.current_page = first_page
        self.dark_mode = initial_dark_mode

        # Color themes
        self.font = 'Inter'
        self.update_colors()
        self.set_color_theme()

        self.root.configure(bg=self.colors.theme["primary_background"])
        
        # Parameters
        self.shape = "square"

        # Main container
        self.main_frame = tk.Frame(root, bg=self.colors.theme["primary_background"])
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Images
        self.load_icon()
        

        # Create main layout
        self.create_layout()

        # Show appropriate page
        if self.current_page == "automatic_scan":
            self.show_automatic_scan_page()
        elif self.current_page == "assist_acquisition":
            self.show_assist_acquisition_page()
        elif self.current_page == "load_position":
            self.show_load_position_page()
        else:
            self.show_placeholder_page(self.current_page.replace('_', ' ').title())
    
    def load_icon(self):   
        # ---------------- Parameters icon ----------------     
        # Process toggle_open.png
        open_img_path = Path(RESSOURCES_DIR) / "icon" / "toggle_open.png"
        self.toggle_open_icon = self.flatten_and_resize(open_img_path, 34, 14, self.colors.theme["secondary_background"], self.colors.theme["toggle_button"])

        # Process toggle_close.png
        close_img_path = Path(RESSOURCES_DIR) / "icon" / "toggle_close.png"
        self.toggle_close_icon = self.flatten_and_resize(close_img_path, 34, 14, self.colors.theme["secondary_background"], self.colors.theme["toggle_button"])
        
        # Process filtre.png
        filtre_path = Path(RESSOURCES_DIR) / "icon" / "filtre.png" 
        self.icon_parameter = self.flatten_and_resize(filtre_path, 18, 18, self.colors.theme["secondary_background"], self.colors.theme["icon"])
        
        # Process clock.png
        clock_path = Path(RESSOURCES_DIR) / "icon" / "clock.png" 
        self.clock_icon = self.flatten_and_resize(clock_path, 18, 18, self.colors.theme["parameters_button_background"], self.colors.theme["tertiary_text"])
        
        # ---------------- Menu icon ----------------
        # Process scan.png
        scan_path = Path(RESSOURCES_DIR) / "icon" / "scan.png" 
        self.scan_icon = self.flatten_and_resize(scan_path, 18, 18, self.colors.theme["primary_background"], self.colors.theme["icon"])
        self.scan_icon_hover = self.flatten_and_resize(scan_path, 18, 18, self.colors.theme["secondary_background"], self.colors.theme["icon"])
        
        # Process load.png
        load_path = Path(RESSOURCES_DIR) / "icon" / "load.png" 
        self.loading_icon = self.flatten_and_resize(load_path, 18, 18, self.colors.theme["primary_background"], self.colors.theme["icon"])
        self.loading_icon_hover = self.flatten_and_resize(load_path, 18, 18, self.colors.theme["secondary_background"], self.colors.theme["icon"])
        
        # Process machine_parameters.png
        machine_parameters_path = Path(RESSOURCES_DIR) / "icon" / "machine_parameters.png" 
        self.machine_parameters_icon = self.flatten_and_resize(machine_parameters_path, 18, 18, self.colors.theme["primary_background"], self.colors.theme["icon"])
        self.machine_parameters_icon_hover = self.flatten_and_resize(machine_parameters_path, 18, 18, self.colors.theme["secondary_background"], self.colors.theme["icon"])
        
        # Process modify.png
        modify_path = Path(RESSOURCES_DIR) / "icon" / "modify.png" 
        self.modify_icon = self.flatten_and_resize(modify_path, 18, 18, self.colors.theme["primary_background"], self.colors.theme["icon"])
        self.modify_icon_hover = self.flatten_and_resize(modify_path, 18, 18, self.colors.theme["secondary_background"], self.colors.theme["icon"])
        
        # Process page.png
        page_path = Path(RESSOURCES_DIR) / "icon" / "page.png" 
        self.page_icon = self.flatten_and_resize(page_path, 18, 18, self.colors.theme["primary_background"], self.colors.theme["icon"])
        self.page_icon_hover = self.flatten_and_resize(page_path, 18, 18, self.colors.theme["secondary_background"], self.colors.theme["icon"])
        
        # Process question.png
        question_path = Path(RESSOURCES_DIR) / "icon" / "question.png" 
        self.question_icon = self.flatten_and_resize(question_path, 18, 18, self.colors.theme["primary_background"], self.colors.theme["icon"])
        self.question_icon_hover = self.flatten_and_resize(question_path, 18, 18, self.colors.theme["secondary_background"], self.colors.theme["icon"])
        
        # Process quit.png
        quit_path = Path(RESSOURCES_DIR) / "icon" / "quit.png" 
        self.quit_icon = self.flatten_and_resize(quit_path, 18, 18, self.colors.theme["quit_button_background"], self.colors.theme["quit_button_text"])
        self.quit_icon_hover = self.flatten_and_resize(quit_path, 18, 18, self.colors.theme["quit_button_background_hover"], self.colors.theme["quit_button_text"])
        
        # ---------------- Main content icon ----------------
        # Process play.png
        play_path = Path(RESSOURCES_DIR) / "icon" / "play.png" 
        self.play_icon = self.flatten_and_resize(play_path, 40, 40, self.colors.theme["primary_background"], self.colors.theme["stroke_button"])
        self.play_icon_hover = self.flatten_and_resize(play_path, 40, 40, self.colors.theme["secondary_background"], self.colors.theme["stroke_button"])
        
        # Process info.png
        info_path = Path(RESSOURCES_DIR) / "icon" / "info.png" 
        self.info_icon = self.flatten_and_resize(info_path, 16, 16, self.colors.theme["primary_background"], self.colors.theme["secondary_text"])
          
    def update_colors(self):
        self.colors = ColorTheme(self.dark_mode)     
            
    def flatten_and_resize(self, img_path, width, height, bg_color, fg_color):
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

    def set_color_theme(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.map('MyCombobox.TCombobox', # Use the style name you defined in create_rounded_dropdown
                       fieldbackground=[('readonly', self.colors.theme["parameters_button_background"])],
                       selectbackground=[('readonly', self.colors.theme["parameters_button_background"])],
                       selectforeground=[('readonly', self.colors.theme["tertiary_text"])],
                       foreground=[('readonly', self.colors.theme["tertiary_text"])],
                       background=[('readonly', self.colors.theme["parameters_button_background"])],
                       arrowcolor=[('readonly', self.colors.theme["tertiary_text"])],
                       bordercolor=[('readonly', self.colors.theme["parameters_button_background"])],
                       darkcolor=[('readonly', self.colors.theme["parameters_button_background"])],
                       lightcolor=[('readonly', self.colors.theme["parameters_button_background"])]
                       )
        self.style.configure('TCombobox.Popdown',
                             background=self.colors.theme["parameters_button_background"],
                             foreground=self.colors.theme["tertiary_text"],
                             selectbackground=self.colors.theme["parameters_button_background"],
                             selectforeground=self.colors.theme["tertiary_text"]
                             )
        # And for the listbox inside the popdown
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
            # Move the downarrow to the left side
            ('Combobox.downarrow', {'side': 'left', 'sticky': 'ns'}),
            # The field (text area) will now be on the right, expanding
            ('Combobox.field', {'sticky': 'nswe', 'children': [
                ('Combobox.padding', {'sticky': 'nswe', 'children': [
                    ('Combobox.textarea', {'sticky': 'nswe', 'expand': 1}) # expand=1 ensures it takes remaining space
                ]})
            ]})
        ]
        self.style.layout('MyCombobox.TCombobox', combobox_layout)

    # Call this method to toggle dark mode and refresh UI
    def toggle_dark_mode(self):
        self.dark_mode = not self.dark_mode
        self.update_colors()
        self.refresh_ui()

    def refresh_ui(self):        
        self.root.configure(bg=self.colors.theme["primary_background"])
        for widget in self.root.winfo_children():
            widget.destroy()
        self.__init__(self.root, self.dark_mode, self.current_page)
  
    def create_layout(self):
        # Top bar with title and controls - FIRST and full width
        self.create_top_bar()

        # Below top bar: main horizontal container (sidebar + content + parameters)
        self.body_frame = tk.Frame(self.main_frame, bg=self.colors.theme["primary_background"])
        self.body_frame.pack(fill=tk.BOTH, expand=True)

        # Sidebar (LEFT) - Pack this FIRST to ensure it stays on the left
        self.create_sidebar()

        # Parameters (RIGHT) - Pack this SECOND so it goes to the right
        self.create_parameters_panel()

        # Content frame (CENTER) - Pack this LAST so it fills the remaining space
        self.content_frame = tk.Frame(self.body_frame, bg=self.colors.theme["primary_background"])
        self.content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Main content inside content frame
        self.main_content = tk.Frame(self.content_frame, bg=self.colors.theme["primary_background"])
        self.main_content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def create_sidebar(self):
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
        # Section title
        title_label = tk.Label(self.sidebar, text=title, bg=self.colors.theme["primary_background"], fg=self.colors.theme["primary_text"],
                              font=(self.font, 11, 'bold'), anchor='w')
        title_label.pack(fill=tk.X, padx=15, pady=(20, 5))
        
        # Menu items
        for text, page_id, icon, icon_hover in items:
            # Add button
            bg_color = self.colors.theme["secondary_background"] if page_id == self.current_page else self.colors.theme["primary_background"]
            if page_id == self.current_page: icon = icon_hover
            button_canvas = self.create_rounded_button(
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
                     
    def create_top_bar(self):
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
        
    def create_parameters_panel(self):
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

        # Content frame for parameters (all but the very last "Name directory")
        self.params_content_frame = tk.Frame(self.params_frame, bg=self.colors.theme["secondary_background"])
        self.params_content_frame.pack(fill=tk.BOTH, expand=True, padx=20)

        # Populate the main content of the parameters panel
        self.create_parameters_content()

        # Name directory at the bottom, packed directly into params_frame
        name_dir_label = tk.Label(self.params_frame, text="Name directory", bg=self.colors.theme["secondary_background"],
                                  fg=self.colors.theme["secondary_text"], font=(self.font, 10))
        name_dir_label.pack(anchor='w', pady=(5, 5), padx=20)

        self.name_directory_entry = self.create_rounded_input(
            self.params_frame, "Arthur_2025_07_22"
        )

    def create_parameters_content(self):
        # Exposure time
        tk.Label(self.params_content_frame, text="Exposure time (ms)", bg=self.colors.theme["secondary_background"], fg=self.colors.theme["secondary_text"], font=(self.font, 10)).pack(anchor='w', pady=(5, 0))
        self.exposure_time_entry = self.create_rounded_input_with_icon(
            self.params_content_frame, "100", self.clock_icon
        )

        # Binning
        tk.Label(self.params_content_frame, text="Binning", bg=self.colors.theme["secondary_background"], fg=self.colors.theme["secondary_text"], font=(self.font, 10)).pack(anchor='w', pady=(5, 0))
        self.binning_dropdown = self.create_rounded_dropdown(
            self.params_content_frame, ["2×2", "3x3"], "2×2"
        )

        # Shutter toggle
        self.shutter_toggle_var = tk.BooleanVar(value=False)
        self.create_custom_toggle(self.params_content_frame, "Shutter", self.shutter_toggle_var)

        # Dual view
        self.dual_view_toggle_var = tk.BooleanVar(value=False)
        self.create_custom_toggle(self.params_content_frame, "Dual view", self.dual_view_toggle_var)

        # Display mode
        tk.Label(self.params_content_frame, text="Display mode", bg=self.colors.theme["secondary_background"], fg=self.colors.theme["secondary_text"], font=(self.font, 10)).pack(anchor='w', pady=(5, 0))
        self.display_mode_dropdown = self.create_rounded_dropdown(
            self.params_content_frame, ["Grayscale"], "Grayscale"
        )

        # Scan Objective
        tk.Label(self.params_content_frame, text="Scan Objective", bg=self.colors.theme["secondary_background"], fg=self.colors.theme["secondary_text"], font=(self.font, 10)).pack(anchor='w', pady=(5, 0))
        self.scan_objective_dropdown = self.create_rounded_dropdown(
            self.params_content_frame, ["4x", "5x", "10x"], "4x"
        )

        # Fluo objective
        tk.Label(self.params_content_frame, text="Fluo objective", bg=self.colors.theme["secondary_background"], fg=self.colors.theme["secondary_text"], font=(self.font, 10)).pack(anchor='w', pady=(5, 0))
        self.fluo_objective_dropdown = self.create_rounded_dropdown(
            self.params_content_frame, ["10x", "20x", "40x"], "10x"
        )

        # Scan shape
        tk.Label(self.params_content_frame, text="Scan shape", bg=self.colors.theme["secondary_background"], fg=self.colors.theme["secondary_text"], font=(self.font, 10)).pack(anchor='w', pady=(5, 0))
        self.scan_shape_dropdown = self.create_rounded_dropdown(
            self.params_content_frame, ["square", "rectangle"], "square"
        )

    # Button
    def create_rounded_input(self, parent, default_value):
        canvas_width = 190 
        canvas_height = 35
        radius = 20 # Corner radius

        canvas = tk.Canvas(parent, width=canvas_width, height=canvas_height,
                           bg=parent.cget("bg"), highlightthickness=0) # Use parent's bg for canvas
        canvas.pack(fill=tk.X, pady=(0, 15), padx=20)

        # Draw the rounded background
        self.draw_rounded_rect(canvas, 0, 0, canvas_width, canvas_height,
                               radius, fill=self.colors.theme["parameters_button_background"],
                               outline=self.colors.theme["parameters_button_background"], tag="input_bg")

        # Create the Entry widget
        entry = tk.Entry(canvas, font=(self.font, 10), bd=0, relief="flat", highlightthickness=0,
                         bg=self.colors.theme["parameters_button_background"], fg=self.colors.theme["tertiary_text"],
                         insertbackground=self.colors.theme["primary_text"]) # Cursor color
        entry.insert(0, default_value)

        # Place the entry widget inside the canvas. Adjust x, y for padding.
        entry_width = canvas_width - 2 * radius # Approximate width of the entry part
        entry_height = canvas_height - 10 # Approximate height of the entry part
        canvas.create_window(radius, canvas_height / 2, window=entry, anchor="w",
                             width=entry_width, height=entry_height)
        return entry
    
    def create_rounded_input_with_icon(self, parent, default_value, icon):

        canvas_width = 190
        canvas_height = 35
        radius = 20
        icon_width = 35

        canvas = tk.Canvas(parent, width=canvas_width, height=canvas_height,
                        bg=parent.cget("bg"), highlightthickness=0)
        canvas.pack(fill=tk.X, pady=(0, 0))

        # Draw the background
        self.draw_rounded_rect(canvas, 0, 0, canvas_width, canvas_height,
                            radius, fill=self.colors.theme["parameters_button_background"],
                            outline=self.colors.theme["parameters_button_background"], tag="input_bg")

        if isinstance(icon, str):
            # It's a text/emoji icon
            tk.Label(canvas, text=icon, bg=self.colors.theme["parameters_button_background"],
                    fg=self.colors.theme["secondary_text"], font=(self.font, 12)).place(x=5, rely=0.5, anchor="w")
        else:
            # Assume it's an image (PhotoImage or ImageTk.PhotoImage)
            canvas.create_image(10, canvas_height // 2, anchor="w", image=icon)
            canvas.image = icon  # Prevent garbage collection


        # Entry widget
        entry = tk.Entry(canvas, font=(self.font, 10), bd=0, relief="flat", highlightthickness=0,
                        bg=self.colors.theme["parameters_button_background"], fg=self.colors.theme["tertiary_text"],
                        insertbackground=self.colors.theme["primary_text"])
        entry.insert(0, default_value)

        entry_width = canvas_width - icon_width - radius
        entry_height = canvas_height - 10
        canvas.create_window(icon_width, canvas_height // 2, window=entry, anchor="w",
                            width=entry_width, height=entry_height)

        return entry
     
    def create_rounded_dropdown(self, parent, options, default):
        canvas_width = 190 
        canvas_height = 35 
        radius = 20 # Corner radius

        canvas = tk.Canvas(parent, width=canvas_width, height=canvas_height,
                           bg=parent.cget("bg"), highlightthickness=0) # Use parent's bg for canvas
        canvas.pack(fill=tk.X, pady=(0, 0))

        # Draw the rounded background
        self.draw_rounded_rect(canvas, 0, 0, canvas_width, canvas_height,
                               radius, fill=self.colors.theme["parameters_button_background"],
                               outline=self.colors.theme["parameters_button_background"], tag="dropdown_bg")

        combo = ttk.Combobox(canvas, values=options, font=(self.font, 10), state='readonly',
                             justify='left', style='MyCombobox.TCombobox')
        combo.set(default)
        
        # Place the combobox inside the canvas.
        combo_width = canvas_width - 10 # Adjust for some padding
        combo_height = canvas_height - 10
        canvas.create_window(5, canvas_height / 2, window=combo, anchor="w",
                             width=combo_width, height=combo_height)

        return combo

    def create_custom_toggle(self, parent, label, boolean_var):
        frame = tk.Frame(parent, bg=self.colors.theme["secondary_background"])
        frame.pack(fill=tk.X, pady=(5, 5))

        tk.Label(frame, text=label, bg=self.colors.theme["secondary_background"],
                 fg=self.colors.theme["secondary_text"], font=(self.font, 10)).pack(side=tk.LEFT)
        
        toggle_canvas = tk.Canvas(frame, width=self.toggle_open_icon.width(),
                          height=self.toggle_open_icon.height(),
                          bg=self.colors.theme["primary_background"], highlightthickness=0)
        toggle_canvas.pack(side=tk.RIGHT, padx = 3)

        def draw_toggle():
            toggle_canvas.delete("all") # Clear previous drawings/images

            if boolean_var.get():
                current_image = self.toggle_open_icon
            else:
                current_image = self.toggle_close_icon

            toggle_canvas.create_image(0, 0, image=current_image, anchor=tk.NW)

        def toggle_command(event=None):
            boolean_var.set(not boolean_var.get())
            draw_toggle()

        toggle_canvas.bind("<Button-1>", toggle_command)
        draw_toggle() # Initial draw
        return boolean_var
                
    def toggle_parameters(self):
        self.show_parameters = not self.show_parameters
        if self.show_parameters:
            self.params_frame.pack(side=tk.RIGHT, fill=tk.Y)
        else:
            self.params_frame.pack_forget()
        
        # Store the after_id and schedule resizing with error handling
        if hasattr(self, 'main_content') and self.main_content.winfo_exists():
            after_id = self.main_content.after(50, self.resize_scan_content_area)
            if not hasattr(self, '_after_ids'):
                self._after_ids = []
            self._after_ids.append(after_id)

    def resize_scan_content_area(self):
        middle_container = self.middle_container_ref
        content_area = self.content_area_ref

        container_width = middle_container.winfo_width()
        container_height = middle_container.winfo_height()

        if self.shape == 'square':
            side = min(container_width, container_height)
            width = height = side
        elif self.shape == 'rectangle':
            height = min(container_height, container_width / 2)
            width = 2 * height
        else:
            height = min(container_height, container_width)
            width = height

        x = (container_width - width) / 2
        y = (container_height - height) / 2

        content_area.place(x=x, y=y, width=width, height=height)

    def create_rounded_button(self, parent, text, command, bg_color, text_color,
                          hover_color, font, width_pixels, height_pixels,
                          corner_radius, side, padx=0, pady=0, padx_text=0, pady_text=0,
                          anchor='center', border_width=0, border_color=None, icon=None, icon_hover=None):
        """
        Create a rounded-corner button on a Canvas that responds to clicks, hover, and shows a hand cursor.
        The entire canvas and its label/widget are bound so that clicks anywhere inside fire `command()`.
        """
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

        # Coordinates
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
        
        shape_id = self.draw_rounded_rect(
            canvas,
            x1 + inset, y1 + inset,
            x2 - inset, y2 - inset,
            max(corner_radius - inset, 0),
            fill=bg_color,
            outline=bg_color,
            tag="button_shape"
        )


        return canvas

    def draw_rounded_rect(self, canvas, x1, y1, x2, y2, radius, fill, outline, tag):
        """Draw a full rounded rectangle using polygon and arcs."""
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

    # Pages   
    def switch_page(self, page_id):
        self.current_page = page_id
        self.refresh_ui()
    
    def show_automatic_scan_page(self):
        # Clear previous widgets if needed
        for widget in self.main_content.winfo_children():
            widget.destroy()

        # Page title
        title_frame = tk.Frame(self.main_content, bg=self.colors.theme["primary_background"])
        title_frame.pack(fill=tk.X)

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

        self.create_rounded_button(
            parent=bottom_frame,
            text="",
            icon=self.play_icon,
            icon_hover=self.play_icon_hover,
            command=self.toggle_parameters, # TODO
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
    
    def show_assist_acquisition_page(self):
        # Clear previous widgets if needed
        for widget in self.main_content.winfo_children():
            widget.destroy()

        # Page title
        title_frame = tk.Frame(self.main_content, bg=self.colors.theme["primary_background"])
        title_frame.pack(fill=tk.X)

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

        self.create_rounded_button(
            parent=bottom_frame,
            text="",
            icon=self.play_icon,
            icon_hover=self.play_icon_hover,
            command=self.toggle_parameters, # TODO
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
        Tooltip(info_label, "Be sure to use the L camera.", posx=0, posy=-30)

        # Trigger resizing after layout completes with error handling
        if hasattr(self, 'main_content') and self.main_content.winfo_exists():
            after_id = self.main_content.after(100, self.resize_scan_content_area)
            if not hasattr(self, '_after_ids'):
                self._after_ids = []
            self._after_ids.append(after_id)
    
    def show_load_position_page(self):
        # Page title
        title_frame = tk.Frame(self.main_content, bg=self.colors.theme["primary_background"])
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Synaptic profil prediction
        prediction_frame = tk.Frame(self.main_content, bg='white', relief=tk.SOLID, bd=1)
        prediction_frame.pack(fill=tk.X, pady=20, padx=20)
        
        pred_title = tk.Label(prediction_frame, text="Synaptic profil prediction",
                             bg='white', font=(self.font, 12, 'bold'))
        pred_title.pack(anchor='w', padx=15, pady=(15, 5))
        
        pred_text = tk.Label(prediction_frame, text="The analysed worm is a\nmutant with a probability\nof 78%",
                           bg='white', font=(self.font, 10), justify=tk.LEFT)
        pred_text.pack(anchor='w', padx=15, pady=(0, 15))
        
        # Wild-type vs Mutation buttons
        buttons_frame = tk.Frame(self.main_content, bg='#2b2b2b')
        buttons_frame.pack(pady=20)
        
        wildtype_frame = tk.Frame(buttons_frame, bg='#2b2b2b')
        wildtype_frame.pack(side=tk.LEFT, padx=20)
        
        wildtype_btn = tk.Button(wildtype_frame, text="⊕", bg='white', fg='black',
                               font=(self.font, 20), width=3, height=2)
        wildtype_btn.pack()
        tk.Label(wildtype_frame, text="Wild-type", bg='#2b2b2b', fg='white',
                font=(self.font, 10)).pack(pady=5)
        
        mutation_frame = tk.Frame(buttons_frame, bg='#2b2b2b')
        mutation_frame.pack(side=tk.LEFT, padx=20)
        
        mutation_btn = tk.Button(mutation_frame, text="⚡", bg='white', fg='black',
                               font=(self.font, 20), width=3, height=2)
        mutation_btn.pack()
        tk.Label(mutation_frame, text="Mutation", bg='#2b2b2b', fg='white',
                font=(self.font, 10)).pack(pady=5)
        
        # Navigation
        nav_frame = tk.Frame(self.main_content, bg='#2b2b2b')
        nav_frame.pack(pady=20)
        
        tk.Label(nav_frame, text="1 / 23", bg='#2b2b2b', fg='white',
                font=(self.font, 10)).pack()
        
        nav_buttons = tk.Frame(nav_frame, bg='#2b2b2b')
        nav_buttons.pack(pady=10)
        
        prev_btn = tk.Button(nav_buttons, text="⏮", bg='white', fg='black',
                           font=(self.font, 16), width=3, height=1)
        prev_btn.pack(side=tk.LEFT, padx=10)
        
        next_btn = tk.Button(nav_buttons, text="⏭", bg='white', fg='black',
                           font=(self.font, 16), width=3, height=1)
        next_btn.pack(side=tk.LEFT, padx=10)
        
        # Launch analysis button
        launch_frame = tk.Frame(self.main_content, bg='#e0e0e0', relief=tk.SOLID, bd=1)
        launch_frame.pack(fill=tk.X, pady=20, padx=50)
        
        launch_btn = tk.Button(launch_frame, text="▶ Launch analysis", bg='#e0e0e0', fg='black',
                             font=(self.font, 12), pady=15,
                             command=lambda: messagebox.showinfo("Info", "Analysis launched!"))
        launch_btn.pack(fill=tk.X)
        
        # Live and Snap buttons
        bottom_controls = tk.Frame(self.main_content, bg='#2b2b2b')
        bottom_controls.pack(side=tk.BOTTOM, pady=20)
        
        live_btn = tk.Button(bottom_controls, text="📹 Live", bg='#e0e0e0', fg='black',
                           font=(self.font, 10), padx=20, pady=10)
        live_btn.pack(side=tk.LEFT, padx=10)
        
        snap_btn = tk.Button(bottom_controls, text="📷 Snap", bg='#e0e0e0', fg='black',
                           font=(self.font, 10), padx=20, pady=10)
        snap_btn.pack(side=tk.LEFT, padx=10)
    
    def show_placeholder_page(self, page_name):
        placeholder = tk.Label(self.main_content, text=f"{page_name} Page\n(Coming soon...)",
                             bg=self.colors.theme["primary_background"], fg=self.colors.theme["primary_text"], font=(self.font, 16))
        placeholder.pack(expand=True)

def main():
    root = tk.Tk()
    app = WormAnalysisApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
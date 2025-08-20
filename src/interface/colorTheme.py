

class ColorTheme:
    """
    A class to define UI color themes for light and dark modes.

    Attributes:
        theme (dict): A dictionary containing all color values mapped to theme elements.
    """
    def __init__(self, dark_mode=False):
        self.theme = {
            # background
            "primary_background": "#262626" if dark_mode else "#ffffff",
            "secondary_background": "#353535" if dark_mode else "#f1f1f1",
            "tertiary_background": "#5b5a5a" if dark_mode else "#d7d7d7",
            
            # text
            "primary_text": "#ffffff" if dark_mode else "#000000",
            "secondary_text": "#e6e6e6" if dark_mode else "#454545",
            "tertiary_text": "#6e6e6e" if dark_mode else "#828282",
            
            # buttons
            "disactive_button": "#262626" if dark_mode else "#ffffff",
            "active_button": "#5c5c5c" if dark_mode else "#f7f7f7",
            "quit_button_background": "#f7f7f7",
            "quit_button_text": "#000000",
            "quit_button_background_hover": "#e1e0e0",
            "toggle_button": "#e6e6e6" if dark_mode else "#828282",
            "dark_mode_button_background": "#ffffff" if dark_mode else "#000000",
            "dark_mode_button_text": "#000000" if dark_mode else "#ffffff",
            "dark_mode_button_background_hover": "#e1e0e0" if dark_mode else "#5b5a5a",
            "parameters_button_background": "#f7f7f7" if dark_mode else "#ffffff",
            "parameters_button_disabled_background": "#353535" if dark_mode else "#f1f1f1",
            "main_button_background": "#262626" if dark_mode else "#ffffff",
            "stroke_button": "#e0e0e0",
            "machine_config_button": "#f7f7f7" if dark_mode else "#f1f1f1",
            
            # tooltip
            "danger_zone": "#FEE9E7",
            "danger_stroke": "#EC221F",
            "danger_text": "#900B09",
            "info_zone": "#E7F3FE",
            "info_stroke": "#1E90FF",
            "info_text": "#0B3D8C",
            
            # others
            "icon": "#ffffff" if dark_mode else "#000000",
            "stroke": "#535353" if dark_mode else "#e0e0e0"
        }
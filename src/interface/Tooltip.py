import tkinter as tk

from src.interface.colorTheme import ColorTheme

class Tooltip:
    """
    A custom tooltip widget for Tkinter with support for themed warning and info messages.

    Displays a styled tooltip with a title, message, and close button when hovering over
    a given widget. The tooltip supports different visual themes ("warning" or "info")
    with corresponding colors provided by a ColorTheme instance.

    Attributes:
        widget (tk.Widget): The Tkinter widget to attach the tooltip to.
        text (str): The message text displayed in the tooltip.
        title (str): The title displayed above the message (default: "Warning").
        theme (str): The visual theme of the tooltip ("warning" or "info").
        posx (int): X offset in pixels relative to the widget (default: 0).
        posy (int): Y offset in pixels relative to the widget (default: 0).
        tooltip_window (tk.Toplevel): The actual tooltip window (created on hover).
        colors (ColorTheme): Instance providing themed color values.
    """
    def __init__(self, widget, text, title="Warning", theme="warning", posx=0, posy=0):
        
        # Initialize the color theme and assign inputs
        self.colors = ColorTheme()   
        self.widget = widget
        self.text = text
        self.title = title
        self.tooltip_window = None
        self.theme = theme
        self.posx = posx
        self.posy = posy
        
        # Set theme-specific colors for background, text, and border
        if self.theme == "warning":
            self.bg = self.colors.theme["danger_zone"]
            self.fg = self.colors.theme["danger_text"]
            self.border_color = self.colors.theme["danger_stroke"]
        elif self.theme == "info":
            self.bg = self.colors.theme["info_zone"]
            self.fg = self.colors.theme["info_text"]
            self.border_color = self.colors.theme["info_stroke"]

        # Bind mouse events to show/hide the tooltip
        widget.bind("<Enter>", self.show_tooltip)
        widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        """
        Display the tooltip window near the associated widget when hovered.

        Skips creation if a tooltip already exists or if the message text is empty.
        """
        if self.tooltip_window or not self.text:
            return

        # Determine the absolute screen position for the tooltip
        x = self.widget.winfo_rootx() + self.posx
        y = self.widget.winfo_rooty() + self.posy

        # Create a top-level window without decorations
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)  # Remove window decorations
        tw.wm_geometry(f"+{x}+{y}")
        tw.configure(bg=self.border_color)
        
        # Inner frame for visual padding and structure
        inner_frame = tk.Frame(tw, bg=self.bg, bd=0, padx=10, pady=8)
        inner_frame.pack(padx=1, pady=1)

        # Top row: icon, title, close button
        top_row = tk.Frame(inner_frame, bg=self.bg)
        top_row.pack(anchor='w', fill=tk.X)

        icon = "❗" if self.theme == "warning" else "ℹ️"
        icon_label = tk.Label(top_row, text=icon, bg=self.bg, fg=self.fg, font=("Inter", 12, "bold"))
        icon_label.pack(side=tk.LEFT)

        title_label = tk.Label(top_row, text=self.title, bg=self.bg, fg=self.fg, font=("Inter", 10, "bold"))
        title_label.pack(side=tk.LEFT, padx=(5, 10))

        close_button = tk.Label(top_row, text="✕", bg=self.bg, fg=self.fg, font=("Inter", 10, "bold"), cursor="hand2")
        close_button.pack(side=tk.RIGHT)
        close_button.bind("<Button-1>", self.hide_tooltip)

        # Message content
        message_label = tk.Label(inner_frame, text=self.text, bg=self.bg, fg=self.fg, font=("Arial", 10), justify='left', wraplength=250)
        message_label.pack(anchor='w', pady=(5, 0))

    def hide_tooltip(self, event=None):
        """
        Destroy the tooltip window if it exists.

        This method is triggered when the mouse leaves the widget or the close button is clicked.
        """
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None
            
        
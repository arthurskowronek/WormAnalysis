import tkinter as tk

from colorTheme import ColorTheme

class Tooltip:
    def __init__(self, widget, text, title="Warning", posx=0, posy=0):
        
        self.colors = ColorTheme()   
        self.widget = widget
        self.text = text
        self.title = title
        self.bg = self.colors.theme["danger_zone"]
        self.fg = self.colors.theme["danger_text"]
        self.border_color = self.colors.theme["danger_stroke"]
        self.tooltip_window = None
        self.posx = posx
        self.posy = posy

        widget.bind("<Enter>", self.show_tooltip)
        widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        if self.tooltip_window or not self.text:
            return

        x = self.widget.winfo_rootx() + self.posx
        y = self.widget.winfo_rooty() + self.posy

        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)  # Remove window decorations
        tw.wm_geometry(f"+{x}+{y}")
        tw.configure(bg=self.border_color)
        
        # Frame inside the border for content
        inner_frame = tk.Frame(tw, bg=self.bg, bd=0, padx=10, pady=8)
        inner_frame.pack(padx=1, pady=1)

        # Top row: icon, title, close button
        top_row = tk.Frame(inner_frame, bg=self.bg)
        top_row.pack(anchor='w', fill=tk.X)

        icon_label = tk.Label(top_row, text="❗", bg=self.bg, fg=self.fg, font=("Inter", 12, "bold"))
        icon_label.pack(side=tk.LEFT)

        title_label = tk.Label(top_row, text=self.title, bg=self.bg, fg=self.fg, font=("Inter", 10, "bold"))
        title_label.pack(side=tk.LEFT, padx=(5, 10))

        close_button = tk.Label(top_row, text="✕", bg=self.bg, fg=self.fg, font=("Inter", 10, "bold"), cursor="hand2")
        close_button.pack(side=tk.RIGHT)
        close_button.bind("<Button-1>", self.hide_tooltip)

        # Message
        message_label = tk.Label(inner_frame, text=self.text, bg=self.bg, fg=self.fg, font=("Arial", 10), justify='left', wraplength=250)
        message_label.pack(anchor='w', pady=(5, 0))

    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None
            
        
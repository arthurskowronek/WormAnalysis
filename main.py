import tkinter as tk
from src.interface.WormAnalysisApp import WormAnalysisApp
from config import config_environment

def main():
    # Create environment
    config_environment()
    # Launch application
    root = tk.Tk()
    app = WormAnalysisApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
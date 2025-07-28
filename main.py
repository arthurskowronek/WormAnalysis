"""
Created on Thursday June 12 10:41:44 2025
@author: Arthur SKOWRONEK, BESSEREAU LAB, INMG, LYON
"""
import tkinter as tk
from src.interface.WormAnalysisApp import WormAnalysisApp
from config import set_up_environment, loadCore

def main():
    # Setup environment
    set_up_environment()
    try: mmc = loadCore()
    except: mmc = None
    
    # Launch application
    root = tk.Tk()
    app = WormAnalysisApp(root, mmc)
    root.mainloop()

if __name__ == "__main__":
    main()
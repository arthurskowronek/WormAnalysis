"""
Created on Thursday June 12 10:41:44 2025
@author: Arthur SKOWRONEK, BESSEREAU LAB, INMG, LYON
"""
import tkinter as tk
from src.interface.WormAnalysisApp import WormAnalysisApp
from config import set_up_environment, loadCore, start_new_session_get_statistics

from config import log_error

# Commande pour créer l'executable: pyinstaller --onefile --icon=icon_worm_detection_analysis.ico --name=Worm_detection main.py

def main():
    # Setup environment
    set_up_environment()
    start_new_session_get_statistics()
    
    # Launch connection with the microscope
    try: 
        mmc = loadCore()
    except Exception as e:
        mmc = None
        log_error(e, "Load core failed")
    
    # Launch application
    try:
        root = tk.Tk()
        app = WormAnalysisApp(root, mmc)
        root.mainloop()
    except Exception as e:
        log_error(e, "Launch application failed")

if __name__ == "__main__":
    main()
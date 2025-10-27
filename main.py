"""
Created on Thursday June 12 10:41:44 2025
@author: Arthur SKOWRONEK, BESSEREAU LAB, INMG, LYON
"""
import os, sys
import pathlib
import tkinter as tk
from src.interface.WormAnalysisApp import WormAnalysisApp
from config import set_up_environment, loadCore, start_new_session_get_statistics, log_error

# Commande pour créer l'executable:    
# For Windows use
"""pyinstaller --onefile --windowed `
  --name=Worm_detection `
  --icon=icon_desktop.ico `
  --add-data "logs;logs" `
  --add-data "models;models" `
  --add-data "ressources;ressources" `
  --collect-all skan `
  --collect-all python_tsp `
  --collect-all sklearn `
  --collect-all numpy `
  --collect-all scipy `
  main.py"""

"""pyinstaller --onefile `
  --name=Worm_detection `
  --icon=icon_desktop.ico `
  --add-data "logs;logs" `
  --add-data "models;models" `
  --add-data "ressources;ressources" `
  --collect-all skan `
  --collect-all python_tsp `
  --collect-all sklearn `
  --collect-all numpy `
  --collect-all scipy `
  main.py"""
    
# For Mac use
"""pyinstaller --onedir --windowed \
    --name=Worm_detection \
    --add-data "logs:logs" \
    --add-data "models:models" \
    --add-data "ressources:ressources" \
    --hidden-import skan \
    main.py 
"""

if getattr(sys, 'frozen', False):
    app_path = pathlib.Path(sys.executable).resolve().parent
    os.chdir(app_path)

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
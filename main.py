"""
Created on Thursday June 12 10:41:44 2025
@author: Arthur SKOWRONEK, BESSEREAU LAB, INMG, LYON
"""
import os
import tkinter as tk
from src.interface.WormAnalysisApp import WormAnalysisApp
from config import set_up_environment, loadCore

TEST = True

def main():
    # Setup environment
    set_up_environment()
    if not TEST:
        Config = "BESSEREAU_Lab.cfg" #The config file has to be in the Micro-Manager root folder. Available : "MMConfig_demo.cfg" "BESSEREAU_Lab.cfg"
        MM_Directory = "C:/Program Files/Micro-Manager-2.0gamma" #Select the folder which contains Micro-Manager.
        os.chdir("C:/Users/imagerie/Desktop/CribleGenetic/") # Give the installation directory (or change to a python line to extract current file directory)
    mmc = loadCore(Config, MM_Directory) if not TEST else None
    
    # Launch application
    root = tk.Tk()
    app = WormAnalysisApp(root, mmc)
    root.mainloop()

if __name__ == "__main__":
    main()
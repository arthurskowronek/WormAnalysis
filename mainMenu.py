"""
Created on Thursday June 12 10:41:44 2025
@author: Arthur SKOWRONEK, BESSEREAU LAB, INMG, LYON
"""

TEST = True

import os
import cv2
import pandas as pd
from pathlib import Path
import pymmcore #Library to connect the the Micro-Manager core

from config import DATA_DIR, RESSOURCES_DIR, MODELS_DIR, LIVE_EXPOSURE_TIME
from src.system.Worm_Position_Manager import WormPositionManager
from src.interface.MainMenu_Page import MainMenu
from src.interface.WormSearchMenu_Page import WormSearchMenu
from src.interface.WormSearchResult_Page import WormSearchResult
from src.interface.WormAcquisitionMenu_Page import WormAcquisitionMenu
from src.interface.AssistAcquisition_Page import AssistAcquisitionPage
from src.interface.MapCamera import MapCamera
from src.interface.Live_Track_Page import LiveTrackPage

#### Loading the MicroManagerCore ####
if not TEST:
    Config = "BESSEREAU_Lab.cfg" #The config file has to be in the Micro-Manager root folder. Available : "MMConfig_demo.cfg" "BESSEREAU_Lab.cfg"
    MM_Directory = "C:/Program Files/Micro-Manager-2.0gamma" #Select the folder which contains Micro-Manager.
    os.chdir("C:/Users/imagerie/Desktop/CribleGenetic/") #Give the installation directory (or change to a python line to extract current file directory)


def main_menu():
    """Main application loop"""
    # Create UI instance
    ui = MainMenu()
    
    # Setup OpenCV window
    cv2.namedWindow("Main menu", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Main menu", ui.width, ui.height)
    cv2.setMouseCallback("Main menu", mouse_callback, ui)
    
    # Main loop
    while True:
        # Draw interface
        img = ui.draw_interface()
        
        # Display
        cv2.imshow("Main menu", img)
        cv2.moveWindow("Main menu", 0, 0)
        
        # Handle keyboard input
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC to quit
            break
            
        if ui.launch_scan or ui.launch_assist_acquisition or ui.launch_saved_positions or ui.quit:
            break
    
    # Cleanup
    cv2.destroyAllWindows()
    
    # Return final state
    if ui.launch_scan:
        return "scan"
    elif ui.launch_assist_acquisition:
        return "assist"
    elif ui.launch_saved_positions:
        return "saved"
    elif ui.quit:
        return "quit"
    else:
        return "none"

def worm_search_menu():
    
    # Create UI instance
    ui = WormSearchMenu()
    
    # Setup OpenCV window
    cv2.namedWindow("Worm Search", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Worm Search", ui.width, ui.height)
    cv2.setMouseCallback("Worm Search", mouse_callback, ui)
    
    # Main loop
    while True:
        # Draw interface
        img = ui.draw_interface()
        
        # Display
        cv2.imshow("Worm Search", img)
        cv2.moveWindow("Worm Search", 0, 0)
        
        # Handle keyboard input
        key = cv2.waitKey(30) & 0xFF
        if key == 27 or ui.exit == True:  # ESC to quit
            ui.exit = True
            break
        elif key != 255:  # Any other key
            ui.handle_key(key)
        
        if ui.launch:
            break
    
    # Cleanup
    cv2.destroyAllWindows()
    
    # Return final state
    return {
        'obj_scan': ui.obj_scan if not ui.scan_input_active else ui.scan_input.replace('.', ''),  
        'dual_view': ui.dual_view,
        'exposure_time': ui.time_exposure,
        "shape": ui.shape,
        'user_directory': ui.user_directory.replace(' ', '_'),
        'exit': ui.exit
    }

def worm_search_result(SCAN_SHAPE, STICHING_IMG, WORM_POSITIONS, WORM_POSITIONS_PROPORTION):
    """Main application loop"""
    # Create UI instance
    ui = WormSearchResult(SCAN_SHAPE, STICHING_IMG, WORM_POSITIONS, WORM_POSITIONS_PROPORTION)
    
    # Setup OpenCV window
    cv2.namedWindow("Worm Search", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Worm Search", ui.width, ui.height)
    cv2.setMouseCallback("Worm Search", mouse_callback, ui)
    
    # Main loop
    while True:
        # Draw interface
        img = ui.draw_interface()
        
        # Display
        cv2.imshow("Worm Search", img)
        cv2.moveWindow("Worm Search", 0, 0)

        # Handle keyboard input
        key = cv2.waitKey(30) & 0xFF
        if key == 27 or ui.exit == True:  # ESC to quit
            ui.exit = True
            break
        
        if ui.analyse:
            break
    
    # Cleanup
    cv2.destroyAllWindows()
    
    return ui.microscope_worm_positions, ui.exit
  
def worm_assist_acquisition_menu():
    
    # Create UI instance
    ui = WormAcquisitionMenu()
    
    # Setup OpenCV window
    cv2.namedWindow("Worm Acquisition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Worm Acquisition", ui.width, ui.height)
    cv2.setMouseCallback("Worm Acquisition", mouse_callback, ui)
    
    # Main loop
    while True:
        # Draw interface
        img = ui.draw_interface()
        
        # Display
        cv2.imshow("Worm Acquisition", img)
        cv2.moveWindow("Worm Acquisition", 0, 0)
        
        # Handle keyboard input
        key = cv2.waitKey(30) & 0xFF
        if key == 27 or ui.exit == True:  # ESC to quit
            ui.exit = True
            break
        elif key != 255:  # Any other key
            ui.handle_key(key)
        
        if ui.launch:
            break
    
    # Cleanup
    cv2.destroyAllWindows()
    
    # Return final state
    return {
        'obj_scan': ui.obj_scan if not ui.scan_input_active else ui.scan_input.replace('.', ''),  
        'obj_fluo': ui.obj_fluo if not ui.fluo_input_active else ui.fluo_input.replace('.', ''),
        'dual_view': ui.dual_view,
        'exposure_time': ui.time_exposure,
        'user_directory': ui.user_directory.replace(' ', '_'),
        'exit': ui.exit
    }
  
def assist_acquisition(CORE, OBJECTIVE_MAGNIFICATION_SCAN, OBJECTIVE_MAGNIFICATION_FLUO, DUALVIEW):
    
    WormPosition = WormPositionManager(Path(RESSOURCES_DIR), new_acquisition = True, table_worm_position = [])
        
    # Get variables
    init_pos_x, init_pos_y = CORE.getXYPosition()
    camera = MapCamera()
    
    # Setup OpenCV window
    ui = AssistAcquisitionPage(CORE, init_pos_x, init_pos_y, OBJECTIVE_MAGNIFICATION_SCAN, OBJECTIVE_MAGNIFICATION_FLUO, DUALVIEW, WormPosition)
    cv2.namedWindow("Assist Acquisition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Assist Acquisition", ui.width, ui.height)
    cv2.setMouseCallback("Assist Acquisition", mouse_callback, ui)
    
    # Main loop
    while True:
        
        #Acquire image
        CORE.snapImage() 
        display_image = CORE.getImage()  
        pos_x, pos_y = CORE.getXYPosition()
        ui.set_image_position(display_image, pos_x, pos_y)  
    
        # update camera map
        camera.update(pos_x, pos_y, init_pos_x, init_pos_y, ui.find_worm)
        ui.map_camera = camera.get_map()
    
        # Draw interface
        img = ui.draw_interface() 
    
        # Display
        cv2.imshow("Assist Acquisition", img)
        cv2.moveWindow("Assist Acquisition", 0, 0)
        
        # User experience
        key = cv2.waitKey(30) & 0xFF
        if key == 27 or ui.exit == True:  # ESC to quit
            ui.exit = True
            pos = WormPosition.get_worm_position(0)
            CORE.setXYPosition(CORE.getXYStageDevice(), pos[0], pos[1])
            break
        elif ui.analyse:
            pos = WormPosition.get_worm_position(0)
            CORE.setXYPosition(CORE.getXYStageDevice(), pos[0], pos[1])
            break
        elif key != 255:  # Any other key
            ui.handle_key(key) 
    
    # Cleanup
    cv2.destroyAllWindows()
    
    return WormPosition.get_all_worm_position(), ui.exit
  
  
def live_track(CORE, NEW_ACQUISITION = True, worm_positions = [], exposure_time = 100):
    """Main application loop"""
    
    WormPosition = WormPositionManager(Path(RESSOURCES_DIR), new_acquisition = NEW_ACQUISITION, table_worm_position = worm_positions)
        
    # Get variables
    csv_path = Path(MODELS_DIR) / "best_model_tracking.csv"
    df = pd.read_csv(csv_path)
    best_row = df.loc[df['best_score'].idxmax()]
    best_scaler = best_row['best_scaler_name']
    best_model = best_row['best_model_name']   

    csv_path = Path(RESSOURCES_DIR) / "parameters.csv"
    params = pd.read_csv(csv_path)
    user_directory = str(params['user_directory'].iloc[0])
    
    init_pos_x, init_pos_y = CORE.getXYPosition()
    
    # Setup OpenCV window
    ui = LiveTrackPage(CORE, init_pos_x, init_pos_y, best_scaler, best_model, user_directory, WormPosition)
    cv2.namedWindow("Live analysis", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live analysis", ui.width, ui.height)
    cv2.setMouseCallback("Live analysis", mouse_callback, ui)
    
    # Set up CORE
    CORE.setExposure(exposure_time)
    
    # Main loop
    while True:
        #Acquire image
        CORE.snapImage() 
        display_image = CORE.getImage()
        pos_x, pos_y = CORE.getXYPosition()
        
        ui._last_raw_frame = display_image.copy()
        
        if not ui.trackbars_visible:
            ui.set_image_position(display_image, pos_x, pos_y)
    
        # Draw interface
        img = ui.draw_interface() 
    
        # Display
        cv2.imshow("Live analysis", img)
        cv2.moveWindow("Live analysis", 0, 0)
        
        # User experience
        key = cv2.waitKey(30) & 0xFF
        if key == 27 or ui.end:  # ESC to quit
            ui.end_of_program()
            break
        elif key != 255:  # Any other key
            ui.handle_key(key)

    # Cleanup
    ui._destroy_trackbar_window()
    cv2.destroyAllWindows()
    
    return True



def mouse_callback(event, x, y, flags, param):
    """Mouse callback function"""
    ui = param
    
    if event == cv2.EVENT_LBUTTONDOWN:
        ui.handle_click(x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        ui.handle_mouse_move(x, y)

def LoadCore(CONFIG, DIRECTORY):
    mmc = pymmcore.CMMCore()
    mmc.setDeviceAdapterSearchPaths([DIRECTORY])
    mmc.loadSystemConfiguration(os.path.join(DIRECTORY, CONFIG))
    return mmc

if __name__ == "__main__":
    
    # Initialize directory
    dirs_to_clear = ["Unclassified","Mutant_prediction","WT_prediction","Scan","Scan_modified"]
    for subdir in dirs_to_clear:
        directory = Path(DATA_DIR) / subdir
        for file in directory.iterdir():
            if file.is_file():
                file.unlink()
    
    
    mode_selected = main_menu()
    
    if mode_selected == "scan":
        
        global_parameters = worm_search_menu()
        
        OBJECTIVE_MAGNIFICATION_SCAN = int(global_parameters.get('obj_scan'))
        SCAN_SHAPE = global_parameters.get('shape')
        DUALVIEW = global_parameters.get('dual_view')
        CAMERA_EXPOSURE_TIME = int(global_parameters.get('exposure_time'))
        USER_DIRECTORY = str(global_parameters.get('user_directory'))
        EXIT = bool(global_parameters.get('exit'))
        
        # write parameters in a csv file
        csv_path = Path(RESSOURCES_DIR) / "parameters.csv"
        params = pd.read_csv(csv_path)
        OBJECTIVE_MAGNIFICATION_FLUO = str(params['obj_fluo'].iloc[0])
        pd.DataFrame({
            'obj_scan': [OBJECTIVE_MAGNIFICATION_SCAN],
            'obj_fluo': [OBJECTIVE_MAGNIFICATION_FLUO],
            'dual_view': [DUALVIEW],
            'exposure_time': [CAMERA_EXPOSURE_TIME],
            'shape': [SCAN_SHAPE],
            'user_directory': [USER_DIRECTORY]
        }).to_csv(csv_path, index=False)
        
        
        if not TEST and not EXIT:
            print("Initiate system...")
            mmc = LoadCore(Config, MM_Directory) 
            mmc.setExposure(CAMERA_EXPOSURE_TIME)
            print("Core successfully loaded")
            
            import src.system.Grid_Search as Crible
            
            WORM_POSITIONS, WORM_POSITIONS_PROPORTION, STICHING_IMG = Crible.ScanSlice(mmc, OBJECTIVE_MAGNIFICATION_SCAN, DUALVIEW, SCAN_SHAPE)
            mmc.setExposure(LIVE_EXPOSURE_TIME)
            WORM_POSITIONS, EXIT = worm_search_result(SCAN_SHAPE, STICHING_IMG, WORM_POSITIONS, WORM_POSITIONS_PROPORTION)
            
            if not EXIT:
                live_track(mmc, True, WORM_POSITIONS, CAMERA_EXPOSURE_TIME)
           
    elif mode_selected == "assist":
        global_parameters = worm_assist_acquisition_menu()
        
        OBJECTIVE_MAGNIFICATION_SCAN = int(global_parameters.get('obj_scan'))
        OBJECTIVE_MAGNIFICATION_FLUO = int(global_parameters.get('obj_fluo'))
        DUALVIEW = global_parameters.get('dual_view')
        CAMERA_EXPOSURE_TIME = int(global_parameters.get('exposure_time'))
        USER_DIRECTORY = str(global_parameters.get('user_directory'))
        EXIT = bool(global_parameters.get('exit'))
        
        # write parameters in a csv file
        csv_path = Path(RESSOURCES_DIR) / "parameters.csv"
        params = pd.read_csv(csv_path)
        SCAN_SHAPE = str(params['shape'].iloc[0])
        pd.DataFrame({
            'obj_scan': [OBJECTIVE_MAGNIFICATION_SCAN],
            'obj_fluo': [OBJECTIVE_MAGNIFICATION_FLUO],
            'dual_view': [DUALVIEW],
            'exposure_time': [CAMERA_EXPOSURE_TIME],
            'shape': [SCAN_SHAPE],
            'user_directory': [USER_DIRECTORY]
        }).to_csv(csv_path, index=False)
        
        if not TEST and not EXIT:
            print("Initiate system...")
            mmc = LoadCore(Config, MM_Directory) 
            mmc.setExposure(LIVE_EXPOSURE_TIME)
            print("Core successfully loaded")
            
            WORM_POSITIONS, EXIT = assist_acquisition(mmc, OBJECTIVE_MAGNIFICATION_SCAN, OBJECTIVE_MAGNIFICATION_FLUO, DUALVIEW)
            
            if not EXIT:
                live_track(mmc, True, WORM_POSITIONS, CAMERA_EXPOSURE_TIME)

    elif mode_selected == "saved":
        print("Loading saved positions...")
        
        # read parameters from the csv file
        param_file = Path(RESSOURCES_DIR) / "parameters.csv"
        params = pd.read_csv(param_file)
        DUALVIEW = bool(params['dual_view'].iloc[0])
        CAMERA_EXPOSURE_TIME = int(params['exposure_time'].iloc[0])
        USER_DIRECTORY = params['user_directory'].iloc[0]
        
        mmc = LoadCore(Config, MM_Directory)
        mmc.setExposure(LIVE_EXPOSURE_TIME)
        live_track(mmc, False, [], CAMERA_EXPOSURE_TIME)
        
    elif mode_selected == "quit": 
        print("Quitting application...")
        
    else:
        print("No action taken.")
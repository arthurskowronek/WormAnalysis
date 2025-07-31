

TEST = True

import os
import cv2
import pandas as pd
from pathlib import Path
import pymmcore #Library to connect the the Micro-Manager core

from config import DATA_DIR, RESSOURCES_DIR, MODELS_DIR, LIVE_EXPOSURE_TIME
from src.system.Worm_Position_Manager import WormPositionManager
from src.interface.Live_Track_Page import LiveTrackPage

import src.system.Grid_Search as Crible
 
  
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
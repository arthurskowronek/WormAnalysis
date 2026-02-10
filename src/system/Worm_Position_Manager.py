import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from config import RESSOURCES_DIR, MICROSCOPE, load_config_file, log_debug_coordinate

from python_tsp.exact import solve_tsp_dynamic_programming # more than 25 pts
from python_tsp.heuristics import solve_tsp_local_search # less than 25 pts

class WormPositionManager:
    """
    Manager for worm positions and related data.
    
    This class handles a CSV file that stores information about worm positions,
    including microscope coordinates, proportional coordinates, model predictions,
    and user-assigned labels. It also manages the order of worms to visit by
    calculating the shortest path using the Traveling Salesman Problem (TSP) algorithm.
    """
    
    def __init__(self, output_folder = Path(RESSOURCES_DIR), new_acquisition = True, table_worm_position = [], filename: str = 'worm_positions.csv', id = 0, corners=None):
        """
        Initializes the WormPositionManager.
        
        Args:
            output_folder (Path): The directory where the CSV file will be stored.
                                  Defaults to the RESSOURCES_DIR.
            new_acquisition (bool): If True, a new CSV file is created. If False,
                                    an existing file is loaded, and the shortest
                                    path is re-calculated. Defaults to True.
            table_worm_position (List): A list of initial worm positions to populate
                                        the CSV file. Each element should be a tuple
                                        or list `(x, y)`. Defaults to an empty list.
            filename (str): The name of the CSV file. Defaults to 'worm_positions.csv'.
        """
        self.output_folder = output_folder
        self.filename = filename
        self.csv_file_path = os.path.join(output_folder, filename)
        
        # Define the columns for the DataFrame.
        self.columns = ['worm_id', 'id_path', 'x_microscope', 'y_microscope', 'x_proportion', 'y_proportion', 'prediction', 'user_label', 'seen']
        self.df = pd.DataFrame(columns=self.columns)
        
        # Create the output directory if it does not exist.
        os.makedirs(output_folder, exist_ok=True)
        
        if new_acquisition:
            self._initialize_csv(table_worm_position, corners)
            self.go_to_first_worm(id)
        else:
            if not os.path.exists(self.csv_file_path):
                self._initialize_csv(table_worm_position)
                self.go_to_first_worm(id)
            else:
                self.df = pd.read_csv(self.csv_file_path)
                #self.find_shortest_path()
                self.go_to_first_worm(id)
      
    def _save_csv(self):
        """Helper to save the current DF to CSV."""
        self.df.to_csv(self.csv_file_path, index=False)
     
    def _initialize_csv(self, table_worm_position = [], corners=None) -> None:
        """
        Creates and initializes a new CSV file with worm position headers.
        
        If `table_worm_position` is provided, it populates the CSV with these
        positions and then calculates the optimal path.

        Args:
            table_worm_position (List): A list of initial worm positions.
        """
        data = {col: [] for col in self.columns}
        self.df = pd.DataFrame(data)
        self._save_csv()
        
        for pos in table_worm_position:
            x,y = pos[0], pos[1]
            self.add_worm_microscope_position(x, y, corners=corners)
            
        # Find shortest path
        self.find_shortest_path()

    def add_worm_microscope_position(self, x: float, y: float, 
                         prediction: float = -1, user_label: str = 'None', corners=None) -> bool:
        """
        Adds a new worm's position to the CSV file.
        
        Args:
            x (float): The x-coordinate from the microscope.
            y (float): The y-coordinate from the microscope.
            prediction (float): The model's prediction for the worm (0 to 1). Defaults to -1.
            user_label (str): The user's label for the worm. Defaults to 'None'.
            
        Returns:
            bool: True if the worm was added successfully (i.e., a new position),
                  False if the position already exists.
        """
        # read the CSV file
        # df = pd.read_csv(self.csv_file_path) # CACHED
        tab_worms = self.get_all_worm_microscope_position()
        # Convert microscope coordinates to proportional coordinates.
        x_proportion, y_proportion = self.transform_microscope_positions_into_proportion(x,y, corners)
        
        new_row = {
            'worm_id': len(self.df),
            'id_path': len(self.df),
            'x_microscope': float(x),
            'y_microscope': float(y),
            'x_proportion': float(x_proportion),
            'y_proportion': float(y_proportion),
            'prediction': float(prediction),
            'user_label': str(user_label) if user_label != 'None' else '',
            'seen': False if len(self.df) > 0 else True
        }
        
        # Add the new line
        if [x,y] not in tab_worms:
            self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
            self._save_csv()
            log_debug_coordinate(f"[WormPos] Added worm {len(self.df)-1} at microscope (x={x}, y={y}) -> prop ({x_proportion:.4f}, {y_proportion:.4f})")
            return True
        else:
            return False
            
    def delete_worm(self, worm_id: int) -> bool:
        """
        Deletes a worm from the CSV file by its unique ID.
        
        After deletion, the `worm_id` and `id_path` columns are reset, and the
        shortest path is re-calculated.

        Args:
            worm_id (int): The unique identifier of the worm to delete.
            
        Returns:
            bool: True if the worm was successfully deleted, False otherwise.
        """
        # df = pd.read_csv(self.csv_file_path) # CACHED
        if self.df.empty:
            return False

        if worm_id not in self.df['worm_id'].values:
            return False

        self.df = self.df[self.df['worm_id'] != worm_id].reset_index(drop=True)

        self.df['worm_id'] = range(len(self.df))
        self.df['id_path'] = range(len(self.df))  # Reset id_path after deletion
        self._save_csv()

        # Optimization: Do NOT recalculate shortest path here.
        # It takes too long (O(N!) or heuristic) for a simple interaction.
        # It will be calculated when entering the "Load Position" page.
        # self.find_shortest_path()

        return True

    def fast_delete_worm(self, worm_id: int) -> bool:
        """
        Deletes a worm from the in-memory DataFrame ONLY.
        Does NOT save to CSV or recalculate paths.
        Used for batch deletions (e.g. during drag).
        """
        if self.df.empty:
            return False

        if worm_id not in self.df['worm_id'].values:
            return False

        self.df = self.df[self.df['worm_id'] != worm_id]
        return True

    def commit_deletions(self):
        """
        Finalizes batch deletions:
        1. Re-indexes worm_ids and id_paths.
        2. Saves to CSV.
        3. 
        Note: Does NOT calculate TSP (find_shortest_path) to save time.
        TSP should be called when switching to 'Show Load Position'.
        """
        if self.df.empty:
            self._save_csv()
            return

        self.df = self.df.reset_index(drop=True)
        self.df['worm_id'] = range(len(self.df))
        # Reset id_path temporarily to match order, or keep as is?
        # If we delete, gaps in id_path might exist if we don't re-index.
        # Simple re-index:
        self.df['id_path'] = range(len(self.df)) 
        
        self._save_csv()
    
    def delete_all_worms(self):
        """
        Deletes all worms from the CSV file
        """
        data = {col: [] for col in self.columns}
        self.df = pd.DataFrame(data)
        self._save_csv()

    # Getters and Setters methods
    def get_worm_microscope_position(self, worm_id: int) -> Optional[pd.Series]:
        """
        Retrieves the microscope coordinates for a specific worm.
        
        Args:
            worm_id (int): The unique identifier of the worm.
            
        Returns:
            Tuple[float, float]: The (x, y) coordinates of the worm. Returns (0, 0)
                                 if the worm is not found.
        """
        # df = pd.read_csv(self.csv_file_path) # CACHED
        if self.df is not None and not self.df.empty:
            row = self.df[self.df['worm_id'] == worm_id]
            if not row.empty:
                
                # return x,y as a tuple
                worm = row.iloc[0]
                x = worm['x_microscope']
                y = worm['y_microscope']
                log_debug_coordinate(f"[WormPos] Retrieved worm {worm_id} pos: ({x}, {y})")
                return x, y
            else:
                print(f"Worm ID {worm_id} not find")
                return 0, 0
        return 0, 0
    
    def get_all_worm_microscope_position(self):
        """
        Retrieves the microscope coordinates for all worms.
        
        Returns:
            List[List[float]]: A list of `[x, y]` coordinate pairs for all worms.
                               Returns an empty list if no worms are found.
        """
        # df = pd.read_csv(self.csv_file_path) # CACHED
        if self.df is not None and not self.df.empty:
            positions = self.df[['x_microscope', 'y_microscope']].astype(int).values.tolist()
            return positions
        else:
            #print("CSV file is empty or not found")
            return []
    
    def get_all_worm_proportion_position(self):
        """
        Retrieves the proportional coordinates and ID for all worms.
        
        Returns:
            List[List[float]]: A list of `[worm_id, x_proportion, y_proportion]`
                               for all worms. Returns an empty list if no worms are found.
        """
        # df = pd.read_csv(self.csv_file_path) # CACHED
        if self.df is not None and not self.df.empty:
            positions = self.df[['worm_id', 'x_proportion', 'y_proportion']].values.tolist()
            return positions
        else:
            #print("CSV file is empty or not found")
            return []
    
    def get_id_worm_seen(self):
        """
        Gets the `worm_id` of the worm currently marked as 'seen'.
        
        Returns:
            int: The `worm_id` of the seen worm. Returns 0 if no worm is marked as seen.
        """
        # df = pd.read_csv(self.csv_file_path) # CACHED

        id_seen = 0
        
        for idx, row in self.df.iterrows():
            if row['seen'] == True:
                id_seen = row['worm_id'] 
        
        return id_seen
    
    def get_id_path_worm_seen(self):
        """
        Gets the `id_path` of the worm currently marked as 'seen'.
        
        Returns:
            int: The `id_path` of the seen worm. Returns 0 if no worm is marked as seen.
        """
        # df = pd.read_csv(self.csv_file_path) # CACHED

        id_path_seen = 0
        
        for idx, row in self.df.iterrows():
            if row['seen'] == True:
                id_path_seen = row['id_path'] 
        
        return id_path_seen
    
    def get_worm_label(self, worm_id: int) -> str:
        """
        Retrieves the user-assigned label for a specific worm.
        
        Args:
            worm_id (int): The unique identifier of the worm.
            
        Returns:
            str: The user label ('Mutant', 'Wild-Type', 'None', etc.).
                 Returns 'None' if the worm is not found or an error occurs.
        """
        try:
            # df = pd.read_csv(self.csv_file_path) # CACHED
            if self.df.empty:
                return 'None'

            row = self.df[self.df['worm_id'] == worm_id]
            if row.empty:
                print(f"Worm ID {worm_id} not find.")
                return 'None'

            return row.iloc[0]['user_label']

        except Exception as e:
            print(f"Error when getting the label: {e}")
            return 'None'
        
    def get_worm_prediction(self, worm_id: int) -> str:
        """
        Retrieves the model's prediction for a specific worm.
        
        Args:
            worm_id (int): The unique identifier of the worm.
            
        Returns:
            float: The prediction value (typically between 0 and 1).
                   Returns -1.0 if the worm is not found or an error occurs.
        """
        try:
            # df = pd.read_csv(self.csv_file_path) # CACHED
            if self.df.empty:
                return 'None'

            row = self.df[self.df['worm_id'] == worm_id]
            if row.empty:
                print(f"Worm ID {worm_id} not find.")
                return 'None'

            return row.iloc[0]['prediction']

        except Exception as e:
            print(f"Error when getting the prediction: {e}")
            return 'None'

    def get_number_of_worms(self):
        """
        Returns the total number of worms in the dataset.
        
        Returns:
            int: The count of worms. Returns 0 if the CSV file is empty or not found.
        """
        # df = pd.read_csv(self.csv_file_path) # CACHED
        return len(self.df) if self.df is not None else 0

    def get_mutant_proportion(self) -> float:
        """
        Calculates the proportion of worms labeled 'Mutant' among all worms that
        have been manually labeled by a user.
        
        Returns:
            float: The proportion of mutants (0.0 to 1.0). Returns 0.0 if no
                   worms have a user label.
        """
        # df = pd.read_csv(self.csv_file_path) # CACHED
        
        if self.df.empty:
            return 0.0
        
        # Filter worms that have a user_label (not empty, not 'None', not NaN)
        labeled_worms = self.df[
            (self.df['user_label'].notna()) & 
            (self.df['user_label'] != '') & 
            (self.df['user_label'] != 'None')
        ]
        
        if labeled_worms.empty:
            return 0.0
        
        # Count mutants among labeled worms
        mutant_count = len(labeled_worms[labeled_worms['user_label'] == 'Mutant'])
        total_labeled = len(labeled_worms)
        
        proportion = mutant_count / total_labeled
        
        return proportion

    def get_mutant_worm_ids(self) -> list:
        """
        Retrieves the list of IDs for worms labeled as 'Mutant'.

        Returns:
            list: A list of integers representing the IDs of mutant worms.
        """
        # df = pd.read_csv(self.csv_file_path) # CACHED
        
        if self.df.empty:
            return []
        
        # Filter for 'Mutant' label
        mutant_worms = self.df[self.df['user_label'] == 'Mutant']
        
        if mutant_worms.empty:
            return []
            
        return mutant_worms['worm_id'].tolist()
    
    def update_worm_label(self, worm_id: int, user_label: str) -> bool:
        """
        Updates the user-assigned label for a specific worm.
        
        Args:
            worm_id (int): The unique identifier of the worm.
            user_label (str): The new label to assign.
            
        Returns:
            bool: True if the update was successful, False otherwise.
        """
        try:
            # df = pd.read_csv(self.csv_file_path) # CACHED
            if self.df.empty:
                return False
                
            mask = self.df['worm_id'] == worm_id
            if not mask.any():
                print(f"Worm ID {worm_id} not find for update")
                return False
            
            self.df.loc[mask, 'user_label'] = str(user_label)
            self._save_csv()
            
            return True
            
        except Exception as e:
            print(f"Error when updating: {e}")
            return False
        
    def update_worm_position(self, worm_id: int, microscope_position_x: float, microscope_position_y: float) -> bool:
        """
        Updates the microscope position for a specific worm.
        
        Args:
            worm_id (int): The unique identifier of the worm.
            microscope_position_x (float): The new x position
            microscope_position_y (float): The new y position
            
        Returns:
            bool: True if the update was successful, False otherwise.
        """
        try:
            # df = pd.read_csv(self.csv_file_path) # CACHED
            if self.df.empty:
                return False
                
            mask = self.df['worm_id'] == worm_id
            if not mask.any():
                print(f"Worm ID {worm_id} not find for update")
                return False
            
            self.df.loc[mask, 'x_microscope'] = float(microscope_position_x)
            self.df.loc[mask, 'y_microscope'] = float(microscope_position_y)
            x_proportion, y_proportion = self.transform_microscope_positions_into_proportion(microscope_position_x,microscope_position_y)
            self.df.loc[mask, 'x_proportion'] = float(x_proportion)
            self.df.loc[mask, 'y_proportion'] = float(y_proportion)
            self._save_csv()
            
            return True
            
        except Exception as e:
            print(f"Error when updating: {e}")
            return False
    
    def update_worm_prediction(self, worm_id: int, prediction: float) -> bool:
        """
        Updates the model's prediction for a specific worm.
        
        Args:
            worm_id (int): The unique identifier of the worm.
            prediction (float): The new prediction value (0 to 1).
            
        Returns:
            bool: True if the update was successful, False otherwise.
        """
        try:
            # df = pd.read_csv(self.csv_file_path) # CACHED
            if self.df.empty:
                return False
                
            mask = self.df['worm_id'] == worm_id
            if not mask.any():
                print(f"Worm ID {worm_id} not find for update")
                return False
            
            self.df.loc[mask, 'prediction'] = float(prediction)
            self._save_csv()
            
            return True
            
        except Exception as e:
            print(f"Error when updating: {e}")
            return False
    
    # Transform coordinates methods
    def transform_microscope_positions_into_proportion(self, x, y, corners=None):
        """
        Transforms microscope coordinates to proportional coordinates (0 to 1).
        
        Args:
            x (float): Microscope x-coordinate.
            y (float): Microscope y-coordinate.
            corners (dict, optional): Scan corners. Defaults to None.
            
        Returns:
            Tuple[float, float]: The transformed (x_prop, y_prop) coordinates.
        """
        if corners:
            start_corner_x = corners.get('start_x')
            start_corner_y = corners.get('start_y')
            end_corner_x = corners.get('end_x')
            end_corner_y = corners.get('end_y')
        else:
            parameters = load_config_file()
            start_corner_x = parameters.get('start_x')
            start_corner_y = parameters.get('start_y')
            end_corner_x = parameters.get('end_x')
            end_corner_y = parameters.get('end_y')
        
        x = (x - start_corner_x) / (end_corner_x - start_corner_x)
        y = (y - start_corner_y) / (end_corner_y - start_corner_y)

        config = load_config_file()
        shape = config.get("shape")
        if shape == "Square":
            # 0,0 is in the top right corner, so we need to change the origin
            if MICROSCOPE == "Macrozoom":
                x_prop = 1 - x
                y_prop = 1 - y  
            elif MICROSCOPE == "Nikon": 
                x_prop = 1 - y
                y_prop = x
        else:
            if MICROSCOPE == "Macrozoom":
                x_prop = 1 - x
                y_prop = 1 - y
            elif MICROSCOPE == "Nikon":
                x_prop = 1 - x
                y_prop = 1 - y
            
        return x_prop, y_prop
        
    def transform_proportion_into_microscope_positions(self, x_prop, y_prop):
        """
        Transforms proportional coordinates (0 to 1) back to microscope coordinates.
        
        Args:
            x_prop (float): Proportional x-coordinate.
            y_prop (float): Proportional y-coordinate.
            
        Returns:
            Tuple[float, float]: The transformed (x_microscope, y_microscope) coordinates.
        """
        parameters = load_config_file()
        start_corner_x = parameters.get('start_x')
        start_corner_y = parameters.get('start_y')
        end_corner_x = parameters.get('end_x')
        end_corner_y = parameters.get('end_y')

        config = load_config_file()
        shape = config.get("shape")
        if shape == "Square":
            if MICROSCOPE == "Macrozoom":
                x = 1 - x_prop
                y = 1 - y_prop
            elif MICROSCOPE == "Nikon":
                x = y_prop 
                y = 1 - x_prop
        else:
            if MICROSCOPE == "Macrozoom":
                x = 1 - x_prop
                y = 1 - y_prop
            elif MICROSCOPE == "Nikon":
                x = 1 - x_prop
                y = 1 - y_prop 

        x_microscope = x * (end_corner_x - start_corner_x) + start_corner_x
        y_microscope = y * (end_corner_y - start_corner_y) + start_corner_y

        return x_microscope, y_microscope
    
    # Change worm being seen methods   
    def go_to_first_worm(self, id = 0):
        """
        Sets the first worm in the TSP-calculated path as the currently 'seen' worm.
        All other worms are marked as 'not seen'.
        """
        print(f"id: {id}")
        # df = pd.read_csv(self.csv_file_path) # CACHED
        
        if self.df.empty:
            print("No worms available in the CSV file.")
            return
        
        # Set all worms to not seen
        self.df['seen'] = False
        
        # Set the first worm in the path (id_path = 0) to seen
        mask_first = self.df['id_path'] == id
        self.df.loc[mask_first, 'seen'] = True
        
        # Save the updated DataFrame
        self._save_csv()
             
    def go_to_next_worm(self):
        """
        Navigates to the next worm in the TSP-calculated path.
        
        The current 'seen' worm is marked as 'not seen', and the worm with
        the next `id_path` is marked as 'seen'. 
        """
        if self.df.empty:
            return

        # Get current seen worm path id
        current_seen = self.df[self.df['seen'] == True]
        if current_seen.empty:
            current_id_path = -1
        else:
            current_id_path = current_seen.iloc[0]['id_path']

        # Determine next id_path
        next_id_path = current_id_path + 1
        if next_id_path >= len(self.df):
            # stay at the last valid index
            next_id_path = len(self.df) - 1

        # Update seen status
        self.df['seen'] = False
        
        # Find the row with the next_id_path and set it to seen
        mask = self.df['id_path'] == next_id_path
        if mask.any():
            self.df.loc[mask, 'seen'] = True
        else:
            # Fallback if id_path is missing for some reason
            if not self.df.empty:
                self.df.iloc[-1, self.df.columns.get_loc('seen')] = True

        self._save_csv()
            
    def go_to_last_worm(self):
        """
        Navigates to the previous worm in the TSP-calculated path.
        
        The current 'seen' worm is marked as 'not seen', and the worm with
        the previous `id_path` is marked as 'seen'.
        """
        if self.df.empty:
            return

        # Get current seen worm path id
        current_seen = self.df[self.df['seen'] == True]
        if current_seen.empty:
            current_id_path = 0
        else:
            current_id_path = current_seen.iloc[0]['id_path']

        # Determine prev id_path
        prev_id_path = current_id_path - 1
        if prev_id_path < 0:
            prev_id_path = 0
        
        self.df['seen'] = False
        
        mask = self.df['id_path'] == prev_id_path
        if mask.any():
            self.df.loc[mask, 'seen'] = True
        else:
             if not self.df.empty:
                self.df.iloc[0, self.df.columns.get_loc('seen')] = True

        self._save_csv()
        
    def go_to_next_mutant(self):
        """
        Navigue vers le prochain mutant après la position actuelle.
        S'arrête s'il n'y en a plus après.
        """
        if self.df.empty:
            return

        # 1. Finds the id_path of the currently "seen" worm
        current_seen = self.df[self.df['seen'] == True]
        if current_seen.empty:
            current_id_path = -1
        else:
            current_id_path = current_seen.iloc[0]['id_path']
        
        # 2. Search for mutants that have an id_path GREATER than the current id_path
        # Use sort_values by id_path to ensure we get the next one in the path order
        next_mutants = self.df[
            (self.df['id_path'] > current_id_path) & 
            (self.df['user_label'] == 'Mutant')
        ].sort_values(by='id_path')
        
        if not next_mutants.empty:
            # We take the very first mutant found after our position
            next_worm_id = next_mutants.iloc[0]['worm_id']
            
            self.df['seen'] = False
            self.df.loc[self.df['worm_id'] == next_worm_id, 'seen'] = True
            self._save_csv()
            print(f"Moving to mutant with id_path {next_mutants.iloc[0]['id_path']}")
        else:
            print("There are no more mutants after this position. We don't move.")

    def go_to_last_mutant(self):
        """
        Navigue vers le mutant précédent avant la position actuelle.
        S'arrête s'il n'y en a plus avant.
        """
        if self.df.empty:
            return

        current_seen = self.df[self.df['seen'] == True]
        if current_seen.empty:
            current_id_path = len(self.df)
        else:
            current_id_path = current_seen.iloc[0]['id_path']
        
        # 2. Search for mutants that have an id_path LOWER than the current id_path
        # Sort descending to find the closest one
        prev_mutants = self.df[
            (self.df['id_path'] < current_id_path) & 
            (self.df['user_label'] == 'Mutant')
        ].sort_values(by='id_path', ascending=False)
        
        if not prev_mutants.empty:
            # We take the first one in the descending list (the closest predecessor)
            prev_worm_id = prev_mutants.iloc[0]['worm_id']
            
            self.df['seen'] = False
            self.df.loc[self.df['worm_id'] == prev_worm_id, 'seen'] = True
            self._save_csv()
            print(f"Moving to previous mutant with id_path {prev_mutants.iloc[0]['id_path']}")
        else:
            print("There is no mutant before this position. We don't move.")
    
    # Others methods
    def find_shortest_path(self):
        """
        Calculates the shortest path to visit all worm positions.
        
        This method solves the Traveling Salesman Problem (TSP) using either an
        exact algorithm (for <= 25 worms) or a heuristic local search (for > 25 worms).
        The calculated path is stored in the `id_path` column, and the DataFrame
        is sorted and saved.
        """
        # Compute dist_matrix from worm positions
        # df = pd.read_csv(self.csv_file_path) # CACHED
        if self.df.empty:
            return
        positions = self.df[['x_microscope', 'y_microscope']].values
        dist_matrix = np.linalg.norm(positions[:, np.newaxis] - positions, axis=2)
        
        if len(self.df) <= 10: # software bug with more points
            # Use exact method
            permutation, dist_opt = solve_tsp_dynamic_programming(dist_matrix)
        else:
            # Use local search method
            permutation, dist_approx = solve_tsp_local_search(dist_matrix)
            
        for i in range(len(self.df)):
            mask = self.df['worm_id'] == permutation[i]
            self.df.loc[mask, 'id_path'] = i
            
        # create new csv file with row in order of 'id_path'
        self.df = self.df.sort_values(by='id_path')
        self._save_csv()
    
    def show_map_worms_position(self):
        """
        Generates an image visualizing the positions of all worms on a map.
        
        The visualization includes:
        - Worm IDs and their positions.
        - Color-coded circles for user labels ('Mutant', 'Wild-Type').
        - Gray circles for predicted labels.
        - A red circle highlight for the 'seen' worm.
        
        Returns:
            np.ndarray: An OpenCV-compatible image (numpy array).
        """
        # df = pd.read_csv(self.csv_file_path) # CACHED
        df = self.df

        # Create a black image
        size = 700
        img = np.zeros((size, size, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        white = (255, 255, 255)
        green = (0, 255, 0)
        orange = (0, 165, 255)
        red = (0, 0, 255)
        gray_green = (100, 140, 100)
        gray_red = (100, 100, 150)

        # Define coordinates range
        min_x, max_x = df['x_microscope'].min(), df['x_microscope'].max()
        min_y, max_y = df['y_microscope'].min(), df['y_microscope'].max()

        def normalize_coord(coord, min_val, max_val):
            if min_val == max_val: max_val +=1
            boundary = 80
            return int((coord - min_val)/(max_val - min_val) * (size - 2 * boundary) + boundary)
    

        # Draw each worm
        for idx, row in df.iterrows():
            x, y = row['x_microscope'], row['y_microscope']
            x = normalize_coord(x, min_x, max_x)
            y = normalize_coord(y, min_y, max_y)
            
            id = int(row['worm_id'])
            user_label = str(row['user_label'])
            prediction = float(row['prediction'])
            seen = row['seen']
            
            if user_label == 'Mutant':
                cv2.circle(img, (x, y), radius=7, color=orange, thickness=-1)
            elif user_label == 'Wild-Type':
                cv2.circle(img, (x, y), radius=7, color=green, thickness=-1)
            else:
                if prediction == -1:
                    size = 5  # length from center to edge of the cross
                    cv2.line(img, (x - size, y - size), (x + size, y + size), color=white, thickness=1)
                    cv2.line(img, (x - size, y + size), (x + size, y - size), color=white, thickness=1)
                elif prediction >= 0.5:
                    cv2.circle(img, (x, y), radius=5, color=gray_red, thickness=-1)
                elif prediction < 0.5:
                    cv2.circle(img, (x, y), radius=5, color=gray_green, thickness=-1)
              
            if seen: 
                  cv2.circle(img, (x, y), radius=15, color=red, thickness=2)
                  
            cv2.putText(img, f"{id}", (x + 10, y), font, 0.4, white, 1)
        
        return img
    
    def show_table_worms_positions(self):
        """
        Generates an image of a table displaying key information for each worm.
        
        The table shows the `worm_id`, model `prediction`, and `user_label`.
        The row for the currently 'seen' worm is highlighted in red.
        
        Returns:
            np.ndarray: An OpenCV-compatible image (numpy array) of the table.
        """
        # df = pd.read_csv(self.csv_file_path) # CACHED
        df = self.df

        # Create blank image
        rows = len(df)
        img_height = 50 + rows * 20  # Adjust height based on number of rows
        img = np.ones((img_height, 400, 3), dtype=np.uint8) * 255

        # Define styles
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        line_height = 20
        black = (0, 0, 0)
        white = (255, 255, 255)
        red = (0, 0, 255)

        # Title row
        header = f"{'worm_id':<10} {'prediction':<12} {'user_label':<12}"
        cv2.putText(img, header, (10, 20), font, font_scale, black, 1)

        # Draw a horizontal line
        cv2.line(img, (10, 25), (790, 25), black, 1)

        # Draw data rows
        for i, row in df.iterrows():
            text = f"{str(row['worm_id']):<10} {row['prediction']:<12.2f} {str(row['user_label']):<12}"
            y = 40 + i * line_height
            if row['seen'] == True:
                cv2.putText(img, text, (10, y), font, font_scale, red, 1)
            else:
                cv2.putText(img, text, (10, y), font, font_scale, black, 1)
                
        return img
   
         

    
    
    
    
    

    
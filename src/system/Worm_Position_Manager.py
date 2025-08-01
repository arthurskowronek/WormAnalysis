import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from config import RESSOURCES_DIR, load_config_file

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
    
    def __init__(self, output_folder = Path(RESSOURCES_DIR), new_acquisition = True, table_worm_position = [], filename: str = 'worm_positions.csv'):
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
        
        # Create the output directory if it does not exist.
        os.makedirs(output_folder, exist_ok=True)
        
        if new_acquisition:
            self._initialize_csv(table_worm_position)
        else:
            if not os.path.exists(self.csv_file_path):
                self._initialize_csv(table_worm_position)
            else:
                self.find_shortest_path()
                self.go_to_first_worm()
         
    def _initialize_csv(self, table_worm_position = []) -> None:
        """
        Creates and initializes a new CSV file with worm position headers.
        
        If `table_worm_position` is provided, it populates the CSV with these
        positions and then calculates the optimal path.

        Args:
            table_worm_position (List): A list of initial worm positions.
        """
        data = {col: [] for col in self.columns}
        df = pd.DataFrame(data)
        df.to_csv(self.csv_file_path, index=False)
        
        for pos in table_worm_position:
            x,y = pos[0], pos[1]
            self.add_worm_microscope_position(x, y)
            
        # Find shortest path
        self.find_shortest_path()

    def add_worm_microscope_position(self, x: float, y: float, 
                         prediction: float = -1, user_label: str = 'None') -> bool:
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
        df = pd.read_csv(self.csv_file_path)
        tab_worms = self.get_all_worm_microscope_position()
        # Convert microscope coordinates to proportional coordinates.
        x_proportion, y_proportion = self.transform_microscope_positions_into_proportion(x,y)
        
        new_row = {
            'worm_id': len(df),
            'id_path': len(df),
            'x_microscope': float(x),
            'y_microscope': float(y),
            'x_proportion': float(x_proportion),
            'y_proportion': float(y_proportion),
            'prediction': float(prediction),
            'user_label': str(user_label) if user_label != 'None' else '',
            'seen': False if len(df) > 0 else True
        }
        
        # Add the new line
        if [x,y] not in tab_worms:
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(self.csv_file_path, index=False)
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
        df = pd.read_csv(self.csv_file_path)
        if df.empty:
            return False

        if worm_id not in df['worm_id'].values:
            return False

        df = df[df['worm_id'] != worm_id].reset_index(drop=True)

        df['worm_id'] = range(len(df))
        df['id_path'] = range(len(df))  # Reset id_path after deletion
        df.to_csv(self.csv_file_path, index=False)

        self.find_shortest_path()

        return True

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
        df = pd.read_csv(self.csv_file_path)
        if df is not None and not df.empty:
            row = df[df['worm_id'] == worm_id]
            if not row.empty:
                
                # return x,y as a tuple
                worm = row.iloc[0]
                x = worm['x_microscope']
                y = worm['y_microscope']
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
        df = pd.read_csv(self.csv_file_path)
        if df is not None and not df.empty:
            positions = df[['x_microscope', 'y_microscope']].values.tolist()
            return positions
        else:
            #print("Le fichier CSV est vide ou introuvable.")
            return []
    
    def get_all_worm_proportion_position(self):
        """
        Retrieves the proportional coordinates and ID for all worms.
        
        Returns:
            List[List[float]]: A list of `[worm_id, x_proportion, y_proportion]`
                               for all worms. Returns an empty list if no worms are found.
        """
        df = pd.read_csv(self.csv_file_path)
        if df is not None and not df.empty:
            positions = df[['worm_id', 'x_proportion', 'y_proportion']].values.tolist()
            return positions
        else:
            #print("Le fichier CSV est vide ou introuvable.")
            return []
    
    def get_id_worm_seen(self):
        """
        Gets the `worm_id` of the worm currently marked as 'seen'.
        
        Returns:
            int: The `worm_id` of the seen worm. Returns 0 if no worm is marked as seen.
        """
        df = pd.read_csv(self.csv_file_path)

        id_seen = 0
        
        for idx, row in df.iterrows():
            if row['seen'] == True:
                id_seen = row['worm_id'] 
        
        return id_seen
    
    def get_id_path_worm_seen(self):
        """
        Gets the `id_path` of the worm currently marked as 'seen'.
        
        Returns:
            int: The `id_path` of the seen worm. Returns 0 if no worm is marked as seen.
        """
        df = pd.read_csv(self.csv_file_path)

        id_path_seen = 0
        
        for idx, row in df.iterrows():
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
            df = pd.read_csv(self.csv_file_path)
            if df.empty:
                return 'None'

            row = df[df['worm_id'] == worm_id]
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
            df = pd.read_csv(self.csv_file_path)
            if df.empty:
                return 'None'

            row = df[df['worm_id'] == worm_id]
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
        df = pd.read_csv(self.csv_file_path)
        return len(df) if df is not None else 0

    def get_mutant_proportion(self) -> float:
        """
        Calculates the proportion of worms labeled 'Mutant' among all worms that
        have been manually labeled by a user.
        
        Returns:
            float: The proportion of mutants (0.0 to 1.0). Returns 0.0 if no
                   worms have a user label.
        """
        df = pd.read_csv(self.csv_file_path)
        
        if df.empty:
            return 0.0
        
        # Filter worms that have a user_label (not empty, not 'None', not NaN)
        labeled_worms = df[
            (df['user_label'].notna()) & 
            (df['user_label'] != '') & 
            (df['user_label'] != 'None')
        ]
        
        if labeled_worms.empty:
            return 0.0
        
        # Count mutants among labeled worms
        mutant_count = len(labeled_worms[labeled_worms['user_label'] == 'Mutant'])
        total_labeled = len(labeled_worms)
        
        proportion = mutant_count / total_labeled
        
        return proportion
    
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
            df = pd.read_csv(self.csv_file_path)
            if df.empty:
                return False
                
            mask = df['worm_id'] == worm_id
            if not mask.any():
                print(f"Worm ID {worm_id} not find for update")
                return False
            
            df.loc[mask, 'user_label'] = str(user_label)
            df.to_csv(self.csv_file_path, index=False)
            
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
            df = pd.read_csv(self.csv_file_path)
            if df.empty:
                return False
                
            mask = df['worm_id'] == worm_id
            if not mask.any():
                print(f"Worm ID {worm_id} not find for update")
                return False
            
            df.loc[mask, 'prediction'] = float(prediction)
            df.to_csv(self.csv_file_path, index=False)
            
            return True
            
        except Exception as e:
            print(f"Error when updating: {e}")
            return False
    
    # Transform coordinates methods
    def transform_microscope_positions_into_proportion(self, x, y):
        """
        Transforms microscope coordinates to proportional coordinates (0 to 1).
        
        Args:
            x (float): Microscope x-coordinate.
            y (float): Microscope y-coordinate.
            
        Returns:
            Tuple[float, float]: The transformed (x_prop, y_prop) coordinates.
        """
        parameters = load_config_file()
        start_corner_x = parameters.get('start_corner_x', 0)
        start_corner_y = parameters.get('start_corner_y', 0)
        end_corner_x = parameters.get('end_corner_x', 1)
        end_corner_y = parameters.get('end_corner_y', 1)
        
        x = (x - start_corner_x) / (end_corner_x - start_corner_x)
        y = (y - start_corner_y) / (end_corner_y - start_corner_y)
        # 0,0 is in the top right corner, so we need to change the origin
        x_prop = 1 - y
        y_prop = x
            
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
        start_corner_x = parameters.get('start_corner_x', 0)
        start_corner_y = parameters.get('start_corner_y', 0)
        end_corner_x = parameters.get('end_corner_x', 1)
        end_corner_y = parameters.get('end_corner_y', 1)

        x = y_prop
        y = 1 - x_prop

        x_microscope = x * (end_corner_x - start_corner_x) + start_corner_x
        y_microscope = y * (end_corner_y - start_corner_y) + start_corner_y

        return x_microscope, y_microscope
    
    # Change worm being seen methods   
    def go_to_first_worm(self):
        """
        Sets the first worm in the TSP-calculated path as the currently 'seen' worm.
        All other worms are marked as 'not seen'.
        """
        df = pd.read_csv(self.csv_file_path)
        
        if df.empty:
            print("No worms available in the CSV file.")
            return
        
        # Set all worms to not seen
        df['seen'] = False
        
        # Set the first worm in the path (id_path = 0) to seen
        mask_first = df['id_path'] == 0
        df.loc[mask_first, 'seen'] = True
        
        # Save the updated DataFrame
        df.to_csv(self.csv_file_path, index=False)
             
    def go_to_newt_worm(self):
        """
        Navigates to the next worm in the TSP-calculated path.
        
        The current 'seen' worm is marked as 'not seen', and the worm with
        the next `id_path` is marked as 'seen'. Wraps around to the beginning
        if at the end of the path.
        """
        df = pd.read_csv(self.csv_file_path)
        
        id_seen = 0

        for idx, row in df.iterrows():
            if row['seen'] == True:
                id_seen = idx
                
        mask = df['id_path'] == id_seen
        if id_seen+1 >= len(df):
            mask2 = df['id_path'] == 0
        else:
            mask2 = df['id_path'] == id_seen+1
        df.loc[mask, 'seen'] = False
        df.loc[mask2, 'seen'] = True
        df.to_csv(self.csv_file_path, index=False)
         
    def go_to_next_mutant(self):
        """
        Navigates to the next worm in the path that has the user label 'Mutant'.
        
        Continues iterating through the path until a 'Mutant' is found. If no
        mutants exist, a message is printed.
        """
        df = pd.read_csv(self.csv_file_path)
        
        label = ''
        if 'Mutant' in df['user_label'].values:
            while label != 'Mutant':
                self.go_to_newt_worm()
                id = self.get_id_worm_seen()
                label = self.get_worm_label(id)
        else:
            print("There is no mutant")
            
    def go_to_last_worm(self):
        """
        Navigates to the previous worm in the TSP-calculated path.
        
        The current 'seen' worm is marked as 'not seen', and the worm with
        the previous `id_path` is marked as 'seen'. Wraps around to the end
        if at the beginning of the path.
        """
        df = pd.read_csv(self.csv_file_path)
        
        id_seen = 0
        
        for idx, row in df.iterrows():
            if row['seen'] == True:
                id_seen = idx
                
        mask = df['id_path'] == id_seen
        if id_seen-1 < 0:
            mask2 = df['id_path'] == len(df)-1
        else:
            mask2 = df['id_path'] == id_seen-1
        
        df.loc[mask, 'seen'] = False
        df.loc[mask2, 'seen'] = True
        df.to_csv(self.csv_file_path, index=False)
        
    def go_to_last_mutant(self):
        """
        Navigates to the previous worm in the path that has the user label 'Mutant'.
        
        Continues iterating backward through the path until a 'Mutant' is found.
        If no mutants exist, a message is printed.
        """
        df = pd.read_csv(self.csv_file_path)
        
        label = ''
        if 'Mutant' in df['user_label'].values:
            while label != 'Mutant':
                self.go_to_last_worm()
                id = self.get_id_worm_seen()
                label = self.get_worm_label(id)  
        else:
            print("There is no mutant")
    
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
        df = pd.read_csv(self.csv_file_path)
        if df.empty:
            return
        positions = df[['x_microscope', 'y_microscope']].values
        dist_matrix = np.linalg.norm(positions[:, np.newaxis] - positions, axis=2)
        
        if len(df) <= 25:
            # Use exact method
            permutation, dist_opt = solve_tsp_dynamic_programming(dist_matrix)
        else:
            # Use local search method
            permutation, dist_approx = solve_tsp_local_search(dist_matrix)
            
        for i in range(len(df)):
            mask = df['worm_id'] == permutation[i]
            df.loc[mask, 'id_path'] = i
            
        # create new csv file with row in order of 'id_path'
        sorted_df = df.sort_values(by='id_path')
        sorted_df.to_csv(self.csv_file_path, index=False)
    
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
        df = pd.read_csv(self.csv_file_path)

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
        df = pd.read_csv(self.csv_file_path)

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
   
         

    
    
    
    
    

    
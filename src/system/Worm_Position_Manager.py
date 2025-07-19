import os
import cv2
import numpy as np
import pandas as pd
from typing import Optional

class WormPositionManager:
    """
    Gestionnaire pour les positions de vers
    
    Cette classe permet de gérer un fichier CSV contenant les positions de vers
    avec leurs prédictions et labels utilisateur.
    """
    
    def __init__(self, output_folder: str, new_acquisition = True, table_worm_position = [], filename: str = 'worm_positions.csv'):
        """
        Initialise le gestionnaire de positions de vers.
        
        Args:
            output_folder (str): Dossier de sortie pour le fichier CSV
            table_worm_position
            filename (str): Nom du fichier CSV (par défaut: 'worm_positions.csv')
        """
        self.output_folder = output_folder
        self.filename = filename
        self.csv_file_path = os.path.join(output_folder, filename)
        
        # Colonnes du DataFrame
        self.columns = ['worm_id', 'id_path', 'x', 'y', 'prediction', 'user_label', 'seen']
        
        # Créer le dossier s'il n'existe pas
        os.makedirs(output_folder, exist_ok=True)
        
        if new_acquisition:
            self._initialize_csv(table_worm_position)
        else:
            if os.path.exists(self.csv_file_path):
                print(f"Fichier CSV existant trouvé : {self.csv_file_path}")
            else:
                print(f"Create new csv file. {self.csv_file_path} not found")
                self._initialize_csv(table_worm_position)
         
    def _initialize_csv(self, table_worm_position = []) -> None:
        """
        Create csv file with worm position if given
        """
        data = {col: [] for col in self.columns}
        df = pd.DataFrame(data)
        df.to_csv(self.csv_file_path, index=False)
        
        for pos in table_worm_position:
            x,y = pos[0], pos[1]
            self.add_worm_position(x, y)
            
        # Find shortest path
        self.find_shortest_path()
        
        #print(f"Fichier CSV créé : {self.csv_file_path}")

    def add_worm_position(self, x: float, y: float, 
                         prediction: float = -1, user_label: str = 'None') -> bool:
        """
        Ajoute une nouvelle position de ver.
        
        Args:
            x: Coordonnée X
            y: Coordonnée Y
            prediction: Prédiction du modèle (float entre 0 et 1)
            user_label: Label utilisateur (str) (par défaut: 'None')
            
        Returns:
            bool: True si l'ajout a réussi, False sinon
        """
        # Lire le DataFrame existant
        df = pd.read_csv(self.csv_file_path)
        tab_worms = self.get_all_worm_position()
        
        new_row = {
            'worm_id': len(df),
            'id_path': len(df),
            'x': float(x),
            'y': float(y),
            'prediction': float(prediction),
            'user_label': str(user_label) if user_label != 'None' else '',
            'seen': False if len(df) > 0 else True
        }
        
        # Ajouter la nouvelle ligne
        if [x,y] not in tab_worms:
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(self.csv_file_path, index=False)
            print(f"Position ajoutée pour le ver: ({x}, {y})")
            return True
        else:
            print(f"La position a déjà été sauvegardé: ({x}, {y})")
            return False
            
    def delete_worm(self, worm_id: int) -> bool:
        """
        Supprime un ver par son ID.
        
        Args:
            worm_id: Identifiant du ver à supprimer
            
        Returns:
            bool: True si la suppression a réussi, False sinon
        """
        df = pd.read_csv(self.csv_file_path)
        if df.empty:
            return False

        if worm_id not in df['worm_id'].values:
            return False

        df = df[df['worm_id'] != worm_id].reset_index(drop=True)

        df['worm_id'] = range(len(df))
        df['id_path'] = range(len(df))  # temporairement pour réinitialiser
        df.to_csv(self.csv_file_path, index=False)

        self.find_shortest_path()

        return True


    def get_worm_position(self, worm_id: int) -> Optional[pd.Series]:
        """
        Récupère la position d'un ver par son ID.
        
        Args:
            worm_id: Identifiant du ver
            
        Returns:
            pd.Series ou None: Données du ver ou None si non trouvé
        """
        df = pd.read_csv(self.csv_file_path)
        if df is not None and not df.empty:
            row = df[df['worm_id'] == worm_id]
            if not row.empty:
                
                # return x,y as a tuple
                worm = row.iloc[0]
                x = worm['x']
                y = worm['y']
                return x, y
            else:
                print(f"Ver ID {worm_id} non trouvé")
                return 0, 0
        return 0, 0
    
    def get_all_worm_position(self):
        """
        Récupère la position de tous les vers
        """
        df = pd.read_csv(self.csv_file_path)
        if df is not None and not df.empty:
            positions = df[['x', 'y']].values.tolist()
            return positions
        else:
            #print("Le fichier CSV est vide ou introuvable.")
            return []
    
    def get_id_worm_seen(self):
        df = pd.read_csv(self.csv_file_path)

        id_seen = 0
        
        for idx, row in df.iterrows():
            if row['seen'] == True:
                id_seen = row['worm_id'] 
        
        return id_seen
    
    def get_worm_label(self, worm_id: int) -> str:
        """
        Récupère le label utilisateur pour un ver donné.
        
        Args:
            worm_id: Identifiant du ver
            
        Returns:
            str or None: Label du ver donné, ou None si non trouvé ou erreur
        """
        try:
            df = pd.read_csv(self.csv_file_path)
            if df.empty:
                return 'None'

            row = df[df['worm_id'] == worm_id]
            if row.empty:
                print(f"Ver ID {worm_id} non trouvé.")
                return 'None'

            return row.iloc[0]['user_label']

        except Exception as e:
            print(f"Erreur lors de la récupération du label: {e}")
            return 'None'
        
    def get_worm_prediction(self, worm_id: int) -> str:
        """
        Récupère le label prédit pour un ver donné.
        
        Args:
            worm_id: Identifiant du ver
            
        Returns:
            str or None: Label du ver donné, ou None si non trouvé ou erreur
        """
        try:
            df = pd.read_csv(self.csv_file_path)
            if df.empty:
                return 'None'

            row = df[df['worm_id'] == worm_id]
            if row.empty:
                print(f"Ver ID {worm_id} non trouvé.")
                return 'None'

            return row.iloc[0]['prediction']

        except Exception as e:
            print(f"Erreur lors de la récupération du label: {e}")
            return 'None'

    def update_worm_label(self, worm_id: int, user_label: str) -> bool:
        """
        Met à jour le label utilisateur pour un ver donné.
        
        Args:
            worm_id: Identifiant du ver
            user_label: Nouveau label utilisateur
            
        Returns:
            bool: True si la mise à jour a réussi
        """
        try:
            df = pd.read_csv(self.csv_file_path)
            if df.empty:
                return False
                
            mask = df['worm_id'] == worm_id
            if not mask.any():
                print(f"Ver ID {worm_id} non trouvé pour mise à jour")
                return False
            
            df.loc[mask, 'user_label'] = str(user_label)
            df.to_csv(self.csv_file_path, index=False)
            
            #print(f"Label mis à jour pour le ver {worm_id}: {user_label}")
            return True
            
        except Exception as e:
            print(f"Erreur lors de la mise à jour: {e}")
            return False
    
    def update_worm_prediction(self, worm_id: int, prediction: float) -> bool:
        """
        Met à jour le label utilisateur pour un ver donné.
        
        Args:
            worm_id: Identifiant du ver
            prediction: Prédiction du modèle (float entre 0 et 1)
            
        Returns:
            bool: True si la mise à jour a réussi
        """
        try:
            df = pd.read_csv(self.csv_file_path)
            if df.empty:
                return False
                
            mask = df['worm_id'] == worm_id
            if not mask.any():
                print(f"Ver ID {worm_id} non trouvé pour mise à jour")
                return False
            
            df.loc[mask, 'prediction'] = float(prediction)
            df.to_csv(self.csv_file_path, index=False)
            
            #print(f"Label mis à jour pour le ver {worm_id}: {prediction}")
            return True
            
        except Exception as e:
            print(f"Erreur lors de la mise à jour: {e}")
            return False
    
    
    def find_shortest_path(self):
        """
        Trouve le chemin le plus court entre les positions des vers.
        Utilise l'algorithme de TSP (Travelling Salesman Problem) pour calculer le chemin.
        """

        # Compute dist_matrix from worm positions
        df = pd.read_csv(self.csv_file_path)
        if df.empty:
            #print("Aucune position de ver disponible pour le calcul du chemin.")
            return
        positions = df[['x', 'y']].values
        dist_matrix = np.linalg.norm(positions[:, np.newaxis] - positions, axis=2)
        
        if len(df) <= 25:
            # Use exact method
            from python_tsp.exact import solve_tsp_dynamic_programming # moins de 25 pts
            #print("Utilisation de la méthode exacte pour les chemins courts.")
            permutation, dist_opt = solve_tsp_dynamic_programming(dist_matrix)
        else:
            # Use local search method
            from python_tsp.heuristics import solve_tsp_local_search # plus de de 25 pts
            #print("Utilisation de la méthode de recherche locale pour les chemins courts.")
            permutation, dist_approx = solve_tsp_local_search(dist_matrix)
            
        for i in range(len(df)):
            mask = df['worm_id'] == permutation[i]
            df.loc[mask, 'id_path'] = i
            
        # create new csv file with row in order of 'id_path'
        sorted_df = df.sort_values(by='id_path')
        sorted_df.to_csv(self.csv_file_path, index=False)
                
    def go_to_newt_worm(self):
        """
        Change worm being seen, go to the next one
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
        Change worm being seen, go to the last one
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
        
        df = pd.read_csv(self.csv_file_path)
        
        label = ''
        if 'Mutant' in df['user_label'].values:
            while label != 'Mutant':
                self.go_to_last_worm()
                id = self.get_id_worm_seen()
                label = self.get_worm_label(id)  
        else:
            print("There is no mutant")
     
    def show_map_worms_position(self):
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
        gray = (100, 100, 100)

        # Define coordinates range
        min_x, max_x = df['x'].min(), df['x'].max()
        min_y, max_y = df['y'].min(), df['y'].max()

        def normalize_coord(coord, min_val, max_val):
            if min_val == max_val: max_val +=1
            boundary = 80
            return int((coord - min_val)/(max_val - min_val) * (size - 2 * boundary) + boundary)
    

        # Draw each worm
        for idx, row in df.iterrows():
            x, y = row['x'], row['y']
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
   
         


    
    
    
    
    

    
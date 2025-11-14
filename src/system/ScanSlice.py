import re
import cv2
import time
import traceback
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from tifffile import imwrite, imread
from collections import defaultdict
from scipy.spatial import cKDTree

from config import MODELS_DIR, RESSOURCES_DIR, DATA_DIR, save_corner_positions_into_yaml_config_file, load_config_file, loadCore, log_error

class ScanSlice:
    """
    Class for controlling a microscope to scan a slide and detect worms using a YOLO model.

    This class orchestrates the entire scanning process, from moving the microscope
    stage and capturing images to processing those images with a YOLO model for
    worm detection. It handles the stitching of individual image tiles into a
    final, complete slice.
    """
    def __init__(self, mmc, grossissement, dual_view=False, scan_shape="Square", overlap_percent=10):
        """
        Initializes the ScanSlice object.
        
        Args:
            mmc: The Micro-Manager core object for microscope control.
            grossissement (str): The magnification level (e.g., "10x").
            dual_view (bool): True if using a dual-view camera setup, False otherwise.
            scan_shape (str): The shape of the scanning area ("Square" or "Rectangle").
                              Defaults to "Square".
            overlap_percent (int): The percentage of overlap between adjacent images
                                   to ensure complete coverage and aid in stitching.
                                   Defaults to 10.
        """
        self.mmc = mmc
        self.grossissement = int(grossissement.get().replace("x", ""))
        self.dual_view = dual_view.get()
        self.scan_shape = scan_shape.get()
        self.overlap_percent = overlap_percent
        
        # Calculate step sizes
        config = load_config_file()
        self.step_size_x = int(config.get("microscope_step_size")) / self.grossissement
        self.step_size_y = self.step_size_x / 2 if self.dual_view else self.step_size_x
        
        # Load YOLO model
        self.model = YOLO(Path(MODELS_DIR) / "YOLO_detection.pt")
        
        # Storage for results
        self.list_bounding_boxes = []
        self.positions_info = []
        self.scan_dir = Path(DATA_DIR) / "Scan"
        self.scan_modified_dir = Path(DATA_DIR) / "Scan_modified"
    


    def safe_set_xy(self, x, y, retries=5, wait_between=1.0):
        """
        Robust wrapper to set XY position: retries, waitForDevice, fallback, and optional unload.
        Returns True on success, raises RuntimeError on persistent failure.
        """
        try:
            xy_label = self.mmc.getXYStageDevice()
        except Exception:
            xy_label = None

        # attempt to increase timeout if available
        old_timeout = None
        try:
            old_timeout = self.mmc.getTimeoutMs()
            self.mmc.setTimeoutMs(max(old_timeout, 5000))
        except Exception:
            old_timeout = None

        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                # choose un label stable pour les waitForDevice
                cur_label = xy_label if xy_label else self.mmc.getXYStageDevice()
                print(f"[safe_set_xy] Attempt {attempt} -> device='{cur_label}' pos=({x},{y})")
                self.mmc.waitForDevice(cur_label)
                # deux signatures possibles ; utiliser la signature avec label si on l'a
                if xy_label:
                    self.mmc.setXYPosition(xy_label, float(x), float(y))
                else:
                    self.mmc.setXYPosition(float(x), float(y))
                self.mmc.waitForDevice(cur_label)

                # restore timeout
                if old_timeout is not None:
                    try: self.mmc.setTimeoutMs(old_timeout)
                    except Exception: pass
                return True

            except Exception as e:
                last_exc = e
                print(f"[safe_set_xy] Attempt {attempt} failed: {e}")
                traceback.print_exc()
                time.sleep(wait_between)

                # fallback: pinger le device avec un mouvement relatif 0,0 (avant dernier essai)
                if attempt == retries - 1:
                    try:
                        print("[safe_set_xy] Fallback ping: setRelativeXYPosition(0,0)")
                        if xy_label:
                            self.mmc.setRelativeXYPosition(xy_label, 0.0, 0.0)
                            self.mmc.waitForDevice(xy_label)
                        else:
                            self.mmc.setRelativeXYPosition(0.0, 0.0)
                            self.mmc.waitForDevice(self.mmc.getXYStageDevice())
                    except Exception as e2:
                        print("[safe_set_xy] Fallback relatif KO:", e2)

                # dernier recours local : unload du device stage (attention, risque)
                if attempt == retries:
                    if xy_label:
                        try:
                            print(f"[safe_set_xy] Last resort: unloadDevice('{xy_label}')")
                            self.mmc.unloadDevice(xy_label)
                            time.sleep(1.0)
                        except Exception as ue:
                            print("[safe_set_xy] unloadDevice a levé:", ue)

        # restore timeout si non remis
        if old_timeout is not None:
            try: self.mmc.setTimeoutMs(old_timeout)
            except Exception: pass

        raise RuntimeError("safe_set_xy: impossible de déplacer le stage après plusieurs tentatives") from last_exc

    def safe_reload_core(self):
        """
        Reload the core/configuration but first attempt to gracefully unload camera devices
        to avoid 'camera already open' errors.
        """
        print("[safe_reload_core] Tentative de reload du core (unload des caméras si présentes).")
        try:
            loaded = self.mmc.getLoadedDevices()
        except Exception:
            loaded = []

        # Décharger les devices ressemblant à une caméra pour diminuer chance d'erreur PVCAM
        for dev in loaded:
            try:
                if "Camera" in dev or "camera" in dev or dev.upper().startswith("CAM"):
                    print(f"[safe_reload_core] Unload device '{dev}'")
                    self.mmc.unloadDevice(dev)
                    time.sleep(0.5)
            except Exception as e:
                print(f"[safe_reload_core] Impossible d'unload '{dev}' : {e}")

        # Maintenant recharger la config / core via ta fonction loadCore()
        try:
            # loadCore doit retourner un objet mmc initialisé ; adapte si ta fonction diffère
            self.mmc = loadCore()
            # si nécessaire, appeler initializeAllDevices() pour être sûr que tout est prêt
            try:
                self.mmc.initializeAllDevices()
            except Exception:
                pass
            print("[safe_reload_core] Core rechargé et devices initialisés")
        except Exception as e:
            print("[safe_reload_core] reload du core a échoué :", e)
            raise

    def initialize_scan(self):
        # calculs initiaux...
        self.actual_step_x = self.step_size_x * (1 - self.overlap_percent / 100)
        self.actual_step_y = self.step_size_y * (1 - self.overlap_percent / 100)

        # Get starting position
        self.start_x, self.start_y = self.mmc.getXYPosition()

        # Close shutter
        self.mmc.setAutoShutter(False)

        # Calcul scan...
        config = load_config_file()
        end_x = self.start_x + (int(config.get("scan_height_length")) if self.scan_shape == "Square" else int(config.get("scan_width_length")))
        end_y = self.start_y + int(config.get("scan_height_length"))
        self.scan_width = int((end_x - self.start_x) / self.actual_step_x)
        self.scan_height = int((end_y - self.start_y) / self.actual_step_y)

        # Move to starting position (robuste)
        try:
            self.safe_set_xy(self.start_x, self.start_y)
        except Exception as e:
            print("Initialisation : safe_set_xy a échoué :", e)
            traceback.print_exc()
            try:
                time.sleep(2.0)
                self.safe_set_xy(self.start_x, self.start_y, retries=3, wait_between=2.0)
            except Exception as e2:
                print("Tentatives locales échouées, on tente reload contrôlé du core")
                try:
                    self.safe_reload_core()
                    # après reload, relire positions et ré-essayer
                    self.start_x, self.start_y = self.mmc.getXYPosition()
                    self.safe_set_xy(self.start_x, self.start_y)
                except Exception as e3:
                    print("Reload contrôlé échoué :", e3)
                    raise

        # s'assurer que device est prêt
        try:
            self.mmc.waitForDevice(self.mmc.getXYStageDevice())
        except Exception:
            pass

        # Initialize working variables
        self.file_count = 1
        self.image = None
        self.final_end_x = 0
        self.final_end_y = 0


    def scan(self, verbose=False):
        """
        Executes the main scanning loop.

        The function performs a serpentine scan, moving the stage, snapping
        an image, and processing the *previous* image while the stage is moving.
        This parallelization optimizes the scanning speed.

        Returns:
            List[List[float]]: A list of final, non-overlapping worm positions
                               detected across the entire slide.
        """
        self.initialize_scan()
        
        # Scan grid
        for y_idx in range(self.scan_height):
            # Alternate X direction for serpentine scanning
            x_range = range(self.scan_width) if y_idx % 2 == 0 else range(self.scan_width - 1, -1, -1)
            
            for x_idx in x_range:
                # Calculate absolute position
                pos_x = self.start_x + x_idx * self.actual_step_x
                pos_y = self.start_y + y_idx * self.actual_step_y
                self.mmc.setXYPosition(self.mmc.getXYStageDevice(), pos_x, pos_y) # Move to next position
                
                # Update final positions (we don't want to get the last position if it is on the same x or y position as the start)
                if pos_x > self.final_end_x: self.final_end_x = pos_x
                if pos_y > self.final_end_y: self.final_end_y = pos_y
                
                if self.image is not None: # We process the previous image in order to do the compute during the microscope movement
                    self.process_image_to_detect_worms()
                
                # Wait for movement to complete
                self.mmc.waitForDevice(self.mmc.getXYStageDevice())
                self.mmc.snapImage() # Capture image
                self.image = self.mmc.getImage()
                
                self.file_name = f"SlideScan_R{y_idx}_C{x_idx}_{self.file_count}.tif"
                imwrite(self.scan_dir / self.file_name, self.image)  # Save image
                
                # Record position info
                self.positions_info.append([self.file_count, pos_x, pos_y, x_idx, y_idx])
                self.file_count += 1
                if verbose: print(f"Image {self.file_count-1}/{self.scan_width*self.scan_height} captured at X={pos_x:.2f}, Y={pos_y:.2f}")
        
        # Return to starting position
        self.mmc.setXYPosition(self.mmc.getXYStageDevice(), self.start_x, self.start_y)
        
        # Process final image
        if self.image is not None:
            imwrite(self.scan_modified_dir / self.file_name, self.image)
        
        # Get the actual corner positions (it was the center before)
        start_corner_x = self.start_x - self.actual_step_x // 2
        start_corner_y = self.start_y - self.actual_step_y // 2
        end_corner_x = self.final_end_x + self.actual_step_x // 2
        end_corner_y = self.final_end_y + self.actual_step_y // 2
        save_corner_positions_into_yaml_config_file(start_corner_x, start_corner_y, end_corner_x, end_corner_y)
        
        return self.get_worms_position()

    def process_image_to_detect_worms(self):
        """
        Processes a single image tile to detect worms.

        This method takes the most recently captured image, performs a worm
        detection using the YOLO model, and saves a modified version of the image
        (e.g., with bounding boxes). It also handles the specific logic for
        dual-view camera setups.
        """
        # Process previous image for worm detection
        last_pos_x = self.positions_info[-1][1]
        last_pos_y = self.positions_info[-1][2]
        _, tile_w_full = self.image.shape
        
        if self.dual_view:
            img_half = self.image[:, tile_w_full // 2:]
            img_half_left = self.image[:, :tile_w_full // 2]
            img_half_left = cv2.normalize(img_half_left, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            
            img_down_right = img_half[tile_w_full // 2:, :]
            img_down_right, _ = self.worm_detection(img_down_right, self.file_count - 1, last_pos_x, last_pos_y)
            
            img_up_right = img_half[:tile_w_full // 2, :]
            img_up_right, _ = self.worm_detection(img_up_right, self.file_count - 1, last_pos_x, last_pos_y)
            
            img_half = np.vstack([img_up_right, img_down_right])
            self.image = np.hstack([img_half_left, img_half])
            imwrite(self.scan_modified_dir / self.file_name, self.image)
        else:
            self.image, _ = self.worm_detection(self.image, self.file_count - 1, last_pos_x, last_pos_y)
            imwrite(self.scan_modified_dir / self.file_name, self.image)
    
    def worm_detection(self, img, id, pos_x=0, pos_y=0, drawing = True):
        """
        Detects worms in a given image using the YOLO model.
        
        Args:
            img (np.ndarray): The input image tile (e.g., a single scan image).
            id (int): The unique ID of the image file.
            pos_x (float): The x-coordinate of the stage when the image was captured.
            pos_y (float): The y-coordinate of the stage when the image was captured.
            drawing (bool): If True, bounding boxes are drawn on the output image.
                            Defaults to False.
            
        Returns:
            tuple: A tuple containing the processed image (with or without drawings)
                   and the updated list of all detected bounding boxes.
        """
        # Get the image in the right format
        image = img.copy()
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # Save temporary image
        temp_path = Path(MODELS_DIR) / "temp_converted_image.png"
        cv2.imwrite(str(temp_path), image)
        
        # Prediction
        prediction = self.model.predict(source=str(temp_path), save=False, verbose=False)
        temp_path.unlink()  # Remove temp file
        
        boxes = prediction[0].boxes
        
        if boxes is not None:
            bounding_boxes = boxes.xyxy.cpu().numpy()
            for bbox in bounding_boxes:                
                x1, y1, x2, y2 = bbox
                
                # Calculate offsets from image center and convert it to stage microns
                step = max(self.step_size_x, self.step_size_y)
                H, W = image.shape[:2]
                dx_um1 = (x1/W - 0.5) * step
                dy_um1 = (y1/H - 0.5) * step
                dx_um2 = (x2/W - 0.5) * step
                dy_um2 = (y2/H - 0.5) * step
                
                # Calculate true worm position on stage
                x_worm1 = pos_x + dy_um1
                y_worm1 = pos_y - dx_um1
                x_worm2 = pos_x + dy_um2
                y_worm2 = pos_y - dx_um2
                
                self.list_bounding_boxes.append([id, x_worm1, y_worm1, x_worm2, y_worm2])
                
                if drawing:
                    # Draw bounding box
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 4)
        
        return image, self.list_bounding_boxes

    def _normalize_boxes(self):
        """
        Ensure each bounding box has x1<=x2 and y1<=y2.
        Modifies self.list_bounding_boxes in-place.
        """
        normalized = []
        for box in self.list_bounding_boxes:
            img_id, x1, y1, x2, y2 = box
            nx1, nx2 = (x1, x2) if x1 <= x2 else (x2, x1)
            ny1, ny2 = (y1, y2) if y1 <= y2 else (y2, y1)
            normalized.append([img_id, nx1, ny1, nx2, ny2])
        self.list_bounding_boxes = normalized

    def get_worms_position(self):
        """
        Consolidates overlapping bounding boxes to determine unique worm positions.
        """
        # 0) Normalize coordinates to guarantee x1<=x2 and y1<=y2
        self._normalize_boxes()

        # 1) First apply per-image NMS (this expects normalized boxes)
        self._apply_nms(iou_threshold=0.8)

        # 2) Build best match overlap map across different images
        best_matches = defaultdict(dict)  # {i: {image_id_other: (j, iou)}}
        n = len(self.list_bounding_boxes)
        for i in range(n):
            id_1 = self.list_bounding_boxes[i][0]
            for j in range(i + 1, n):
                id_2 = self.list_bounding_boxes[j][0]
                # Only compare boxes from different images (as you intended)
                if id_1 != id_2:
                    if self._boxes_overlap(self.list_bounding_boxes[i], self.list_bounding_boxes[j]):
                        iou = self._compute_iou(self.list_bounding_boxes[i], self.list_bounding_boxes[j])

                        # Save best match per image pair (keep the highest IoU)
                        if id_2 not in best_matches[i] or iou > best_matches[i][id_2][1]:
                            best_matches[i][id_2] = (j, iou)
                        if id_1 not in best_matches[j] or iou > best_matches[j][id_1][1]:
                            best_matches[j][id_1] = (i, iou)

        # 3) Keep only mutual best matches (i <-> j)
        added_pairs = set()
        for i, matches in best_matches.items():
            for id_other, (j, _) in matches.items():
                # mutual best match check (ensure index j has best for i's image id)
                if best_matches.get(j, {}).get(self.list_bounding_boxes[i][0], (None, -1))[0] == i:
                    pair = tuple(sorted((i, j)))
                    added_pairs.add(pair)

        overlapping_pairs = list(added_pairs)
        overlapping_boxes = self._merge_overlapping_sublists(overlapping_pairs)

        # 4) Add non-overlapping singletons
        flat_overlapping_indices = [idx for sub in overlapping_boxes for idx in sub]
        values_overlapping_boxes = set(int(x) for x in flat_overlapping_indices)  # set of int indices
        for i in range(n):
            if i not in values_overlapping_boxes:
                overlapping_boxes.append([i])

        # 5) Compute centers of each group
        positions_worms = []
        for sublist in overlapping_boxes:
            tab_x, tab_y = [], []
            for idx in sublist:
                _, x1, y1, x2, y2 = self.list_bounding_boxes[idx]
                tab_x.append((x1 + x2) / 2.0)
                tab_y.append((y1 + y2) / 2.0)
            x = sum(tab_x) / len(tab_x)
            y = sum(tab_y) / len(tab_y)
            positions_worms.append([x, y])

        # 6) Cluster nearby centers using KD-Tree
        filtered_positions = []
        if positions_worms:
            pts = np.array(positions_worms)

            # choose min_dist sensibly:
            # - either a fixed value in the same coordinate units (e.g. 50),
            # - or derive from average bbox size to be scale-adaptive.
            # Here we compute an adaptive distance from median bbox size:
            widths = []
            heights = []
            for _, x1, y1, x2, y2 in self.list_bounding_boxes:
                widths.append(abs(x2 - x1))
                heights.append(abs(y2 - y1))
            if widths and heights:
                median_size = float(np.median(widths) + np.median(heights)) / 2.0
                min_dist = max(10.0, median_size * 0.6)  # e.g. 60% of median bbox size
            else:
                min_dist = 50.0

            from scipy.spatial import cKDTree
            tree = cKDTree(pts)
            visited = np.zeros(len(pts), dtype=bool)
            clusters = []

            for i in range(len(pts)):
                if visited[i]:
                    continue
                neighbors = tree.query_ball_point(pts[i], min_dist)
                visited[neighbors] = True
                cluster_center = pts[neighbors].mean(axis=0)
                clusters.append(cluster_center)

            filtered_positions = [c.tolist() for c in clusters]

        return filtered_positions

    # Helpers methods for bounding box processing
    def _boxes_overlap(self, box1, box2):
        """
        Checks if two bounding boxes overlap.

        Args:
            box1 (List): The first bounding box [id, x1, y1, x2, y2].
            box2 (List): The second bounding box [id, x1, y1, x2, y2].

        Returns:
            bool: True if the boxes overlap, False otherwise.
        """
        id_1, x1_1, y1_1, x2_1, y2_1 = box1
        id_2, x1_2, y1_2, x2_2, y2_2 = box2
        
        if id_1 != id_2:
            if x2_1 <= x1_2 or x2_2 <= x1_1:
                return False
            if y2_1 <= y1_2 or y2_2 <= y1_1:
                return False
            return True
        else:
            return False
    
    def _merge_overlapping_sublists(self, sublists):
        """
        Merges sublists that share at least one common element.

        Args:
            sublists (List[Tuple]): A list of tuples, where each tuple represents
                                    a pair of overlapping bounding box indices.

        Returns:
            List[List]: A list of lists, where each inner list contains the indices
                        of a group of mutually overlapping bounding boxes.
        """
        groups = []
        
        for sub in sublists:
            sub_set = set(sub)
            merged = False
            
            for group in groups:
                if sub_set & group:
                    group |= sub_set
                    merged = True
                    break
            
            if not merged:
                groups.append(sub_set)
        
        changed = True
        while changed:
            changed = False
            new_groups = []
            while groups:
                first, *rest = groups
                first = set(first)
                
                merged = False
                for i, other in enumerate(rest):
                    if first & other:
                        rest.pop(i)
                        first |= other
                        merged = True
                        changed = True
                        break
                new_groups.append(first)
                groups = rest
            groups = new_groups
        
        return [sorted(list(g)) for g in groups]
    
    def _compute_iou(self, box1, box2):
        """
        Computes the Intersection over Union (IoU) of two bounding boxes.

        Args:
            box1 (List): The first bounding box [id, x1, y1, x2, y2].
            box2 (List): The second bounding box [id, x1, y1, x2, y2].

        Returns:
            float: The IoU value, a float between 0.0 and 1.0.
        """
        _, x1_1, y1_1, x2_1, y2_1 = box1
        _, x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
     
    def _apply_nms(self, iou_threshold=0.7):
        """
        Applique la Non-Maximum Suppression (NMS) pour supprimer les boîtes
        englobantes qui se chevauchent fortement au sein d'une même image.
        """
        
        # 1. Grouper les boîtes par ID d'image
        boxes_by_image = defaultdict(list)
        for i, box in enumerate(self.list_bounding_boxes):
            image_id = box[0]
            # Stocker la boîte et son index original
            boxes_by_image[image_id].append((i, box))

        new_list_bounding_boxes = []
        original_indices_map = {} # Pour garder une trace des indices
        
        # 2. Appliquer NMS pour chaque image
        for image_id, boxes_with_indices in boxes_by_image.items():
            
            keep = set(range(len(boxes_with_indices))) # Indices à conserver dans le groupe
            
            for i in range(len(boxes_with_indices)):
                if i not in keep:
                    continue
                
                # Récupérer l'index original et les coordonnées de la boîte de référence
                idx1_original, box1 = boxes_with_indices[i]
                
                for j in range(i + 1, len(boxes_with_indices)):
                    if j not in keep:
                        continue
                    
                    # Récupérer l'index original et les coordonnées de la boîte à comparer
                    idx2_original, box2 = boxes_with_indices[j]
                    
                    # NMS : Si l'IoU est supérieur au seuil, on supprime la boîte j (celle de l'index supérieur)
                    # Note : On réutilise _compute_iou, qui fonctionne pour les boîtes de même ID.
                    iou = self._compute_iou(box1, box2)
                    
                    if iou >= iou_threshold:
                        keep.remove(j)
            
            # 3. Ajouter les boîtes conservées à la nouvelle liste
            for i in keep:
                original_index, box = boxes_with_indices[i]
                new_list_bounding_boxes.append(box)
                
        # Remplacer l'ancienne liste par la nouvelle liste filtrée
        self.list_bounding_boxes = new_list_bounding_boxes  
         
    # Others methods
    def reconstruct_slice(self, verbose=False):
        """
        Reconstructs a single, large image from the individual scanned tiles.

        This function reads all the saved image tiles, crops them to remove
        the overlap, and stitches them together into a final, seamless image.
        The resulting image is then saved as a JPEG.
        """
        output_path = Path(RESSOURCES_DIR) / "stitched_final.jpg"
        pattern = r"SlideScan_R(\d+)_C(\d+)_\d+\.tif"
        
        # -- 1 -- Collect image positions
        file_list = [f for f in self.scan_dir.iterdir() if f.suffix == ".tif"]
        positions = []
        
        for file_path in file_list:
            fname = file_path.name
            m = re.match(pattern, fname)
            if not m:
                continue
            grid_row = int(m.group(1))
            grid_col = int(m.group(2))
            positions.append((fname, grid_row, grid_col))
        
        # --2 -- Determine grid size
        max_x = max(p[1] for p in positions)
        max_y = max(p[2] for p in positions)
        
        sample_image = imread(self.scan_dir / positions[0][0]) # Read a sample image
        tile_h_full, tile_w_full = sample_image.shape
        tile_w_half = tile_w_full // 2 if self.dual_view else tile_w_full
        
        # Margins to crop
        margin_x = int(tile_w_half * self.overlap_percent/200)
        margin_y = int(tile_h_full * self.overlap_percent/200)
        
        # Final tile size after all crops
        crop_w = tile_w_half - 2 * margin_x
        crop_h = tile_h_full - 2 * margin_y
        
        # Final stitched image size
        stitched_height = (max_y + 1) * crop_w
        stitched_width = (max_x + 1) * crop_h
        stitched_image = np.zeros((stitched_height, stitched_width), dtype=sample_image.dtype)
        
        # -- 3 -- Stitch images
        i = 0
        for fname, x_idx, y_idx in positions:
            i += 1
            if verbose: print(f"Processing tile {i}")
            img_full = imread(self.scan_dir / fname)
            
            # 1. Crop to right half if dual view
            img_half = img_full[:, tile_w_full // 2:] if self.dual_view else img_full
            
            # 2. Remove 5% on all four sides
            img_cropped = img_half[
                margin_y: tile_h_full - margin_y,
                margin_x: tile_w_half - margin_x
            ]
            
            # Flip x to go from bottom to top (row), y is regular (col)
            row = y_idx
            col = max_x - x_idx
            
            y_pos = row * crop_h
            x_pos = col * crop_w
            stitched_image[y_pos:y_pos + crop_h, x_pos:x_pos + crop_w] = img_cropped
        
        # -- 4 -- Save final image
        img = stitched_image.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        img = (img * 255).astype(np.uint8)
        pil_image = Image.fromarray(img)
        
        if pil_image.mode != "L":
            pil_image = pil_image.convert("L")
        
        if self.scan_shape == "Square":
            pil_image = pil_image.resize((1424, 1424), Image.LANCZOS)
        else:
            pil_image = pil_image.resize((1064, 1748), Image.LANCZOS)
            pil_image = pil_image.rotate(270, expand=True)
        
        pil_image = pil_image.convert('RGB')
        
        pil_image.save(output_path, 'JPEG', quality=95)
        if verbose: print(f"✅ Final stitched image saved to: {output_path}")

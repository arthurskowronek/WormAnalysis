import re
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from tifffile import imwrite, imread
from collections import defaultdict

from config import MODELS_DIR, RESSOURCES_DIR, DATA_DIR, save_corner_positions_into_yaml_config_file

class ScanSlice:
    """
    Class for scanning slides and detecting worms using YOLO
    """
    def __init__(self, mmc, grossissement, dual_view=False, scan_shape="square", overlap_percent=10):
        """
        Initialize ScanSlice
        
        Args:
            mmc: Micro-Manager core object
            grossissement: Magnification level
            dual_view: Whether using dual view mode
            scan_shape: Shape of scan area ("square" or other)
            overlap_percent: Percentage overlap between images
        """
        self.mmc = mmc
        self.grossissement = int(grossissement.get().replace("x", ""))
        self.dual_view = dual_view.get()
        self.scan_shape = scan_shape.get()
        self.overlap_percent = overlap_percent
        print(f"ScanSlice initialized with grossissement={self.grossissement}, dual_view={self.dual_view}, scan_shape={self.scan_shape}, overlap_percent={self.overlap_percent}")
        
        # Calculate step sizes
        self.step_size_x = 13180 / self.grossissement
        self.step_size_y = self.step_size_x / 2 if self.dual_view else self.step_size_x
        
        # Load YOLO model
        self.model = YOLO(Path(MODELS_DIR) / "YOLO_detection.pt")
        
        # Storage for results
        self.list_bounding_boxes = []
        self.positions_info = []
        self.scan_dir = Path(DATA_DIR) / "Scan"
        self.scan_modified_dir = Path(DATA_DIR) / "Scan_modified"
    
    def initialize_scan(self):
        # Calculate actual steps considering overlap
        self.actual_step_x = self.step_size_x * (1 - self.overlap_percent / 100)
        self.actual_step_y = self.step_size_y * (1 - self.overlap_percent / 100)
        
        # Get starting position
        self.start_x, self.start_y = self.mmc.getXYPosition()
        
        # Calculate scan area
        end_x = self.start_x + (26000 if self.scan_shape == "square" else 45000) # TODO: récupérer ces valeurs depuis config machine
        end_y = self.start_y + 26000
        
        # Get the actual corner positions (it was the center before)
        start_corner_x = self.start_x - self.step_size_x // 2
        start_corner_y = self.start_y - self.step_size_y // 2
        end_corner_x = end_x + self.step_size_x // 2
        end_corner_y = end_y + self.step_size_y // 2
        save_corner_positions_into_yaml_config_file(start_corner_x, start_corner_y, end_corner_x, end_corner_y)
        
        # Compute the scan dimensions
        self.scan_width = int((end_x - self.start_x) / self.actual_step_x)
        self.scan_height = int((end_y - self.start_y) / self.actual_step_y)
        
        # Move to starting position
        self.mmc.setXYPosition(self.mmc.getXYStageDevice(), self.start_x, self.start_y)
        self.mmc.waitForDevice(self.mmc.getXYStageDevice())
        
        # Initialize working variables
        self.file_count = 1
        self.image = None
        self.final_end_x = 0
        self.final_end_y = 0
    
    def scan(self):
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
                if pos_x > final_end_x: final_end_x = pos_x
                if pos_y > final_end_y: final_end_y = pos_y
                
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
                print(f"Image {self.file_count-1}/{self.scan_width*self.scan_height} captured at X={pos_x:.2f}, Y={pos_y:.2f}")
        
        # Process final image
        if self.image is not None:
            imwrite(self.scan_modified_dir / self.file_name, self.image)
        
        # Return to starting position
        self.mmc.setXYPosition(self.mmc.getXYStageDevice(), self.start_x, self.start_y)
        
        return self.get_worms_position()

    def process_image_to_detect_worms(self):
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
    
    def worm_detection(self, img, id, pos_x=0, pos_y=0, drawing = False):
        """
        Detect worms in image using YOLO
        
        Args:
            img: Input image
            id: Image ID
            pos_x: X position on stage
            pos_y: Y position on stage
            
        Returns:
            tuple: (processed_image, updated_bounding_boxes_list)
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

    def get_worms_position(self):
        """Get final worm positions by merging overlapping detections"""
        overlapping_pairs = []
        best_matches = defaultdict(dict)  # {idx1: {id2: (idx2, iou)}}
        
        for i in range(len(self.list_bounding_boxes)):
            id_1 = self.list_bounding_boxes[i][0]
            for j in range(i + 1, len(self.list_bounding_boxes)):
                id_2 = self.list_bounding_boxes[j][0]
                
                if id_1 != id_2:
                    if self._boxes_overlap(self.list_bounding_boxes[i], self.list_bounding_boxes[j]):
                        iou = self._compute_iou(self.list_bounding_boxes[i], self.list_bounding_boxes[j])
                        
                        # Save best match for i with picture id_2
                        if id_2 not in best_matches[i] or iou > best_matches[i][id_2][1]:
                            best_matches[i][id_2] = (j, iou)
                        
                        # And best match for j with picture id_1
                        if id_1 not in best_matches[j] or iou > best_matches[j][id_1][1]:
                            best_matches[j][id_1] = (i, iou)
        
        # Build final list of best overlaps (ensure mutual best match)
        added_pairs = set()
        for i, matches in best_matches.items():
            for id_other, (j, _) in matches.items():
                # Only keep if mutual best match
                if best_matches[j].get(self.list_bounding_boxes[i][0], (None, -1))[0] == i:
                    pair = tuple(sorted((i, j)))
                    added_pairs.add(pair)
        
        overlapping_pairs = list(added_pairs)
        overlapping_boxes = self._merge_overlapping_sublists(overlapping_pairs)
        
        # Add non-overlapping boxes
        flat_overlapping_boxes = [item for sublist in overlapping_boxes for item in sublist]
        values_overlapping_boxes = np.unique(np.array(flat_overlapping_boxes))
        for i in range(len(self.list_bounding_boxes)):
            if i not in values_overlapping_boxes:
                overlapping_boxes.append([i])
        
        # Get centers
        positions_worms = []
        for sublist in overlapping_boxes:
            tab_x, tab_y = [], []
            for idx in sublist:
                _, x1, y1, x2, y2 = self.list_bounding_boxes[idx]
                tab_x.append((x1 + x2) / 2)
                tab_y.append((y1 + y2) / 2)
            x = sum(tab_x) / len(tab_x)
            y = sum(tab_y) / len(tab_y)
            positions_worms.append([x, y])
        
        return positions_worms
    
    # Helpers methods for bounding box processing
    def _boxes_overlap(self, box1, box2):
        """Check if two bounding boxes overlap"""
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
        """Merge overlapping sublists"""
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
        """Compute Intersection over Union (IoU) for two boxes"""
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
        
    # Others methods
    def reconstruct_slice(self):
        """Reconstruct the full slice from individual tiles"""
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
        
        # Margins to crop: 5% on each side
        margin_x = int(tile_w_half * 0.05)
        margin_y = int(tile_h_full * 0.05)
        
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
            print(f"Processing tile {i}")
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
        
        if self.scan_shape == "square":
            pil_image = pil_image.resize((1424, 1424), Image.LANCZOS)
        else:
            pil_image = pil_image.resize((1064, 1748), Image.LANCZOS)
            pil_image = pil_image.rotate(270, expand=True)
        
        pil_image = pil_image.convert('RGB')
        
        pil_image.save(output_path, 'JPEG', quality=95)
        print(f"✅ Final stitched image saved to: {output_path}")

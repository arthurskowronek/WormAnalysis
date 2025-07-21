import re
import os
import cv2
import time
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from tifffile import imsave, imread
from collections import defaultdict

from config import MODELS_DIR, RESSOURCES_DIR, DATA_DIR

def worm_detection(img: np.ndarray, Step_Size_X, Step_Size_Y, id: int, list_bounding_boxes, pos_x=0, pos_y=0) -> np.ndarray:
    """
    Segment worm from background using YOLO
    
    Args:
        img: Input image
        
    Returns:
        image: Image with detected worms
        list_bounding_boxes: List of bounding boxes for detected worms
    """
    
    model = YOLO(Path(MODELS_DIR) / "YOLO_detection.pt")
    image = img.copy()

    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Save temporary image
    temp_path = Path(MODELS_DIR) / "temp_converted_image.png"
    cv2.imwrite(str(temp_path), image)
    
    # Predict
    prediction = model.predict(source=str(temp_path), save=False, verbose=False)
    os.remove(temp_path)
    #temp_path.unlink()  # Remove temp file
    
    boxes = prediction[0].boxes

    if boxes is not None:
        bounding_boxes = boxes.xyxy.cpu().numpy()
        for bbox in bounding_boxes:
            
            print("Worm detected")
            
            x1, y1, x2, y2 = bbox

            # offsets from image center
            H, W = image.shape[:2]
            dx_pix1 = x1/W - 0.5
            dy_pix1 = y1/H - 0.5
            dx_pix2 = x2/W - 0.5
            dy_pix2 = y2/H - 0.5

            # convert to stage microns
            Step = max(Step_Size_X, Step_Size_Y)
            dx_um1 = dx_pix1 * Step
            dy_um1 = dy_pix1 * Step
            dx_um2 = dx_pix2 * Step
            dy_um2 = dy_pix2 * Step

            # true worm position on stage
            x_worm1 = pos_x + dy_um1
            y_worm1 = pos_y - dx_um1
            x_worm2 = pos_x + dy_um2
            y_worm2 = pos_y - dx_um2

            list_bounding_boxes.append([id, x_worm1, y_worm1, x_worm2, y_worm2])
            
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 4)
        
    return image, list_bounding_boxes         

def _boxes_overlap(box1, box2):
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

def _merge_overlapping_sublists(sublists):
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

def _compute_iou(box1, box2):
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

def get_worms_position(list_bounding_boxes):

    overlapping_pairs = []
    best_matches = defaultdict(dict)  # {idx1: {id2: (idx2, iou)}}

    for i in range(len(list_bounding_boxes)):
        id_1 = list_bounding_boxes[i][0]
        for j in range(i + 1, len(list_bounding_boxes)):
            id_2 = list_bounding_boxes[j][0]

            if id_1 != id_2:
                if _boxes_overlap(list_bounding_boxes[i], list_bounding_boxes[j]):
                    iou = _compute_iou(list_bounding_boxes[i], list_bounding_boxes[j])

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
            if best_matches[j].get(list_bounding_boxes[i][0], (None, -1))[0] == i:
                pair = tuple(sorted((i, j)))
                added_pairs.add(pair)

    overlapping_pairs = list(added_pairs)
    overlapping_boxes = _merge_overlapping_sublists(overlapping_pairs)

    # Add non-overlapping boxes
    flat_overlapping_boxes = [item for sublist in overlapping_boxes for item in sublist]
    values_overlapping_boxes = np.unique(np.array(flat_overlapping_boxes))
    for i in range(len(list_bounding_boxes)):
        if i not in values_overlapping_boxes:
            overlapping_boxes.append([i])

    # Get centers
    positions_worms = []
    for sublist in overlapping_boxes:
        tab_x, tab_y = [], []
        for idx in sublist:
            _, x1, y1, x2, y2 = list_bounding_boxes[idx]
            tab_x.append((x1 + x2) / 2)
            tab_y.append((y1 + y2) / 2)
        x = sum(tab_x) / len(tab_x)
        y = sum(tab_y) / len(tab_y)
        positions_worms.append([x, y])

    return positions_worms

def transform_positions_into_proportion(end_x, end_y, positions_worms, start_x, start_y, Step_Size_X, Step_Size_Y):
    # transform position into proportion of the scan
    start_corner_x = start_x - Step_Size_X//2
    start_corner_y = start_y - Step_Size_Y//2
    end_corner_x = end_x + Step_Size_X//2
    end_corner_y = end_y + Step_Size_Y//2
    
    # write paramters in a csv file
    csv_path = Path(RESSOURCES_DIR) / "config_positions.csv"
    pd.DataFrame({
        'start_corner_x': [start_corner_x],
        'start_corner_y': [start_corner_y],
        'end_corner_x': [end_corner_x],
        'end_corner_y': [end_corner_y]
    }).to_csv(csv_path, index=False)
    
    positions_worms_proportion = []
    
    for worm in positions_worms:
        x = worm[0]
        y = worm[1]
        x = (x - start_corner_x)/(end_corner_x - start_corner_x)
        y = (y - start_corner_y)/(end_corner_y - start_corner_y)
        # 0,0 is in the top right corner, so we need to change the origin
        x_prop = 1-y
        y_prop = x
        positions_worms_proportion.append([x_prop,y_prop])
    
    return positions_worms_proportion

def reconstructSlice(name_directory, DualView=False, Shape = "square"):
    # === Configuration ===
    output_path = Path(RESSOURCES_DIR) / "stitched_final.tif"
    pattern = r"SlideScan_R(\d+)_C(\d+)_\d+\.tif"

    # === Collect image positions ===
    file_list = [f for f in os.listdir(name_directory) if f.endswith(".tif")]
    positions = []
    
    for fname in file_list:
        m = re.match(pattern, fname)
        if not m: continue
        grid_row = int(m.group(1))
        grid_col = int(m.group(2))
        positions.append((fname, grid_row, grid_col))

    # Determine grid size
    max_x = max(p[1] for p in positions)
    max_y = max(p[2] for p in positions)

    # Read a sample image
    sample_image = imread(os.path.join(name_directory, positions[0][0]))
    tile_h_full, tile_w_full = sample_image.shape
    tile_w_half = tile_w_full // 2 if DualView else tile_w_full  # keep right half only

    # Margins to crop: 5% on each side
    margin_x = int(tile_w_half * 0.05)  # left & right
    margin_y = int(tile_h_full * 0.05)  # top & bottom

    # Final tile size after all crops
    crop_w = tile_w_half - 2 * margin_x
    crop_h = tile_h_full - 2 * margin_y

    # Final stitched image size
    stitched_height = (max_y + 1) * crop_w
    stitched_width = (max_x + 1) * crop_h
    stitched_image = np.zeros((stitched_height, stitched_width), dtype=sample_image.dtype)

    # === Stitch images ===
    i=0
    for fname, x_idx, y_idx in positions:
        i += 1
        print(i)
        img_full = imread(os.path.join(name_directory, fname))

        # 1. Crop to right half
        img_half = img_full[:, tile_w_full // 2:] if DualView else img_full

        # 2. Remove 5% on all four sides
        img_cropped = img_half[
            margin_y : tile_h_full - margin_y,
            margin_x : tile_w_half - margin_x
        ]

        # Flip x to go from bottom to top (row), y is regular (col)
        row = y_idx
        col = max_x - x_idx

        y_pos = row * crop_h
        x_pos = col * crop_w
        stitched_image[y_pos:y_pos+crop_h, x_pos:x_pos+crop_w] = img_cropped
    

    # === Save final image ===
    img = stitched_image.astype(np.float32)
    img = (img - img.min())/(img.max() - img.min())
    img = (img * 255).astype(np.uint8)
    pil_image = Image.fromarray(img)
    pil_image.save(output_path)
    print(f"✅ Final stitched image saved to: {output_path}")
    if pil_image.mode != "L":
        pil_image = pil_image.convert("L")

    if Shape == "square":
        pil_image = pil_image.resize((1424, 1424), Image.LANCZOS) 
    else:
        pil_image = pil_image.resize((1064, 1748), Image.LANCZOS)
        pil_image = pil_image.rotate(270, expand=True)
    pil_image = pil_image.convert('RGB')
    pil_image = np.array(pil_image)

    return pil_image
    
def ScanSlice(mmc, Grossissement, DualView, Scan_shape):
    """Fonction principale pour balayer la lame et capturer des images"""
    

    # Initialisation 
    Step_Size_X = 13180/Grossissement
    Step_Size_Y = Step_Size_X/2 if DualView else Step_Size_X
    Overlap_Percent = 10

    # Créer le dossier de captures s'il n'existe pas
    name_directory = Path(DATA_DIR) / "Scan"
    if not os.path.exists(name_directory):
        os.makedirs(name_directory)
    else:
        for f in os.listdir(name_directory):
            os.remove(os.path.join(name_directory, f))
            
    # Créer le dossier de captures s'il n'existe pas
    name_directory = Path(DATA_DIR) / "Scan_modified"
    if not os.path.exists(name_directory):
        os.makedirs(name_directory)
    else:
        for f in os.listdir(name_directory):
            os.remove(os.path.join(name_directory, f))

    
    # Désactiver l'obturateur automatique
    mmc.setAutoShutter(False)
    mmc.setShutterOpen(True)
    
    # Obtenir la position de départ
    start_x, start_y = mmc.getXYPosition()

    # Calculer les pas réels en tenant compte du chevauchement
    actual_step_x = Step_Size_X * (1 - Overlap_Percent / 100)
    actual_step_y = Step_Size_Y * (1 - Overlap_Percent / 100)
    end_x = start_x + 26000 if Scan_shape == "square" else start_x + 45000
    end_y = start_y + 26000 
    Scan_Width = int((end_x - start_x)/actual_step_x)
    Scan_Height = int((end_y - start_y)/actual_step_y)


    
    # Créer un tableau pour stocker les informations de position
    positions_info = []
    list_bounding_boxes = []
    
    # Compteur pour les fichiers
    file_count = 1

    # Se déplacer à la position de départ
    mmc.setXYPosition(mmc.getXYStageDevice(), start_x, start_y)
    mmc.waitForDevice(mmc.getXYStageDevice())
    
    """print(f"Démarrage du scan: {Scan_Width}x{Scan_Height} positions")
    print(f"Position de départ: X={start_x}, Y={start_y}")
    print(f"Pas en X: {actual_step_x} µm, Pas en Y: {actual_step_y} µm")
    print(f"Chevauchement: {Overlap_Percent}%")"""

    start_time = time.time()
    image = None
    end_x = 0
    end_y = 0

    # Boucles pour scanner la grille
    for y_idx in range(Scan_Height):
        # Direction alternée en X pour un balayage en serpentin (plus efficace)
        x_range = range(Scan_Width) if y_idx % 2 == 0 else range(Scan_Width-1, -1, -1)
        
        for x_idx in x_range:
            # Calculer la position absolue
            pos_x = start_x + x_idx * actual_step_x
            pos_y = start_y + y_idx * actual_step_y
            mmc.setXYPosition(mmc.getXYStageDevice(), pos_x, pos_y)
            
            if pos_x > end_x:
                end_x = pos_x
            if pos_y > end_y:
                end_y = pos_y

            # Find worms
            if image is not None:
                last_pos_x = positions_info[-1][1]
                last_pos_y = positions_info[-1][2]
                _, tile_w_full = image.shape
                if DualView:
                    img_half = image[:, tile_w_full // 2:]
                    img_half_left = image[:, :tile_w_full // 2]
                    img_half_left = cv2.normalize(img_half_left, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    img_down_right = img_half[tile_w_full // 2:, :]
                    img_down_right, list_bounding_boxes = worm_detection(img_down_right, Step_Size_X, Step_Size_Y, file_count-1, list_bounding_boxes, last_pos_x, last_pos_y)
                    img_up_right = img_half[:tile_w_full // 2, :]
                    img_up_right, list_bounding_boxes = worm_detection(img_up_right, Step_Size_X, Step_Size_Y, file_count-1, list_bounding_boxes, last_pos_x, last_pos_y)
                    img_half = np.vstack([img_up_right, img_down_right])
                    image = np.hstack([img_half_left, img_half])
                    imsave(f"{Path(DATA_DIR)} /Scan_modified/ {file_name}", image)
                else:
                    image, list_bounding_boxes = worm_detection(image, Step_Size_X, Step_Size_Y, file_count-1, list_bounding_boxes, last_pos_x, last_pos_y)
                    imsave(f"{Path(DATA_DIR)}/Scan_modified/{file_name}", image)


            # Attendre que le déplacement soit terminé
            mmc.waitForDevice(mmc.getXYStageDevice())
            
            # Capturer l'image
            mmc.snapImage()
            image = mmc.getImage()
            # Use R for the grid‐row (y_idx) and C for the grid‐col (x_idx)
            file_name = f"SlideScan_R{y_idx}_C{x_idx}_{file_count}.tif"
            imsave(f"{Path(DATA_DIR)}/Scan/{file_name}", image)
        
            
            # Enregistrer les informations de position
            positions_info.append([file_count, pos_x, pos_y, x_idx, y_idx])
            file_count += 1
            print(f"Image {file_count-1}/{Scan_Width*Scan_Height} capturée à X={pos_x:.2f}, Y={pos_y:.2f}")


    imsave(f"{Path(DATA_DIR)}/Scan_modified/{file_name}", image)


    end_time = time.time()
    total_sec = end_time - start_time
    nb_min = int((total_sec)/60)
    nb_sec = round(total_sec-60*nb_min,2)
    
    print(f"Scan terminé! {file_count - 1} images ont été capturées en {nb_min}min et {nb_sec}s.")

    positions_worms = get_worms_position(list_bounding_boxes)    
    positions_worms_proportion = transform_positions_into_proportion(end_x, end_y, positions_worms, start_x, start_y, Step_Size_X, Step_Size_Y)

    # Revenir à la position de départ
    mmc.setXYPosition(mmc.getXYStageDevice(), start_x, start_y)
    
    # Fermer l'obturateur
    mmc.setShutterOpen(False)
    
    stiching_img = reconstructSlice(Path(DATA_DIR) / "Scan", DualView, Scan_shape)

    return positions_worms, positions_worms_proportion, stiching_img



import cv2
import numpy as np
from pathlib import Path

# --- Configuration ---
INPUT_DIR = Path("data/images_to_transform2")
OUTPUT_DIR = Path("data/images_to_annotate_test_clahe")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialisation du CLAHE
# Ajustez clipLimit (ex: 1.0, 2.0, 4.0) pour tester la force du contraste
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))

# --- Traitement ---
image_files = list(INPUT_DIR.glob("*.tif"))

for i, image_path in enumerate(image_files):
    # Lecture (UNCHANGED pour gérer le 8 ou 16 bits)
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    
    if img is None:
        continue

    # Conversion en 8-bits nécessaire pour le CLAHE d'OpenCV
    if img.dtype != np.uint8:
        img_8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    else:
        img_8bit = img

    # Application du CLAHE
    image_clahe = clahe.apply(img_8bit)

    # Sauvegarde
    filename = image_path.stem + ".png"
    new_path = OUTPUT_DIR / filename
    
    cv2.imwrite(str(new_path), image_clahe)
    print(f"[{i+1}/{len(image_files)}] Traité : {filename}")

print("\nTest CLAHE terminé.")
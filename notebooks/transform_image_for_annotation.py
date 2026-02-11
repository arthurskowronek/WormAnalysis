import cv2
import numpy as np
from pathlib import Path

# Load images
DIRECTORY = Path("data/images_to_transform")
TYPE = "Detection"

for i, image_path in enumerate(DIRECTORY.glob("*.tif")):
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    image = img.copy()
         
    if TYPE == "Segmentation":
        image = image.astype(np.float32)
        vmin = np.percentile(image, 0.5)
        vmax = np.percentile(image, 99.5)
        if vmax <= vmin:
            vmax = vmin + 1.0
        # Scale to 0-255 and clip
        image_normalized = np.clip((image - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
    elif TYPE == "Detection":
        image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Save image
    filename = image_path.stem + ".png" 
    if TYPE == "Segmentation":
        new_path = Path("data/images_to_annotate_for_segmentation") / filename
    elif TYPE == "Detection":
        new_path = Path("data/images_to_annotate_for_detection") / filename
        
    cv2.imwrite(str(new_path), image_normalized)
    print(f"Saved transformed image : {i}")
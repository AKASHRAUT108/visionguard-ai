import numpy as np
from PIL import Image
import torch
from segment_anything import sam_model_registry, SamPredictor
import os

class Segmenter:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Path to the downloaded SAM checkpoint
        checkpoint = os.path.join(os.path.dirname(__file__), "sam_vit_h_4b8939.pth")
        if not os.path.exists(checkpoint):
            print("SAM checkpoint not found. Segmentation will be disabled.")
            self.predictor = None
            return

        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(self.device)
        self.predictor = SamPredictor(sam)

    def segment(self, image: Image.Image):
        if self.predictor is None:
            return None

        # Convert PIL to numpy array (RGB)
        img = np.array(image.convert("RGB"))
        self.predictor.set_image(img)

        # Generate masks with a simple prompt (e.g., center point)
        # In a real system, you'd use the defect classifier's bounding box or a learned prompt.
        # For demo, we use a point in the center.
        h, w = img.shape[:2]
        input_point = np.array([[w//2, h//2]])
        input_label = np.array([1])  # foreground point

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )

        # Return the mask with the highest score
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        return mask.tolist()  # convert to list for JSON serialization
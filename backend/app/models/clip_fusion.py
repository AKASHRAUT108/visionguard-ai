from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

class CLIPFusion:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def fuse(self, image: Image.Image, text: str):
        if not text:
            return 0.0
        inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # logits_per_image gives image-text similarity score
            similarity = outputs.logits_per_image.item()
        return similarity
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import os
import random

class DefectClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "Ahmed-Abdelkhalek/vit-base-patch16-224-mvtec-ad"  # Example – may need replacement
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
            self.model = AutoModelForImageClassification.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            self.ready = True
        except Exception as e:
            print(f"Could not load pretrained model: {e}")
            print("Using dummy classifier for demonstration.")
            self.ready = False

        # Define image transforms (used if model loads or for dummy)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Defect categories (simplified)
        self.categories = ["scratch", "dent", "color_stain", "crack", "hole", "normal"]

    def predict(self, image: Image.Image):
        if self.ready:
            # Preprocess using the model's feature extractor
            inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                confidence, pred_idx = torch.max(probs, dim=-1)
                predicted_class = self.model.config.id2label[pred_idx.item()]
                # For MVTec, the class could be the defect type, or "good" vs "defect".
                # We'll map to our categories if needed. For simplicity, return as is.
                return {
                    "defect_class": predicted_class,
                    "confidence": confidence.item()
                }
        else:
            # Dummy prediction for demo
            return {
                "defect_class": random.choice(self.categories),
                "confidence": random.uniform(0.6, 0.99)
            }
from transformers import pipeline

class RootCauseClassifier:
    def __init__(self):
        # Use a zero-shot classification model
        self.classifier = pipeline("zero-shot-classification",
                                   model="facebook/bart-large-mnli")
        self.candidate_labels = [
            "Machine Calibration",
            "Raw Material Defect",
            "Operator Error",
            "Environmental Factor",
            "Wear & Tear"
        ]

    def predict(self, text: str):
        if not text:
            return {"root_cause": "No text provided", "confidence": 0.0}
        result = self.classifier(text, self.candidate_labels)
        return {
            "root_cause": result['labels'][0],
            "confidence": result['scores'][0]
        }
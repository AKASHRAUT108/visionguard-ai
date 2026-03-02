import random

class FailurePredictor:
    def __init__(self):
        pass

    def predict(self, sensor_data=None):
        """
        Dummy predictor: returns a random failure probability.
        In a real implementation, you'd load a trained LSTM model.
        """
        return round(random.uniform(0, 1), 2)
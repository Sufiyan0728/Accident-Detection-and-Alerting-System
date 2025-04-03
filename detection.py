import tensorflow as tf
from tensorflow.keras.models import model_from_json

class AccidentDetectionModel:
    def __init__(self, model_json_file, model_weights_file):
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.model = model_from_json(loaded_model_json)
            self.model.load_weights(model_weights_file)
    
    def predict_accident(self, img):
        self.preds = self.model.predict(img)
        return "Accident" if self.preds[0][0] > 0.5 else "No Accident", self.preds
import joblib
import json
import numpy as np
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path('best_rf_model')
    model = joblib.load(model_path)

def run(raw_data):
    data = np.array(json.loads(raw_data)['data']).reshape(1, -1)
    prediction = model.predict(data)
    return json.dumps({"prediction": prediction.tolist()})

from azureml.core.model import Model
import json
import numpy as np
import os
import pickle
import joblib



def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'auto_ml_model')
    model = joblib.load(model_path)

# input_sample = np.array([[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
# output_sample = np.array([3726.995])

def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    # make prediction
    y_hat = model.predict(data)
    # you can return any data type as long as it is JSON-serializable
    return y_hat.tolist()

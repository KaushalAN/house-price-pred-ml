from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model (assuming you've saved it as a .pkl file)
model = pickle.load(open('estimator.pkl', 'rb'))
@app.route('/')
def home():
    return "House Price Prediction API"
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']  # expecting a JSON object with 'data' key
    prediction = model.predict(np.array(data).reshape(1, -1))  # adjust based on your model
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
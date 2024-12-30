from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('classmodel.pkl', 'rb'))  # Update with your model file name
encoder =pickle.load(open('labelencoder.pkl','rb'))

# Define a function for prediction
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    pred = model.predict(input_data)[0]
    prediction = (encoder.inverse_transform(pred.reshape(1,-1)))[0]
    return prediction

# Route for the main page
@app.route('/')
def home():
    return render_template('home.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    sepal_length = float(data['sepal_length'])
    sepal_width = float(data['sepal_width'])
    petal_length = float(data['petal_length'])
    petal_width = float(data['petal_width'])
    
    species = predict_species(sepal_length, sepal_width, petal_length, petal_width)
    return jsonify({'species': species})

if __name__ == '__main__':
    app.run(debug=True)

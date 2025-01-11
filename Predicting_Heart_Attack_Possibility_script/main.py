from flask import Flask, request, jsonify
from flask_cors import CORS  # To handle Cross-Origin Resource Sharing (CORS)
import joblib  # For loading the trained model

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load the pre-trained model (update path to where you saved the model)
model = joblib.load('C:/Users/Sarakhodiry/Downloads/Classification_best_model.pkl')


@app.route('/')
def home():
    return "Welcome to the Heart Attack Prediction API!"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request (should be in JSON format)
        data = request.get_json()

        # Extract features from the data
        features = [data['age'], data['sex'], data['cp'], data['trestbps'], data['chol'],
                    data['fbs'], data['restecg'], data['thalach'], data['exang'],
                    data['oldpeak'], data['slope'], data['ca'], data['thal']]

        # Make a prediction using the model
        prediction = model.predict([features])

        # Return the prediction result as JSON
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        # Handle any errors and return them
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    # Run the app with debug mode for easier development and debugging
    app.run(debug=True)

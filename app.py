from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

# Load the trained model
with open('fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

# Define feature names
FEATURES = ['TX_AMOUNT', 'DAY', 'HOUR', 'TERMINAL_FRAUD_RATIO', 'CUSTOMER_FRAUD_RATIO']

@app.route('/')
def home():
    """
    Home page to input transaction data.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict endpoint for fraud detection.
    """
    try:
        # Parse input data from the form
        data = {
            "TX_AMOUNT": float(request.form['TX_AMOUNT']),
            "DAY": int(request.form['DAY']),
            "HOUR": int(request.form['HOUR']),
            "TERMINAL_FRAUD_RATIO": float(request.form['TERMINAL_FRAUD_RATIO']),
            "CUSTOMER_FRAUD_RATIO": float(request.form['CUSTOMER_FRAUD_RATIO'])
        }

        # Convert input data to DataFrame
        df = pd.DataFrame([data])

        # Make predictions
        prediction = model.predict(df[FEATURES])

        # Return the prediction result as a response
        result = "Fraudulent" if prediction[0] == 1 else "Legitimate"
        return render_template('result.html', result=result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

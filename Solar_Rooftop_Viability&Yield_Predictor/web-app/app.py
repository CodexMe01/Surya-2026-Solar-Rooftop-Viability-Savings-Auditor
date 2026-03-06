import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# 1. Load the Models AND the Scaler
modelViable = pickle.load(open("./SolarViable.pkl", "rb"))
modelSaving = pickle.load(open("./SolarSavingPred.pkl", "rb"))
scaler = pickle.load(open("./scaler.pkl", "rb")) # You must save/pickle this from Colab

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # The string inside .get() MUST match the 'name' attribute in HTML
        input_data = {
            'monthly_bill': float(request.form.get('monthly_bill', 0)),
            'roof_area': float(request.form.get('roof_area', 0)),
            'shading_perc': float(request.form.get('shading_perc', 0)),
            'irradiance': 5.0, # Placeholder for now
            'dust_index': 3.0   # Placeholder for now
        }
        
        # ... rest of your scaling and prediction code
        
    except TypeError as e:
        return f"Error: Data mismatch. {e}"
    # 3. Convert to DataFrame (Matches column order of training data)
    df_input = pd.DataFrame([input_data])
    
    # 4. SCALE THE INPUT (The most important step)
    scaled_features = scaler.transform(df_input)

    # 5. Predict Viability
    viability = modelViable.predict(scaled_features)

    if viability[0] == 1:
        savings = modelSaving.predict(scaled_features)
        # Assuming savings is a numpy array, get the value and format it
        output = f"Viable! Estimated Annual Savings: ₹{round(float(savings[0]), 2)}"
    else:
        output = "Not Viable to install solar panels based on current shading/cost."

    return render_template(
        "index.html", prediction_text=output
    )

if __name__ == "__main__":
    app.run(debug=True)
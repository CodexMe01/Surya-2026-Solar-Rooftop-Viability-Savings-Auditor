*Surya 2026: Solar Rooftop Viability and Savings Auditor
Project Overview
Surya 2026 is an end-to-end Machine Learning web application designed to help Indian homeowners assess the feasibility of solar energy for their rooftops. The tool uses a Dual-Stage ML Pipeline to first classify viability and then predict precise annual savings.

Key Features
Intelligent Gatekeeping: Uses a Logistic Regression classifier to determine if a rooftop is viable based on shading, area, and utility costs.

Precise Yield Prediction: Employs a Linear Regression model to estimate annual savings in INR for viable locations.

Environmental Sensitivity: Accounts for critical Indian factors such as Irradiance (solar intensity) and Dust Index (efficiency loss due to pollution).

Industrial Preprocessing: Implements StandardScaler transformations to ensure unbiased feature weightage and faster gradient descent convergence.

Tech Stack
Backend: Python (Flask)

Machine Learning: Scikit-Learn, Pandas, NumPy

Frontend: HTML5, CSS3

Model Management: Pickle for model and scaler serialization

Project Architecture
The project follows a modular Industrial Pipeline:

Data Layer: Synthetic dataset modeling 1,000+ Indian households with features like monthly bill, roof area, and shading percentage.

Preprocessing: Feature selection via Pearson Correlation Matrix to avoid multicollinearity and scaling via Z-score normalization.

Model Layer:

Model A (Classifier): Predicts viability (1/0) using Logistic Regression.

Model B (Regressor): Predicts estimated annual savings for viable inputs using Linear Regression.

Deployment: Flask-based web interface for real-time inference.

Repository Structure
app.py: Flask Application logic and API endpoints.

SolarViable.pkl: Pre-trained Logistic Regression model.

SolarSavingPred.pkl: Pre-trained Linear Regression model.

scaler.pkl: Saved StandardScaler object containing training mean and standard deviation.

requirements.txt: List of dependencies for deployment.

templates/: Folder containing the HTML UI.

static/: Folder containing CSS and design assets.

Installation and Local Setup
Clone the repository:
git clone https://github.com/yourusername/surya-2026.git

Install Dependencies:
pip install -r requirements.txt

Run the Application:
python app.py

Access the UI:
Open http://127.0.0.1:5000 in your web browser.

Data Preprocessing Logic
To prevent data leakage, the StandardScaler was fitted only on the training data. The resulting mean and standard deviation are stored in scaler.pkl and applied to user inputs in real-time during prediction to ensure the model receives data in the exact format it was trained on.

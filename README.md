# Surya 2026: Solar Rooftop Viability and Savings Auditor

## Project Overview
Surya 2026 is an end-to-end Machine Learning web application designed to help Indian homeowners assess the feasibility of solar energy for their rooftops. The tool uses a dual-stage ML pipeline to first classify viability and then predict precise annual savings.

## Key Features
* **Intelligent Gatekeeping**: Uses a Logistic Regression classifier to determine if a rooftop is viable based on shading, area, and utility costs.
* **Precise Yield Prediction**: Employs a Linear Regression model to estimate annual savings in INR for viable locations.
* **Environmental Sensitivity**: Accounts for critical factors such as Irradiance (solar intensity) and Dust Index (efficiency loss due to pollution).
* **Industrial Preprocessing**: Implements StandardScaler transformations to ensure unbiased feature weightage and faster gradient descent convergence.



## Tech Stack
* **Backend**: Python (Flask)
* **Machine Learning**: Scikit-Learn, Pandas, NumPy
* **Frontend**: HTML5, CSS3
* **Model Management**: Pickle for model and scaler serialization

## Project Architecture
The project follows a modular Industrial Pipeline:
1. **Data Layer**: Synthetic dataset modeling 1,000+ Indian households with features like monthly bill, roof area, and shading percentage.
2. **Preprocessing**: Feature selection via Pearson Correlation Matrix to avoid multicollinearity and scaling via Z-score normalization.
3. **Model Layer**:
    * **Model A (Classifier)**: Predicts viability (1/0) using Logistic Regression.
    * **Model B (Regressor)**: Predicts estimated annual savings for viable inputs using Linear Regression.
4. **Deployment**: Flask-based web interface for real-time inference.



## Repository Structure
* **app.py**: Flask Application logic and API endpoints.
* **SolarViable.pkl**: Pre-trained Logistic Regression model.
* **SolarSavingPred.pkl**: Pre-trained Linear Regression model.
* **scaler.pkl**: Saved StandardScaler object containing training mean and standard deviation.
* **requirements.txt**: List of dependencies for deployment.
* **templates/**: Folder containing the HTML UI.
* **static/**: Folder containing CSS and design assets.

## Installation and Local Setup

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/yourusername/surya-2026.git](https://github.com/yourusername/surya-2026.git)

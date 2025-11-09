# 🛠️ Pipe Thickness Loss Predictor

A machine learning powered pipeline health assessment system that predicts pipeline condition based on operational parameters. This application uses a trained Random Forest model to analyze inputs like pipe size, thickness, corrosion, temperature and more — and provides a prediction with visual feedback for maintenance decision-making.

## 📘 Overview

The Pipe Thickness Loss Predictor is a Flask-based Machine Learning web application designed to predict the condition of oil and gas pipelines (Normal, Moderate, or Critical) using pipeline parameters such as size, thickness, material, grade, pressure, temperature, and corrosion impact.
It empowers maintenance teams to detect risks early, prevent pipeline failures, and optimize maintenance scheduling through intelligent predictions and visual insights.

## Key Features ✨

* **Predictive Analytics**: Evaluate pipeline health condition using ML inference based on real operational parameters measured in the field.
* **Data-Driven Decision Support**: Capture user inputs and predicted outputs in a log for future retraining, traceability, and long-term pipeline performance monitoring.
* **Interactive Input UI**: Clean and user-friendly front-end interface for parameter entry, styled with industrial/energy-themed background visuals.
* **Color-Based Visual Feedback**: Predictions are shown with color indicators for immediate interpretation and high situational awareness.
* **Real-Time ML Inference**: Utilizes classical ML algorithm (Random Forest) for fast and reliable predictions on live user inputs.

## Technology Stack 🛠️

* **Frontend**:	HTML, CSS, JavaScript
* **Backend	Python**: (Flask Framework)
* **Machine Learning**: Scikit-learn, Pandas, NumPy
* **Visualization**:Matplotlib
* **Storage**: CSV-based logging
* **Deployment Ready**: Render / Hugging Face Spaces / Railway

---

## 🚀 Setup and Installation

### Prerequisites

* Python 3.x installed on your system
* pip installed


### 1. Clone the Repository

```bash
git clone https://github.com/your-username/pipe-thickness-loss-predictor.git
cd pipe-thickness-loss-predictor
```

### 2. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install all the required Python libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```


### 4. Run the Application

```bash
python app.py
```

### 5. Open in Browser

```bash
http://127.0.0.1:5500/
```

## 📂 Project Structure

```bash
PipeThicknessLossPredictor/
│
├── app.py                     # Main Flask application
├── model.pkl                  # Trained Random Forest model
├── predictions_log.csv         # Saved user input + predictions
│
├── static/
│   ├── style.css              # Styling and animations
│   └── bg.jpg                 # Background image
│
├── templates/
│   ├── index.html             # Input form page
│   └── result.html            # Prediction result page
│
├── market_pipe_thickness_loss_dataset.csv  # Dataset used for training
└── requirements.txt
```

## 🧠 Model Details

* **Algorithm**: Random Forest Classifier
* **Input Features**:
Pipe Size (mm), Thickness (mm), Material, Grade, Max Pressure (psi), Temperature (°C),
Corrosion Impact (%), Thickness Loss (mm), Material Loss (%), Time (Years)
* **Output**: Condition → Normal, Moderate, Critical

## Evaluation:

Accuracy: ~95%

Classification Report & Confusion Matrix included


from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict()

        # Convert inputs to correct data types
        input_data = {
            "Pipe_Size_mm": float(data["Pipe_Size_mm"]),
            "Thickness_mm": float(data["Thickness_mm"]),
            "Material": data["Material"],
            "Grade": data["Grade"],
            "Max_Pressure_psi": float(data["Max_Pressure_psi"]),
            "Temperature_C": float(data["Temperature_C"]),
            "Corrosion_Impact_Percent": float(data["Corrosion_Impact_Percent"]),
            "Thickness_Loss_mm": float(data["Thickness_Loss_mm"]),
            "Material_Loss_Percent": float(data["Material_Loss_Percent"]),
            "Time_Years": int(data["Time_Years"])
        }

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]

        # Add prediction result before saving
        input_data["Prediction"] = prediction

        # Safe CSV saving logic
        log_path = "predictions_log.csv"

        # If file exists and is valid
        if os.path.exists(log_path):
            try:
                existing_df = pd.read_csv(log_path)
                updated_df = pd.concat([existing_df, pd.DataFrame([input_data])], ignore_index=True)
                updated_df.to_csv(log_path, index=False)
            except pd.errors.EmptyDataError:
                # If file is empty, rewrite with headers
                pd.DataFrame([input_data]).to_csv(log_path, index=False)
        else:
            # If file doesn’t exist, create new with headers
            pd.DataFrame([input_data]).to_csv(log_path, index=False)

        return render_template("result.html", prediction=prediction)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True, port=5500)

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


'''@app.route("/predict", methods=["POST"])
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

        # 🔹 Add prediction result to the data before saving
        input_data["Prediction"] = prediction

        # 🔹 Save to CSV (append mode)
        log_path = "predictions_log.csv"
        if os.path.exists(log_path):
            existing_df = pd.read_csv(log_path)
            updated_df = pd.concat([existing_df, pd.DataFrame([input_data])], ignore_index=True)
            updated_df.to_csv(log_path, index=False)
        else:
            pd.DataFrame([input_data]).to_csv(log_path, index=False)

        explanations = {
            "Normal": "✅ Pipe is safe. No immediate action required.",
            "Moderate": "⚠️ Pipe shows moderate wear. Monitor closely.",
            "Critical": "❌ Pipe is critical. Replacement/repair needed urgently."
        }

        # 🔹 Render result page
        return render_template("result.html",
                               prediction=prediction,
                               explanation=explanations.get(prediction, ""))

    except Exception as e:
        return f"Error: {str(e)}"'''


'''@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict()

        # Prepare input
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

        explanations = {
            "Normal": "✅ Pipe is safe. No immediate action required.",
            "Moderate": "⚠️ Pipe shows moderate wear. Monitor closely.",
            "Critical": "❌ Pipe is critical. Replacement/repair needed urgently."
        }

        return render_template("result.html",
                               prediction=prediction,
                               explanation=explanations.get(prediction, ""))

    except Exception as e:
        return f"Error: {str(e)}"'''

if __name__ == "__main__":
    app.run(debug=True, port=5500)


'''from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained pipeline model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict()

        # Convert form inputs to correct types
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

        # Create DataFrame for model
        input_df = pd.DataFrame([input_data])

        # Run prediction
        prediction = model.predict(input_df)[0]

        return render_template("index.html", prediction=prediction)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Run on port 5500 since you use that
    app.run(debug=True, port=5500)'''


'''# Step 1: Load and Explore Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset
df = pd.read_csv('market_pipe_thickness_loss_dataset.csv')
print(df.head())
print(df.info())

# Visualize numeric columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
sns.pairplot(df[numeric_columns])
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Step 2: Preprocessing
# Assume 'ThicknessLoss' is the target variable (change if needed)
X = df.drop(columns=['ThicknessLoss'])
y = df['ThicknessLoss']

# Handle categorical variables if any
X = pd.get_dummies(X, drop_first=True)

#model features
model_features = ['Pipe_Size_mm', 'Thickness_mm', 'Max_Pressure_psi', 'Temperature_C', 'Corrosion_Impact_Percent', 'Material_Loss_Percent', 'Time_Years', 'Material_Fiberglass', 'Material_HDPE', 'Material_PVC', 'Material_Stainless Steel', 'Grade_API 5L X52', 'Grade_API 5L X65', 'Grade_ASTM A106 Grade B', 'Grade_ASTM A333 Grade 6', 'Condition_Moderate', 'Condition_Normal']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluate
predictions = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

# Step 5: Save the Model
joblib.dump(model, '../backend/model.pkl')

'''

from flask import Flask, render_template, request

# Import your Random Forest classifier and other necessary libraries
import joblib
import numpy as np

# Load the trained Random Forest model
RF = joblib.load('trained_random_forest_model.pkl')

# Initialize Flask application
app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the form
    person_id = int(request.form['person_id'])
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    occupation = int(request.form['occupation'])
    sleep_duration = float(request.form['sleep_duration'])
    quality_of_sleep = int(request.form['quality_of_sleep'])
    physical_activity_level = int(request.form['physical_activity_level'])
    stress_level = int(request.form['stress_level'])
    bmi_category = int(request.form['bmi_category'])
    blood_pressure = int(request.form['blood_pressure'])
    heart_rate = int(request.form['heart_rate'])
    daily_steps = int(request.form['daily_steps'])

    # Prepare input data for prediction
    input_data = np.array([[person_id, gender, age, occupation, sleep_duration, quality_of_sleep,
                            physical_activity_level, stress_level, bmi_category,
                            blood_pressure, heart_rate, daily_steps]])

    # Use the trained Random Forest classifier to make prediction
    prediction = RF.predict(input_data)

    # Render the result template with the prediction
    return render_template('result.html', prediction=prediction[0])

# Run the application
if __name__ == '__main__':
    app.run(debug=True)

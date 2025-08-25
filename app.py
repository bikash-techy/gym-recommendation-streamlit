import pickle
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load saved models
scaler = pickle.load(open("Models/scaler.pkl", 'rb'))
model = pickle.load(open("Models/model.pkl", 'rb'))
input_label_encoders = pickle.load(open("Models/input_label_encoders.pkl", 'rb'))
output_label_encoders = pickle.load(open("Models/output_label_encoders.pkl", 'rb'))

# Load dataset to get unique values for dropdowns
df = pd.read_csv("gym_recommendation_with_health.csv")
unique_values = {
    'sex': df['Sex'].dropna().unique().tolist(),
    'hypertension': df['Hypertension'].dropna().unique().tolist(),
    'diabetes': df['Diabetes'].dropna().unique().tolist(),
    'level': df['Level'].dropna().unique().tolist(),
    'fitness_goal': df['Fitness Goal'].dropna().unique().tolist(),
    'fitness_type': df['Fitness Type'].dropna().unique().tolist(),
    'health_condition': df['HealthCondition'].dropna().unique().tolist()
}

def safe_encode(encoder, value, col_name):
    if value not in encoder.classes_:
        print(f"Warning: {value} not in {col_name} classes, using {encoder.classes_[0]}")
        return encoder.transform([encoder.classes_[0]])[0]
    return encoder.transform([value])[0]

def calculate_bmi(height_cm, weight_kg):
    return weight_kg / ((height_cm / 100) ** 2)

def validate_health_data(age, weight, height, entered_bmi):
    errors = []
    warnings = []
    calculated_bmi = calculate_bmi(height, weight)

    # Age validation
    if age < 1 or age > 150:
        errors.append("Please enter a valid age between 1-150 years")
    elif age > 120:
        warnings.append("Please verify the age entered")

    # Weight validation
    if weight < 5 or weight > 300:
        errors.append("Please enter a valid weight between 5-300 kg")
    elif weight > 200:
        warnings.append("Please verify the weight entered")

    # Height validation
    if height < 30 or height > 335:
        errors.append("Please enter a valid height between 30-335 cm")
    elif height > 250:
        warnings.append("Please verify the height entered")
    elif age >= 18 and height < 100:
        warnings.append("Height seems unusually low for an adult")

    # BMI comparison
    if abs(calculated_bmi - entered_bmi) > 0.5:
        errors.append(f"The BMI you entered doesn't match your height and weight. Calculated BMI: {calculated_bmi:.1f}, You entered: {entered_bmi}")

    # Height-based body type expectations
    if height >= 213:  # 7+ feet
        if weight < 70 or weight > 120:
            warnings.append("Your weight seems low or high for your height (7+ feet). Expected range: 70-120 kg")
        if calculated_bmi < 18.5:
            errors.append("BMI too low for your height - this may indicate health concerns")
    elif height >= 183:  # 6-7 feet
        if weight < 55 or weight > 100:
            warnings.append("Weight may be too low or high for optimal health. Expected range: 60-100 kg")
        if calculated_bmi < 17:
            errors.append("BMI critically low for your height")
    elif height >= 152:  # 5-6 feet
        if calculated_bmi < 16:
            errors.append("Severely underweight - please consult a doctor")
        elif calculated_bmi > 40:
            errors.append("Severely obese - please consult a doctor")

    # Age-height relationship
    if age < 18 and height > 200:
        warnings.append("Unusual height for age - please verify")
    elif age > 60 and height > 200:  # Assuming height increase is unlikely
        warnings.append("Height typically doesn't increase after age 30")

    # Extreme combinations
    if age > 100 and weight > 150:
        warnings.append("Please verify these values")
    if height > 250 and weight < 80:
        errors.append("This combination seems medically unlikely")
    if age < 16 and weight > 120:
        warnings.append("Please consult healthcare provider")

    # BMI category warnings
    if calculated_bmi < 16:
        warnings.append("Severely underweight - immediate medical attention recommended")
    elif 16 <= calculated_bmi < 17:
        warnings.append("Underweight - consider nutritional counseling")
    elif calculated_bmi > 35:
        warnings.append("Obesity Class II - medical supervision recommended")
    elif calculated_bmi > 40:
        warnings.append("Obesity Class III - immediate medical consultation required")

    return errors, warnings, calculated_bmi

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            sex = request.form['sex']
            age = float(request.form['age'])
            height = float(request.form['height'])
            weight = float(request.form['weight'])
            hypertension = request.form['hypertension']
            diabetes = request.form['diabetes']
            bmi = float(request.form['bmi'])
            level = request.form['level']
            fitness_goal = request.form['fitness_goal']
            fitness_type = request.form['fitness_type']
            waist_circumference = float(request.form['waist_circumference'])
            health_condition = request.form['health_condition']

            errors, warnings, calculated_bmi = validate_health_data(age, weight, height, bmi)

            if errors:
                return render_template('index.html', errors=errors, warnings=warnings, unique_values=unique_values)
            else:
                features = np.array([[safe_encode(input_label_encoders['Sex'], sex, 'Sex'),
                                    age, height, weight,
                                    safe_encode(input_label_encoders['Hypertension'], hypertension, 'Hypertension'),
                                    safe_encode(input_label_encoders['Diabetes'], diabetes, 'Diabetes'),
                                    bmi, safe_encode(input_label_encoders['Level'], level, 'Level'),
                                    safe_encode(input_label_encoders['Fitness Goal'], fitness_goal, 'Fitness Goal'),
                                    safe_encode(input_label_encoders['Fitness Type'], fitness_type, 'Fitness Type'),
                                    waist_circumference,
                                    safe_encode(input_label_encoders['HealthCondition'], health_condition, 'HealthCondition')]])
                print("Features shape:", features.shape)

                scaled_features = scaler.transform(features)
                print("Scaled features shape:", scaled_features.shape)

                encoded_preds = model.predict(scaled_features)[0]
                print("Encoded predictions:", encoded_preds)

                output_cols = ['Exercises', 'Equipment', 'Diet', 'Recommendation']
                result = {col: output_label_encoders[col].inverse_transform([encoded_preds[i]])[0] for i, col in enumerate(output_cols)}
                print("Result:", result)

                return render_template('index.html', result=result, warnings=warnings, unique_values=unique_values)
        except Exception as e:
            return render_template('index.html', errors=[f"Error: {str(e)}"], unique_values=unique_values)

    return render_template('index.html', unique_values=unique_values)

if __name__ == '__main__':
    app.run(debug=True)
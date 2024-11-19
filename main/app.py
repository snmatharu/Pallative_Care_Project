from flask import Flask, request, render_template
import joblib
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

# Load the saved model, scaler, and encoder mappings
saved_data = joblib.load('worker_prediction_model.pkl')
model = saved_data['model']
scaler_params = saved_data['scaler']
encodings = saved_data['encoder']  # Added encoder mappings
print(encodings)
# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form inputs
        number_male = float(request.form.get('Number_male', 0))
        number_female = float(request.form.get('Number_female', 0))
        stats_canada_pop = float(request.form.get('Statistics_Canada_population', 0))
        physician_ratio = float(request.form.get('Physician_to_100000_population_ratio', 0))
        volume = float(request.form.get('Volume', 0))
        province = request.form.get('Province/territory', 'Unknown')
        category = request.form.get('CategorySupplyKey', 'Unknown')

        # Handle encoded inputs
        print(province)
        province_encoded = encodings['Province/territory'].get(province, -1)
        print(province_encoded)
        category_encoded = encodings['CategorySupplyKey'].get(category, -1)

        if province_encoded == -1 or category_encoded == -1:
            raise ValueError("Invalid Province/territory or CategorySupplyKey input")

        # Scale the test case values based on the scaler_params
        scaled_test_case = {}
        for feature, params in scaler_params.items():
            if feature == 'Numbermale':
                value = number_male
            elif feature == 'Numberfemale':
                value = number_female
            elif feature == 'Statistics Canada population':
                value = stats_canada_pop
            elif feature == 'Physician-to 100,000 population ratio':
                value = physician_ratio
            elif feature == 'Volume':
                value = volume
            else:
                continue
            
            # Apply scaling formula: (value - mean) / scale
            scaled_value = (value - params['mean']) / params['scale']
            scaled_test_case[f'{feature}_scaled'] = scaled_value

        # Add encoded features to the test case
        scaled_test_case['Province/territory_encoded'] = province_encoded
        scaled_test_case['CategorySupplyKey_encoded'] = category_encoded

        # Convert to DataFrame for prediction
        X_test_case = pd.DataFrame([scaled_test_case])
        X_test_case = X_test_case[['Numbermale_scaled', 'Numberfemale_scaled', 
                                   'Statistics Canada population_scaled', 
                                   'Physician-to 100,000 population ratio_scaled', 
                                   'Volume_scaled', 
                                   'Province/territory_encoded', 
                                   'CategorySupplyKey_encoded']]

        # Perform the prediction using the loaded model
        prediction = model.predict(X_test_case)

        # Retrieve the mean and scale for rescaling the predicted value
        mean = scaler_params['Number of physicians']['mean']
        scale = scaler_params['Number of physicians']['scale']
        
        # Rescale the predicted value
        original_predicted_value = (prediction[0] * scale) + mean

        # Render result in the template
        return render_template('index.html', prediction=round(original_predicted_value, 2))

    except Exception as e:
        # Render error in the template
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)

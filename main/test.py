import joblib
import pandas as pd

def load_model_and_predict(test_case):
    # Load the saved model, scaler, and encoder mappings
    saved_data = joblib.load('worker_prediction_model.pkl')
    model = saved_data['model']
    scaler_params = saved_data['scaler']
    
    # Assuming scaler_params is loaded from the saved model

    # Scaled test case dictionary
    scaled_test_case = {}

    # Scale the test case values based on the scaler_params
    for feature, params in scaler_params.items():
        print(feature)
        if feature in test_case:
            value = test_case[feature]
            # Apply scaling formula: (value - mean) / scale
            scaled_value = (value - params['mean']) / params['scale']
            scaled_test_case[f'{feature}_scaled'] = scaled_value
        
    print("Scaled Test Case:", scaled_test_case)
    
    # Convert to DataFrame for prediction
    X_test_case = pd.DataFrame([scaled_test_case])
    X_test_case = X_test_case[['Numbermale_scaled', 'Numberfemale_scaled', 'Statistics Canada population_scaled', 'Physician-to 100,000 population ratio_scaled', 'Volume_scaled']]
    print("Test Case DataFrame:", X_test_case)
    
    # Make the prediction
    prediction = model.predict(X_test_case)
    # Retrieve the mean and scale for this feature
    mean = scaler_params['Number of physicians']['mean']
    scale = scaler_params['Number of physicians']['scale']

    # Rescale the predicted value
    original_predicted_value = (prediction[0] * scale) + mean
    print(f"Rescaled Predicted Value: {original_predicted_value:.4f}")

    # return prediction[0]

# Example usage
test_case = {
    'Numbermale': 60,
    'Numberfemale': 40,
    'Statistics Canada population': 100000,
    'Physician-to 100,000 population ratio': 150.5,
    'Volume': 5000
}

load_model_and_predict(test_case)

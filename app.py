import pandas as pd
import joblib
from flask import Flask, request, jsonify
import os

# Initialize the Flask application
app = Flask(__name__)

# Load the trained logistic regression model
try:
    model = joblib.load('logistic_regression_model.joblib')
    print("Flask app initialized and model loaded.")
except FileNotFoundError:
    print("Error: 'logistic_regression_model.joblib' not found. Make sure the model file is in the same directory as app.py or provide the correct path.")
    exit() # Exit if model cannot be loaded

# Load preprocessing artifacts
try:
    preprocessing_artifacts = joblib.load('preprocessing_artifacts.joblib')
    model_features = preprocessing_artifacts['model_features']
    mode_date_added = preprocessing_artifacts['mode_date_added']
    mode_rating = preprocessing_artifacts['mode_rating']
    input_categorical_cols = preprocessing_artifacts['input_categorical_cols']
    print("Preprocessing artifacts loaded.")
except FileNotFoundError:
    print("Error: 'preprocessing_artifacts.joblib' not found. Ensure artifacts are saved from the training notebook.")
    exit() # Exit if preprocessing artifacts cannot be loaded

# --- Preprocessing Function ---
def parse_duration(duration):
    """Parses duration string to numeric minutes."""
    if 'min' in str(duration):
        return int(str(duration).replace(' min', ''))
    elif 'Season' in str(duration):
        return int(str(duration).replace(' Seasons', '').replace(' Season', '')) * 60 
    return 0 

def preprocess_input(data_row):
    """Preprocesses a single input data row for model prediction."""
    input_df = pd.DataFrame([data_row])

    for col in ['director', 'cast', 'country']:
        if col not in input_df.columns: 
            input_df[col] = 'Unknown'
        input_df[col].fillna('Unknown', inplace=True)
    
    if 'rating' not in input_df.columns:
        input_df['rating'] = mode_rating # Use loaded mode
    input_df['rating'].fillna(mode_rating, inplace=True)

    if 'date_added' not in input_df.columns:
        input_df['date_added'] = mode_date_added # Use loaded mode
    input_df['date_added'] = pd.to_datetime(input_df['date_added'], errors='coerce', format='mixed')
    input_df['date_added'].fillna(mode_date_added, inplace=True) # Fill NaT with loaded string mode, then will be re-parsed as datetime if needed

    if 'description' in input_df.columns:
        input_df.drop('description', axis=1, inplace=True)
    
    input_df['month_added'] = input_df['date_added'].dt.month
    input_df['year_added'] = input_df['date_added'].dt.year
    input_df.drop('date_added', axis=1, inplace=True)

    input_df['duration_numeric'] = input_df['duration'].apply(parse_duration)
    input_df.drop('duration', axis=1, inplace=True)

    cols_to_encode = [col for col in input_categorical_cols if col in input_df.columns] # Use loaded list
    encoded_input_df = pd.get_dummies(input_df, columns=cols_to_encode, drop_first=True)
    
    processed_data = encoded_input_df.reindex(columns=model_features, fill_value=0) # Use loaded model features
    
    return processed_data

# --- API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict_api():
    """Predicts the type of content (Movie/TV Show) based on input data."""
    try:
        data = request.get_json(force=True)
        
        if isinstance(data, list):
            input_data = data[0]
        else:
            input_data = data
            
        processed_data = preprocess_input(input_data)
        
        prediction = model.predict(processed_data)
        prediction_proba = model.predict_proba(processed_data)

        result_label = 'TV Show' if prediction[0] == 1 else 'Movie'
        
        response = {
            'prediction': result_label,
            'confidence_Movie': prediction_proba[0][0],
            'confidence_TV_Show': prediction_proba[0][1]
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Run the Flask app ---
if __name__ == '__main__':
    # In Colab, you might need to use a tool like ngrok for external access.
    # For local testing within Colab, '0.0.0.0' works.
    app.run(host='0.0.0.0', port=5000)

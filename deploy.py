import os
import time
import threading
import pandas as pd
import numpy as np
import joblib
from pyflowmeter.sniffer import create_sniffer

# Load the saved components
scaler = joblib.load('minmax_scaler.pkl')
rf_binary = joblib.load('rf_binary_model.pkl')
rf_multi = joblib.load('rf_multi_model.pkl')
le = joblib.load('label_encoder.pkl')
feature_columns = joblib.load('feature_columns.pkl')
to_drop = joblib.load('dropped_correlated_columns.pkl')

def preprocess_new_data(new_df):
    # Handle missing (hyphens/spaces to NaN)
    new_df.replace(["-", " "], np.nan, inplace=True)
    
    # Drop highly correlated features using saved list
    new_df.drop(columns=to_drop, inplace=True, errors='ignore')
    
    # Select only the feature columns used in training
    new_df = new_df[feature_columns]
    
    # Normalize
    new_scaled = pd.DataFrame(scaler.transform(new_df), columns=new_df.columns)
    
    return new_scaled

def predict(X_new):
    # Binary prediction
    pred_binary = rf_binary.predict(X_new)
    
    # Initialize predictions as 'Normal'
    predictions = np.array(['Normal'] * len(pred_binary), dtype=object)
    
    # For predicted attacks, run multi-class
    mask_attack = (pred_binary == 1)
    if np.any(mask_attack):
        X_attack = X_new[mask_attack]
        pred_multi = rf_multi.predict(X_attack)
        predictions[mask_attack] = le.inverse_transform(pred_multi)
    
    return predictions

# CSV output file
csv_file = 'flows.csv'

# Remove existing CSV if present
if os.path.exists(csv_file):
    os.remove(csv_file)

# Create sniffer for live capture (change 'eth0' to your interface if needed; None for all interfaces)
sniffer = create_sniffer(
    input_interface=None,  # Or 'eth0', 'wlan0', etc.
    to_csv=True,
    output_file=csv_file
)

# Start sniffer in a separate thread
def run_sniffer():
    sniffer.start()
    sniffer.join()

sniffer_thread = threading.Thread(target=run_sniffer)
sniffer_thread.start()

print("Starting live network capture... Press Ctrl+C to stop.")

last_len = 0
try:
    while True:
        time.sleep(10)  # Check every 10 seconds
        if os.path.exists(csv_file):
            try:
                df_new = pd.read_csv(csv_file)
                if len(df_new) > last_len:
                    new_rows = df_new.iloc[last_len:]
                    # Rename columns if needed (assuming pyflowmeter outputs match CICFlowMeter columns; adjust if necessary)
                    # e.g., df_new.rename(columns={"some_col": "Label"}, inplace=True) if mismatches
                    preprocessed = preprocess_new_data(new_rows)
                    preds = predict(preprocessed.values)  # Convert to numpy if needed
                    for i, pred in enumerate(preds):
                        print(f"Prediction for flow {last_len + i + 1}: {pred}")
                    last_len = len(df_new)
            except Exception as e:
                print(f"Error reading/processing CSV: {e}")
except KeyboardInterrupt:
    print("Stopping capture...")
finally:
    sniffer.stop()
    sniffer_thread.join()
    print("Capture stopped. Final predictions processed.")

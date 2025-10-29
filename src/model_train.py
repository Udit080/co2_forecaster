# src/model_train.py
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.data_prep import load_and_prep_data

MODEL_PATH = "sarima_model.joblib"
DATA_FILE = "monthly_flask_co2_mlo.csv"

def train_and_save_model():
    """Trains the SARIMA model and saves it to disk."""
    
    # 1. Load Prepared Data
    ts, _ = load_and_prep_data(DATA_FILE)
    
    # 2. Define and Train SARIMA Model
    # SARIMA(0, 1, 1)x(0, 1, 1, 12) is a common, robust starting point for CO2 data
    print("Training SARIMA model...")
    model = SARIMAX(
        ts,
        order=(0, 1, 1),      # Non-seasonal order (p, d, q)
        seasonal_order=(0, 1, 1, 12), # Seasonal order (P, D, Q, S=12 for monthly)
        enforce_stationarity=False,
        enforce_invertibility=False,
        initialization='approximate_diffuse' # Required for older statsmodels versions
    )
    
    model_fit = model.fit(disp=False) # disp=False suppresses full output
    
    # 3. Save Model
    joblib.dump(model_fit, MODEL_PATH)
    print(f"SARIMA model successfully trained and saved to {MODEL_PATH}")

if __name__ == '__main__':
    # Set the path one level up for the main file
    # Note: In production, the data file should be stored securely or accessed via a remote storage.
    train_and_save_model()
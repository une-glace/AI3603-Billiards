
import argparse
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_model(data_path, output_path):
    # Load data
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    X = data['X']
    y = data['y']
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)
    
    # Model
    # Use HistGradientBoostingRegressor for better non-linear capture
    model = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.1,
        max_depth=10,
        random_state=42
    )
    
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = model.score(X_test, y_test)
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R2: {r2:.4f}")
    
    # Save
    checkpoint = {
        'scaler': scaler,
        'model': model
    }
    joblib.dump(checkpoint, output_path)
    print(f"Model saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='train/dataset_final.npz')
    parser.add_argument('--out', type=str, default='eval/checkpoints/cqr.joblib')
    args = parser.parse_args()
    
    train_model(args.data, args.out)

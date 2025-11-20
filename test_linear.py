import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from quant_tcn_riskmap.data_loader import download_data
from quant_tcn_riskmap.features import FeatureEngineer

def main():
    print("Testing Linear Model...")
    df = download_data("SPY", "2005-01-01", "2024-01-01")
    fe = FeatureEngineer(window_size=64)
    features_df, _ = fe.create_features(df)
    
    # Create simple dataset (no window, just current features)
    # Target is next return > 0
    targets = (features_df['log_ret'].shift(-1) > 0).astype(int)
    
    # Align
    valid_indices = features_df.index[:-1]
    X = features_df.loc[valid_indices].values
    y = targets.loc[valid_indices].values
    timestamps = features_df.index[:-1]
    
    # Train/Test Split (2018)
    test_mask = (timestamps.year >= 2018)
    train_mask = (timestamps.year < 2018) & (timestamps.year >= 2010)
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    # Normalize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    # Logistic Regression
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    
    print(f"Train Acc: {train_acc:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    
    if test_acc > 0.51:
        print("Signal detected!")
    else:
        print("No signal detected in linear features.")

if __name__ == "__main__":
    main()

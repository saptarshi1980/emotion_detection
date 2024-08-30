import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier
import time
from sklearn.ensemble import GradientBoostingClassifier
import yaml
from logger import logging
from exception import CustomException
import os,sys
import yaml
import time


def load_model(file_path: str):
    """Loads a trained model from a pickle file."""
    
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
            return model
    except Exception as e:
        raise CustomException(e,sys)


def load_test_data(file_path: str) -> pd.DataFrame:
    """Loads the test data from a CSV file."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise CustomException(e,sys)


def split_features_and_labels(df: pd.DataFrame):
    """Splits the DataFrame into features (X) and labels (y)."""
    try:
        X = df.iloc[:, :-1].values  # All columns except the last one
        y = df.iloc[:, -1].values   # The last column
        return X, y
    except Exception as e:
        raise CustomException(e,sys)


def evaluate_model(model, X: np.ndarray, y: np.ndarray) -> dict:
    """Evaluates the model using various metrics and returns the results."""
    try:
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]


    # Calculate evaluation metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        auc = roc_auc_score(y, y_pred_proba)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
    except Exception as e:
        raise CustomException(e,sys)

def save_metrics(metrics: dict, file_path: str):
    """Saves the evaluation metrics to a JSON file."""
    try:
        logging.info("Loggin file path-",file_path)
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
    except Exception as e:
        raise CustomException(e,sys)
    

def main():
    # Define file paths
    timestamp = time.strftime("%Y%m%d-%H%M")
    model_file_path = './models/model.pkl'
    test_file_path = './data/features/test_bow.csv'
    metrics_file_path = 'metrics.json'
    metrics_history_file_path = f"./models/history/metrics_{timestamp}.json"

    # Load the trained model
    model = load_model(model_file_path)

    # Load the test data
    test_data = load_test_data(test_file_path)

    # Split test data into features and labels
    X_test, y_test = split_features_and_labels(test_data)

    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test)

    # Save the evaluation metrics
    save_metrics(metrics, metrics_file_path)
    save_metrics(metrics, metrics_history_file_path)

if __name__ == "__main__":
    main()

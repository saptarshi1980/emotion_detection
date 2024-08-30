import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
import yaml
from logger import logging
from exception import CustomException
import os,sys
import yaml
import time



def load_feature_data(file_path: str) -> pd.DataFrame:
    """Loads the feature-engineered data from a CSV file."""
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

    

def train_model(X: np.ndarray, y: np.ndarray):
    """Trains a model on the provided features and labels."""
    #model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    try:
        params = yaml.safe_load(open('params.yaml','r'))['model_building']
        n_estimators = params['n_estimators']
        learning_rate = params['learning_rate']
        model = GradientBoostingClassifier(n_estimators=n_estimators,learning_rate=learning_rate)
        model.fit(X, y)
        return model
    except Exception as e:
        raise CustomException(e,sys)


def save_model(model, file_path: str):
    """Saves the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)           
    except Exception as e:
        raise CustomException(e,sys)


def main():
    # Define file paths
    timestamp = time.strftime("%Y%m%d-%H%M")
    #output_file = f"models/model_{timestamp}.pkl"
    train_file_path = './data/features/train_tfidf.csv'
    model_file_path = './models/model.pkl'
    model_history_file_path = f"./models/history/model_{timestamp}.pkl"
    
    # Load the training data
    train_data = load_feature_data(train_file_path)

    # Split data into features and labels
    X_train, y_train = split_features_and_labels(train_data)

    # Train the XGBoost model
    model = train_model(X_train, y_train)

    # Save the trained model
    save_model(model, model_file_path)
    save_model(model, model_history_file_path)

if __name__ == "__main__":
    main()

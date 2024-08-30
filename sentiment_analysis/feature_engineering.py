import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml
import yaml
from logger import logging
from exception import CustomException
import os,sys
import yaml
import time


def load_processed_data(train_file_path: str, test_file_path: str):
    """Loads the processed training and test data from CSV files."""
    try:
        train_data = pd.read_csv(train_file_path)
        test_data = pd.read_csv(test_file_path)
        return train_data, test_data
    except Exception as e:
        raise CustomException(e,sys)

def handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing data with empty strings."""
    try:
        return df.fillna('')
    except Exception as e:
        raise CustomException(e,sys)

def extract_features(X_train: pd.Series, X_test: pd.Series, max_features: int):
    """Applies Bag of Words (CountVectorizer) to the training and test data."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        return X_train_tfidf, X_test_tfidf
    except Exception as e:
        raise CustomException(e,sys)

def create_feature_dataframe(X_tfidf, y) -> pd.DataFrame:
    """Creates a DataFrame from the Bag of Words features and labels."""
    try:
        df = pd.DataFrame(X_tfidf.toarray())
        df['label'] = y
        return df
    except Exception as e:
        raise CustomException(e,sys)

def save_feature_data(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str):
    """Saves the feature-engineered training and test data to CSV files."""
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(output_dir, "train_tfidf.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test_tfidf.csv"), index=False)
    except Exception as e:
        raise CustomException(e,sys)    

def main():
    # Define file paths
    logging.info("Performing- feature Engineering")
    max_features = yaml.safe_load(open('params.yaml','r'))['feature_engineering']['max_features']
    train_file_path = './data/processed/train_processed.csv'
    test_file_path = './data/processed/test_processed.csv'
    output_dir = os.path.join("data", "features")

    # Load data
    train_data, test_data = load_processed_data(train_file_path, test_file_path)

    # Handle missing data
    train_data = handle_missing_data(train_data)
    test_data = handle_missing_data(test_data)

    # Extract features and labels
    X_train = train_data['content'].values
    y_train = train_data['sentiment'].values
    X_test = test_data['content'].values
    y_test = test_data['sentiment'].values

    # Apply Bag of Words
    X_train_tfidf, X_test_tfidf = extract_features(X_train, X_test,max_features=max_features)

    # Create DataFrames for features
    train_df = create_feature_dataframe(X_train_tfidf, y_train)
    test_df = create_feature_dataframe(X_test_tfidf, y_test)

    # Save the processed feature data
    save_feature_data(train_df, test_df, output_dir)

if __name__ == "__main__":
    main()

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from logger import logging
from exception import CustomException
import os,sys
import yaml


def load_data(file_path: str) -> pd.DataFrame:
    """Loads the CSV data into a DataFrame."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise CustomException(e,sys)
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the DataFrame by dropping unnecessary columns and filtering rows."""
    
    try:
        df = df.drop(columns=['tweet_id'])
        df = df[df['sentiment'].isin(['neutral', 'sadness'])]
        df = df.copy()  # Ensure we work on a copy to avoid SettingWithCopyWarning
        df['sentiment'] = df['sentiment'].replace({"neutral": 0, "sadness": 1})
        return df
    except Exception as e:
        raise CustomException(e,sys)
    

def split_data(df: pd.DataFrame, test_size:float, random_state: int = 42):
    """Splits the data into training and test sets."""
    try:
        return train_test_split(df, test_size=test_size, random_state=random_state)
    except Exception as e:
        raise CustomException(e,sys)

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str):
    """Saves the training and test data to CSV files."""
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
    except Exception as e:
        raise CustomException(e,sys)    

def main():
    # Define paths and parameters
    test_size = yaml.safe_load(open('params.yaml','r'))['data_ingestion']['test_size']
    input_file_path = './data/tweet_emotions.csv'
    data_path = os.path.join("data", "raw")
    
    # Load data
    df = load_data(input_file_path)
    
    # Preprocess data
    final_df = preprocess_data(df)
    
    # Split data
    train_data, test_data = split_data(final_df,test_size=test_size)
    
    # Save data
    save_data(train_data, test_data, data_path)

if __name__ == "__main__":
    
    main()

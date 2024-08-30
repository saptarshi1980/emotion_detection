import os
import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re
from logger import logging
from exception import CustomException
import os,sys
import yaml



def load_data(train_file_path: str, test_file_path: str):
    """Loads the training and test data from CSV files."""
    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)
    return train_data, test_data

def download_nltk_resources():
    """Downloads necessary NLTK resources."""
    nltk.download('wordnet')
    nltk.download('stopwords')

def lemmatization(text: str) -> str:
    """Lemmatizes the input text."""
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text: str) -> str:
    """Removes stop words from the input text."""
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])

def remove_numbers(text: str) -> str:
    """Removes numbers from the input text."""
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text: str) -> str:
    """Converts the input text to lower case."""
    return " ".join([word.lower() for word in text.split()])

def remove_punctuations(text: str) -> str:
    """Removes punctuation from the input text."""
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text.strip()

def remove_urls(text: str) -> str:
    """Removes URLs from the input text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df: pd.DataFrame, min_length: int = 3):
    """Removes sentences from the DataFrame that are smaller than a specified length."""
    df['text'] = df['text'].apply(lambda x: np.nan if len(str(x).split()) < min_length else x)

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes the text by applying various preprocessing steps."""
    df['content'] = df['content'].apply(lower_case)
    df['content'] = df['content'].apply(remove_stop_words)
    df['content'] = df['content'].apply(remove_numbers)
    df['content'] = df['content'].apply(remove_punctuations)
    df['content'] = df['content'].apply(remove_urls)
    df['content'] = df['content'].apply(lemmatization)
    return df

def save_processed_data(train_data: pd.DataFrame, test_data: pd.DataFrame, output_dir: str):
    """Saves the processed training and test data to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    train_data.to_csv(os.path.join(output_dir, "train_processed.csv"), index=False)
    test_data.to_csv(os.path.join(output_dir, "test_processed.csv"), index=False)

def main():
    # Define file paths
    logging.info("Performing- DATA PRE-PROCESSING")
    train_file_path = './data/raw/train.csv'
    test_file_path = './data/raw/test.csv'
    output_dir = os.path.join("data", "processed")

    # Load data
    train_data, test_data = load_data(train_file_path, test_file_path)

    # Download NLTK resources
    download_nltk_resources()

    # Normalize the text data
    train_processed_data = normalize_text(train_data)
    test_processed_data = normalize_text(test_data)

    # Save the processed data
    save_processed_data(train_processed_data, test_processed_data, output_dir)

if __name__ == "__main__":
    main()

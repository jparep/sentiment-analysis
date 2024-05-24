import pandas as pd
import re
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearnex import patch_sklearn
patch_sklearn()

def initialize_resource():
    """
    Initializes and downloads necessary NLTK resources and patches scikit-learn with Intel optimizations.
    """
    import nltk
    nltk.download("stopwords", quiet=True) # Quiet mode suppresses the console output

def load_data(filepath):
    """
    Loads the dataset from a CSV file, ensuring all text data are treated as strings.
    Fills missing values with a placeholder string "missing".

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded DataFrame with preprocessed 'review' column.
    """
    df = pd.read_csv(filepath)
    df["review"] = df["review"].astype(str)
    df["review"] = df["review"].fillna("missing", inplace=True)
    return df


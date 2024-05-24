#Importing all the neccessary libraries
import pandas as pd
import re
from bs4 import BeautifulSoup
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords') # Download the stopwords dataset
from nltk.corpus import stopwords
from sklearnex import patch_sklearn 
patch_sklearn()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

df = pd.read_csv('IMDB_dataset.csv')

def preprocess_data(text):
    text = BeautifulSoup(text, "html.parser")
    text = re.sub(r"[^a-zA-Z]+", " ", text)
    text = text.lower()
    text = [lemmatizer.lemmatize(word) for word in text.split(" ")] # Reduce words to their conical and basic form
    text = [lemmatizer.lemmatize(word, "v") for word in text]
    text = [word for word in text if not word in stop_words]
    result = " ".join(text)
    return result

df["preprocessed_review"] = df["review"].apply(lambda text: preprocess_data(text))
df['sentiment'] = df["sentiment"].map({"negative": 0, "positive": 1})

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

lem = WordNetLemmatizer()
stopWords = set(stopwords.words("english"))

data = pd.read_csv('IMDB_dataset.csv')
print(data.head(5))


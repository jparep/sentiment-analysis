# Importing all the necessary libraries
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
nltk.download('stopwords')  # Download the stopwords dataset
from nltk.corpus import stopwords
from sklearnex import patch_sklearn 
patch_sklearn()

# Initialize lemmatizer and stopwords set
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Load dataset
df = pd.read_csv('IMDB_dataset.csv')

# Ensure reviews are treated as strings and handle missing values
df['review'] = df['review'].astype(str)
df['review'].fillna("missing", inplace=True)

def preprocess_data(text):
    try:
        # Parse the text as HTML and get clean text
        text = BeautifulSoup(text, "html.parser").get_text()
        # Remove non-alphabet characters and extra spaces
        text = re.sub(r"[^a-zA-Z]+", " ", text)
        text = text.lower()  # Convert to lowercase
        # Lemmatize words (nouns and verbs)
        text = [lemmatizer.lemmatize(word) for word in text.split()]
        text = [lemmatizer.lemmatize(word, "v") for word in text]
        # Remove stopwords
        text = [word for word in text if word not in stop_words]
        result = " ".join(text)
        return result
    except Exception as e:
        print(f"Error processing text: {e}")
        return ""

# Apply preprocessing to the review data
df["preprocessed_review"] = df["review"].apply(preprocess_data)

# Map sentiment to binary values
df['sentiment'] = df["sentiment"].map({"negative": 0, "positive": 1})

# Prepare features and target variables
feature = df['preprocessed_review']
target = df["sentiment"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=121)

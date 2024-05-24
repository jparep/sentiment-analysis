import pandas as pd
import re
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearnex import patch_sklearn
patch_sklearn()

def initialize_resources():
    """
    Initializes and downloads necessary NLTK resources and patches scikit-learn with Intel optimizations.
    """
    import nltk
    nltk.download("stopwords", quiet=True)  # Download stopwords
    nltk.download("wordnet", quiet=True)    # Download WordNet for lemmatization

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
    df["review"] = df["review"].fillna("missing")
    return df

def preprocess_data(text, lemmatizer, stop_words):
    """
    Cleans and preprocesses the text by removing HTML tags, non-alphabetical characters,
    converting to lowercase, lemmatizing, and removing stopwords.

    Args:
        text (str): The text to preprocess.
        lemmatizer (WordNetLemmatizer): An instance of WordNetLemmatizer.
        stop_words (set): A set of stopwords.

    Returns:
        str: The preprocessed text.
    """
    try:
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r"[^a-zA-Z]+", " ", text).lower()
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        processed_text = " ".join(words)
        return processed_text
    except Exception as e:
        print(f"Error processing text: {e}")
        return ""

def prepare_data(df, lemmatizer, stop_words):
    """
    Prepares data by applying text preprocessing and encoding the sentiment labels.

    Args:
        df (pandas.DataFrame): DataFrame containing the 'review' and 'sentiment' columns.
        lemmatizer (WordNetLemmatizer): An instance of WordNetLemmatizer.
        stop_words (set): A set of stopwords.

    Returns:
        Tuple: Features and target variables split into training and test sets.
    """
    df["processed_review"] = df["review"].apply(lambda x: preprocess_data(x, lemmatizer, stop_words))
    df['sentiment'] = df["sentiment"].map({"negative": 0, "positive": 1})
    X_train, X_test, y_train, y_test = train_test_split(df["processed_review"], df["sentiment"], test_size=0.2, random_state=121)
    return X_train, X_test, y_train, y_test

def vectorize_data(X_train, X_test):
    """
    Vectorizes text features using CountVectorizer.

    Args:
        X_train (iterable): Training feature set.
        X_test (iterable): Test feature set.

    Returns:
        Tuple: Transformed training and test feature sets.
    """
    vectorizer = CountVectorizer(min_df=1, max_df=0.95)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return vectorizer, X_train_vec, X_test_vec

def train_and_evaluate(X_train_vec, X_test_vec, y_train, y_test):
    """
    Trains an SVM classifier and evaluates it on the test set.

    Args:
        X_train_vec, X_test_vec: Vectorized training and test features.
        y_train, y_test: Training and test labels.

    Returns:
        str: The classification report.
    """
    clf = SVC()
    clf.fit(X_train_vec, y_train)
    predictions = clf.predict(X_test_vec)
    return clf, classification_report(y_test, predictions)

def predict_new_sample(vectorizer, model, new_sample):
    """
    Predicts the sentiment of new, unseen text samples.

    Args:
        vectorizer (CountVectorizer): The vectorizer used during training.
        model (SVC): The trained SVM classifier.
        new_sample (str): The new text sample to predict.

    Returns:
        int: Predicted sentiment label.
    """
    new_sample_vec = vectorizer.transform([new_sample])
    return model.predict(new_sample_vec)

# Main execution logic
if __name__ == "__main__":
    # Initialize and define parameters
    initialize_resources()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    # Load, prepare and vectorize data
    df = load_data("IMDB_dataset.csv")
    X_train, X_test, y_train, y_test = prepare_data(df, lemmatizer, stop_words)
    vectorizer, X_train_vec, X_test_vec = vectorize_data(X_train, X_test)
    
    # Train and evaluate data
    model, class_report = train_and_evaluate(X_train_vec, X_test_vec, y_train, y_test)
    print(class_report)

    # Example prediction
    new_sample = "This movie was an excellent portrayal of character development."
    predicted_sentiment = predict_new_sample(vectorizer, model, new_sample)
    print(f"Predicted sentiment for new sample: {'Positive' if predicted_sentiment[0] else 'Negative'}")

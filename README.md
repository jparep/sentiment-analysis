# SVM Text Classification on IMDB Movie Reviews

## Project Overview

This project applies a Support Vector Machine (SVM) model to classify text data from the IMDB movie reviews dataset. The SVM model is used to predict the accuracy of text classification and demonstrate substantial improvements over other methods. The dataset comprises 50,000 movie reviews, split into 40,000 training instances and 10,000 testing instances.

## Key Features

- **Text Processing**: Tokenization and lemmatization processes prepare the data by simplifying words to their basic forms and removing stopwords.
- **Vectorization**: Uses a count vectorizer to transform text data into numerical feature vectors.
- **Model Training**: The SVM model is trained on the processed data to ensure accurate sentiment prediction.
- **Testing**: New movie reviews are classified to test the model's accuracy, reflecting robust performance across different data subsets.

## Installation

To set up the project environment, follow these steps:

```bash
git clone https://github.com/yourusername/svm-text-classification-imdb.git
cd svm-text-classification-imdb
pip install -r requirements.txt

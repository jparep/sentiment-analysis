# SVM Text Classification on IMDB Movie Reviews

## Project Overview

This project applies a Support Vector Machine (SVM) model to classify text data from the IMDB movie reviews dataset. The SVM model is used to predict the accuracy of text classification and demonstrate substantial improvements over other methods. The dataset comprises 50,000 movie reviews, split into 40,000 training instances and 10,000 testing instances.

## Key Features

- **Text Processing**: Tokenization and lemmatization processes prepare the data by simplifying words to their basic forms and removing stopwords.
- **Vectorization**: Uses a count vectorizer to transform text data into numerical feature vectors.
- **Model Training**: The SVM model is trained on the processed data to ensure accurate sentiment prediction.
- **Testing**: New movie reviews are classified to test the model's accuracy, reflecting robust performance across different data subsets.

## Installation

To set up the project environment, follow these steps, To run the script, navigate to the script's directory and run:

```bash
git clone https://github.com/jparep/sentiment-analysis.git
cd sentiment-analysis
conda env create -f environment.yml
python sentiment_analysis.py
```

## Usage

1. **Initialization**: Download necessary NLTK resources and patches scikit-learn for performance optimization.
2. **Data Loading**: Load your dataset from a CSV file where texts and their corresponding sentiment labels are stored.
3. **Preprocessing**: Apply text preprocessing to clean and prepare the data.
4. **Vectorization**: Transform preprocessed text data into numerical vectors.
5. **Training**: Train the SVM model using the vectorized data.
6. **Evaluation**: Evaluate the model performance on a test dataset.
7. **Prediction**: Predict sentiment for new, unseen text samples.


## Project Structure

**sentiment_analysis.py**: Main Python script containing the workflow of loading data, preprocessing, training, evaluation, and prediction.


## Configuration

The project uses a configuration for the SVM and CountVectorizer which can be adjusted according to the specifics of the dataset and requirements of the user.

## Data

The dataset used should be in a CSV format with at least two columns: one for the review text and one for the sentiment label. Adjust the load_data function if your CSV format differs.
Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your enhancements.

## License

This project is open-sourced under the MIT License. See the LICENSE file for more details.

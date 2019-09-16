import sys
# import libraries
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import pickle

import warnings
warnings.filterwarnings("ignore")


def load_data(database_filepath):
    """
        Loads data from database
        Args:
            database_filepath: the database file path
        Returns:
            X: features dataframe
            Y: labels dataframe
            category_names: y label categories
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse', con=engine)
    X, Y = df['message'], df.iloc[:, 4:]
    # Convert multiple category label to binary
    Y['related'] = Y['related'].map(lambda x: 1 if x == 2 else x)
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    """
        Tokenizes text data

        Args:
        text: text string
        Returns:
        An array of text after normalizing, tokenizing and lemmatizing
    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize text
    tokenized = word_tokenize(text)

    #Stop words removal
    sw_removed = [w for w in tokenized if w not in stopwords.words("english")]

    # Returns the lemmatized text
    return [WordNetLemmatizer().lemmatize(w, pos='v') for w in sw_removed]


def build_model():
    """
    The function to build with defined pipeline trained with GridSearchCV on a range of parameters.
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(OneVsRestClassifier(RandomForestClassifier())))])

    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                  'vect__max_features': (500, 1000, 5000),
                  'vect__max_df': (0.5, 0.75, 1.0)
                  }
    model = GridSearchCV(pipeline, param_grid=parameters)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
        Evaluate the model on a test dataset
        Args:
            model: Trained model
            X_test: Features test data
            Y_test: Labels test data
            category_names: category names in array
    """
    # predict
    y_pred = model.predict(X_test)
    print("\n")
    # print classification report
    print("################################ Classification report ##################################\n")
    print(classification_report(Y_test.values, y_pred, target_names=category_names))

    print("################## Classification accuracy scores for each category #####################\n")
    for i in range(len(Y_test.columns)):
        print("Accuracy score for {}: {} " .format(Y_test.columns[i], round(accuracy_score(Y_test.values[:, i], y_pred[:, i]),3)))
    print("\n")

def save_model(model, model_filepath):
    """
        Save the model to a Python pickle
        Args:
            model: Trained model
            model_filepath: File path where model is saved
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

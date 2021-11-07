import sys

import pandas as pd
from sqlalchemy import create_engine

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

import pickle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    """Data from database is loaded.

    Parameters:
        database_filepath (str): database file path

    Returns:
        Columns with X and Y, and category names
    """
    # build database path string
    database_filepath_str = 'sqlite:///'+database_filepath

    # create SQLite engine
    engine = create_engine(database_filepath_str)

    # build table name
    last_occ = database_filepath_str.rfind('/')+1
    table_name = database_filepath_str[last_occ:-3]
    print('     Table: ',table_name)

    # import SQLite engine to dataframe
    df = pd.read_sql_table(table_name, engine)
    print('     Dataframe Dimensions:',df.shape)

    # define feature and target variables X and Y
    columns_x = df.columns[1]
    columns_y = df.columns[4:]

    X = df[columns_x].values
    Y = df[columns_y].values

    # return values
    return X, Y, df[columns_y].columns


def tokenize(text):
    """Messages text is tokenized.

    Parameters:
        text: text from messages

    Returns:
        tokens after text is tokenized
    """

    # init tokenize elements
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    # normalize case and remove punctuation
    text =  re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]

    return tokens



def build_model():
    """Machine Learning model is built using Pipeline and GridSearchCV

    Parameters:
        none

    Returns:
        GridSearchCV object model
    """

    # create machine learning pipeline
    pipeline = Pipeline([
                        ('vect',CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
                        ])

    parameters = {
            'tfidf__use_idf': (True, False),
            'vect__max_df': (0.5, 1.0),
            'clf__estimator__n_neighbors': [5]
            }
    cv = GridSearchCV(pipeline, param_grid=parameters,verbose=3)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """The ML model is evaluated by comparing Y_pred and Y_test

    Parameters:
        model: ML model
        X_test: columns X from testing data
        Y_test: columns Y from testing data
        category_names: names of the categories

    Returns:
        none
    """
    # predict on test data
    Y_pred = model.predict(X_test)

    # show classification report
    for i in range(Y_test.shape[1]):
        print('feature number: ',i+1)
        print('category name:',category_names[i])
        print(classification_report(Y_test[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    """Machine Learning model is saved in a pickle file

    Parameters:
        model: Machine Learning model
        model_filepath (str): path where pickle file is saved

    Returns:
        none
    """
    # export model as a pickle file
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


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

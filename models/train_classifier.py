import sys

import numpy as np

import pandas as pd

from sqlalchemy import create_engine

import re

import pickle

import nltk

from nltk.tokenize import word_tokenize,sent_tokenize

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from sklearn.multioutput import MultiOutputClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.model_selection  import GridSearchCV

nltk.download(['punkt','stopwords','wordnet'])


def load_data(database_filepath):
    
    """ Takes input database file path and creates sqlite engine and returns data frames used for our model and category names """
    
    engine = create_engine('sqlite:///'+ database_filepath)

    df = pd.read_sql_table('messages', engine)
    
    X = df.message
    
    Y = df.iloc[:,4:]
    
    category_names = list(df.columns[4:])
    
    return X, Y, category_names


def tokenize(text):
    
    """ Cleans the data files which is given as text to numerical for us to perform machine learning classification """
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    tokens = word_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    
    for tok in tokens:
        
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """ Machine learning pipeline is created """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
    'vect__min_df': [1, 5],
    'tfidf__use_idf':[True, False],
    'clf__estimator__n_estimators':[10, 20]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """ takes model,data and category names as inputs and evaluates it and generates classification report """
    y_pred = model.predict(X_test)
    
    pred_data = pd.DataFrame(y_pred, columns = category_names)
    
    for column in category_names:
        print('_ '* 50 )
        
        print('\n')
    
        print('column: {}\n'.format(column))
        
        print(classification_report(Y_test[column],pred_data[column]))


def save_model(model, model_filepath):
    """ dumps model as pickle so it can be used later """
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """ Performs and shows how model is performing and also error message when encountered with error """
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

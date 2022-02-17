### Downloading necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords

### Importing libraries
import sys
import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

### Importing packages for the Machine Learning Model
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

### Importing packages to save the model
import joblib

### Declaring strings to be detected and the list of stop words
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
stop_words = nltk.corpus.stopwords.words("english")

### Declaring the error message
error_message = '''
Please, provide the filepaths of the messages and categories datasets as the first and second arguments,
respectively, as well as the path of the database to save the cleansed data as the third argument.
Example: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
'''


###############################################################################################################

### Loading the data from the SQL server
def load_data(db_path):
    """
    Loads the data used in the model - messages.csv and categories.csv
    Both files are located in the home directory
    
    INPUT:  None
    
    OUTPUT: X  - Numpy array originated from the dataframe saved in the SQL server. It contains the predictor.
            y  - Numpy array originated from the dataframe saved in the SQL server. It contains the labels.
            df - The dataframe saved in the SQL server from process_data.py  
    """
    
    ### Declaring the query and the engine to connect to the database and pull the data from the previous step
    print('Loading data')
    querystring = """SELECT * from disaster_response_mod"""
    engine = create_engine('sqlite:///{}'.format(db_path))
    
    ### Connecting to the database and querying the data
    df = pd.read_sql(querystring, engine)

    ### Dropping columns and converting to Numpy arrays
    X = df[['message']].values
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1).values
    print('Data is ready to be tokenized')
    
    return X, y, df
    
###############################################################################################################

### Creating the Tokenizer
def tokenize(text):
    ''''
    Tokenizer that gets a string and converts it in tokens, to be used in the Machine Learning model
    
    INPUT: df1_path - The location of messages.csv
           df2_path - The location of categories.csv
    
    OUTPUT: Lemmatized and tokenized text
    '''
    
    ### Getting list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    ### Replacing each url in text string with placeholder
    for url in detected_urls:
        bbtext = text.replace(url, "urlplaceholder")
    
    ### Tokenizing text
    tokens = word_tokenize(text)
    
    ### Initiating lemmatizer
    lemmatizer = WordNetLemmatizer()
        
    ### Lemmatizing, normalizing case, and removing leading/trailing white space and stop words
    return[lemmatizer.lemmatize(tok).lower().strip() for tok in tokens if tok not in stop_words]
    
###############################################################################################################

### Building the model
def build_model(X, y):
    
    '''
    Splits the data into training and testing sets, created the pipeline and finds the best combination of
    hyperparameters to optmize the model
    
    INPUT:  X  - Numpy array originated from the dataframe saved in the SQL server. It contains the predictor.
            y  - Numpy array originated from the dataframe saved in the SQL server. It contains the labels.
    
    OUTPUT: X_test - Testing data, after running train_test_split. To be used for evalutating the model.
            y_test - Teating labels, after running train_test_split. To be used for evaluating the model.
            cv     - The optimized model, after performing Grid Search
    
    '''
    
    ### Performing train test split
    print('Splitting the data')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    ### Flattening arrays
    X_train = X_train.ravel()
    X_test = X_test.ravel()
    
    ### Building the Pipeline
    print('Building the Pipeline')
    pipeline = Pipeline([('cvectorizer', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('rf_clf', RandomForestClassifier())
                        ])

    ### Specifying parameters for grid search
    parameters = {'cvectorizer__ngram_range': ((1, 1), (1, 2)),
                  'cvectorizer__max_df': (0.5, 0.75, 1.0),
                  'cvectorizer__max_features': (None, 5000, 10000),
                  'tfidf__use_idf': (True, False),
                  'rf_clf__min_samples_leaf':[1, 2],
                  'rf_clf__min_samples_split': [2, 4],
                  'rf_clf__n_estimators': [10, 30, 50]}
    
    ### Creating grid search object
    #cv = GridSearchCV(pipeline, param_grid=parameters)
    cv = RandomizedSearchCV(pipeline, param_distributions=parameters)
    
    ### Fitting the model
    print('Fitting the model')
    cv.fit(X_train, y_train)
    
    ### Printing best hyperparamaters and returning the model
    print('Printing best hyperparamaters and returning the model')
    print(cv.best_params_)
    
    return X_test, y_test, cv


###############################################################################################################

### Evaluating the model
def evaluate_model(model, X_test, y_test, df):
    
    """
    Loads the data used in the model - messages.csv and categories.csv
    Both files are located in the home directory
    
    INPUT: model  - The optimized model, with its hyperparameters optimized after Grid Search
           X_test - Testing data, after running train_test_split
           y_test - Teating labels, after running train_test_split
           df     - The dataframe saved in the SQL server from process_data.py
    
    OUTPUT: None
    """
    
    print('Printing metrics')
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean() 
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
    
    print('Creating the Classification report\n')
    label_columns = df.drop(['id', 'message', 'original', 'genre'], axis=1).columns.tolist()
    print(classification_report(y_test, y_pred, target_names=label_columns))

###############################################################################################################

### Saving the model
def save_model(model, model_path):
    
    """
    Saves the model to a pickle file
    
    INPUT:  model      - The optimized model, with its hyperparameters optimized after Grid Search
            model_path - The path where the pickle file will be saved
    
    OUTPUT: None
    """
    print('Saving the model')
    joblib.dump(model, model_path)
    print('Model saved!')
    
###############################################################################################################

### Running main
def main():
    if len(sys.argv) != 3:
        print(error_message)
    
    else:
        db_path, model_path = sys.argv[1:]
        
        print('Calling functions')
        X, y, df =  load_data(db_path)
        X_test, y_test, cv_final = build_model(X, y)
        evaluate_model(cv_final, X_test, y_test, df)
        save_model(cv_final.best_estimator_, model_path)
        print('End of code')
        
        
if __name__ == "__main__":
    main()
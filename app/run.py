import json
import plotly
import pandas as pd
import joblib

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Heatmap
import sqlalchemy
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

### Loading data
print('Loading data')
table_name  = 'disaster_response_mod'
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table(table_name, engine)
print("Data loaded! Now, let's load the model!")

### Loading model
print('Loading model')
model = joblib.load("../models/classifier.pkl")
print("Model loaded! Now, let's display all images!")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    
    # Bar chart data
    print('Preparing the data for plotting')    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Heatmap data
    columns_to_drop = ['id', 'message', 'original', 'genre']
    df2 = df.drop(columns_to_drop, axis=1).astype(float)
    categories = df2.columns.values
    category_count = [sum(df2[cat]) for cat in categories]
    correlation = df2.corr()
    correlation_val = correlation.values
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [Bar(x=genre_names, y=genre_counts)],

            'layout': {
                      'title': 'Distribution of Message Genres',
                       'yaxis': {'title': "Count"},
                       'xaxis': {'title': "Genre"}
                      }
        },
        
        {
            'data' : [Bar(x=categories, y=category_count)],          
        
        
            'layout': {
                       'title': 'Distribution of Categories',
                       'yaxis': {'title': "Count"},
                       'xaxis': {'title': "Genre"}
                      }
        
        },
        
        
        {
            'data' : [Heatmap(z=correlation_val, x=categories, y=categories)],          
        
        
            'layout': {
                       'title': 'Heatmap of Categories Correlation',
                       'height': 1500
                      }
        
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()

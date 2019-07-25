import json
import plotly
import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    # Select only the category columns
    category_ratios = df.drop(['id', 'message', 'original', 'genre'], axis=1).sum() / len(df)
    # Sort the categories in descending order
    category_ratios = category_ratios.sort_values(ascending=False)
    categories = list(category_ratios.index)

    set(string.punctuation)
    # Get all words in rows
    all_words_in_messages = pd.Series(' '.join(df['message']).lower().split())
    # Remove the punctuations
    all_words_in_messages = all_words_in_messages[~all_words_in_messages.isin(set(string.punctuation))]
    # Remove the stopwords
    most_frequent_words = all_words_in_messages[~all_words_in_messages.isin(stopwords.words("english"))].value_counts()[
                          :10]
    # Get the  top ten most frequent words
    most_frequent_words_names = list(most_frequent_words.index)
    most_frequent_words_pct = 100 * np.array(list(most_frequent_words.values)) / df.shape[0]


    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=categories,
                    y=category_ratios
                )
            ],

            'layout': {
                'title': 'Message Ratio by Category',
                'yaxis': {
                    'title': "Ratios",
                    'automargin':True
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -45,
                    'automargin':True
                }
            }
        },
        {
            'data': [
                Bar(
                    x=most_frequent_words_names,
                    y=most_frequent_words_pct
                )
            ],

            'layout': {
                'title': 'Top 10 Words Frequency in Percentage',
                'yaxis': {
                    'title': 'Frequency',
                    'automargin': True
                },
                'xaxis': {
                    'title': 'Top 10 words',
                    'automargin': True
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON, data_set=df)


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
    app.run(host='127.0.0.1', port=3001, debug=True)


if __name__ == '__main__':
    main()
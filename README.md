# Disaster Response Pipeline Project

## Overview

This project analyzes disaster data from Figure Eight to build a model for an API that classifies disaster messages.
A web app is created for an emergency worker to input a new message and  have the message classified in several message categories. As such to be able to response to the message in the most timely and appropriate manner.  
To understand the model and the data that trains it, the web app will visualize the past data. The Results section below displays its visualizations. 

To build the model, the project has an ETL pipeline that process the message data `process_data.py` and save it to a SQL database `DisasterResponse.db`. 
An ML pipeline `train_classifier.py` then loads the database and train a classifier using NLP techniques and Random Forest classifier. The trained model is then saved to a pickle file `classifier.pkl`. 

The Flask web app loads the model to perform message classification with message entered by user. It also loads the SQL database to perform data visualization. 

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:3001/ in your browser

## Results

### The Webapp

[image1]: https://github.com/jiewwantan/Disaster_Response_ETL_NLP_pipelines/blob/master/clf_webapp.png "Message Classifier"
![Message Classifier][image1]

### Message Genre Visualized

[image2]: https://github.com/jiewwantan/Disaster_Response_ETL_NLP_pipelines/blob/master/message_genre.png "Message Genre"
![Message Genre][image2]

### Message Ratio Visualized

[image3]: https://github.com/jiewwantan/Disaster_Response_ETL_NLP_pipelines/blob/master/message_ratio.png "Message Ratio"
![Message Ratio][image3]

### Word Frequency Visualized

[image4]: https://github.com/jiewwantan/Disaster_Response_ETL_NLP_pipelines/blob/master/words_freq.png "Word Frequency"
![Word Frequency][image4]


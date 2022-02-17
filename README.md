# Udacity-Diaster-Response-Project

## Project Motivation

In this project, you'll apply these skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

In the Project Workspace, you'll find a data set containing real messages that were sent during disaster events. You will be creating a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

Your project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. This project will show off your software skills, including your ability to create basic data pipelines and write clean, organized code!



## Requirements
- Python 3.5+
- NumPy
- Pandas
- SciPy
- Sciki-Learn
- NLTK
- SQLalchemy
- Pickle library
- Flask
- Plotly

## Files in the Repository

### Folders

```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # input data 
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse.db   # database where the data will be saved

- models
|- train_classifier.py
|- disaster_model.pkl  # saved model

- notebooks
|- ETL Pipeline Preparation.ipynb # Jupyter notebook used to load and cleanse the data. The code in this notebook was modularized into ```process_data.py```
|- ML Pipeline Preparation.ipynb  # Jupyter notebook used to run the Machine model. The code in this notebook was modularized into ```train_classifier.py``` 

- README.md
```

### Scripts

```process_data.py```: This script does the following

1. Loads the messages and categories datasets
2. Merges the two datasets
3. Cleans the data
4. Stores it in a SQLite database


```train_classifier.py```: This script does the following

1. Loads data from the SQLite database
2. Splits the dataset into training and test sets
3. Builds a text processing and machine learning pipeline
4. Trains and tunes a model using GridSearchCV
5. Outputs results on the test set
6. Exports the final model as a pickle file


```run.py```: This script does the following

1. Loads data from the SQLite database
2. Loads the model (pickle file) saved at the end of train_classifier.py
3. Displays visuals and receives user input text for model


## Executing Scripts - Instructions:
### Run the following commands in the project's root directory to set up database and model:

Part 1 - Running ```process_data.py```

`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

Part 2 - Running ```train_classifier.py```

`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

Part 3 - Running ```run.py```

`python run.py`

#### Go to http://0.0.0.0:3001/


## Results

Results can be found in the folder "Screenshots"


## Acknowledgements
https://pandas.pydata.org
<br/>
https://numpy.or
http://nltk.org</br/>
<br/>
https://scikit-learn.org
<br/>
https://www.sqlalchemy.org/
<br/>
https://github.com/othneildrew/Best-README-Template/blob/master/README.md
</br>
https://flask.palletsprojects.com/en/2.0.x/
</br>
https://plotly.com/

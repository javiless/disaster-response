### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Execution Instructions](#execution)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.  The code should run with no issues using Python versions 3.*.

## Project Motivation<a name="motivation"></a>

For this project, I was interested in deploying a web app that successfully classifies messages introduced by the users that are in an emergency situation. For this purpose I built a Machine Learning (ML) pipeline that trains a ML model based on a training dataset of more than 26,000 emergency messages.

## File Descriptions <a name="files"></a>

- app
 - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # training dataset with the messages
|- disaster_messages.csv  # training dataset with the categories of the messages
|- process_data.py  # script that generate the cleaned database based on both csv files.

- models
|- train_classifier.py # script that loads data from database, tokenizes it, builds the ML model, evaluates it and saves it as a pkl file. 

- README.md

## Execution Instructions<a name="execution"></a>

1. Download all files to your local machine.

2. Run the following commands in the project's root directory to set up your database and model.

    2.1 To run ETL pipeline that cleans data and stores in database
    
        >python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/INSERT_YOUR_NAME_DB.db
        
        
    2.2 To run ML pipeline that trains classifier and saves
        
        >python models/train_classifier.py data/INSERT_YOUR_NAME_DB.db models/INSERT_YOUR_CLASSIFIER_NAME.pkl

3. Run the following command in the "app" directory to run your web app.
    `python run.py`

3. Go to web URL shown in the terminal after "Running on ...", and enter a message to see how it is classified for each of the disaster response categories.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to [Figure Eight](https://appen.com/) and Udacity (https://www.udacity.com/) for providing the datasets, means and motivation for carrying out this project.

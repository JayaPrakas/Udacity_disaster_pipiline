# Udacity Disaster Response Pipeline

## Table of contents

- [Installation](#installation)
- [Project Description](#Project-Description)
- [Project summary](#Project-summary)
- [Project motivation](#project-motivation)
- [Interacting with this project](#Interacting-with-this-project)
- [File descriptions](#File-descriptions)
- [Instructions](#Instructions)
- [Acknowledgements](#Acknowledgements)
- [License](#License)


## Installation

The following are used along with python.

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- flask
- sqlalchemy
- json
- pickle
- joblib

## Project Description

This project is of creating disaster response using python and using it as web app with flask.

## Project summary

In this project, a model is built to analyze disaster data from Figure Eight that classifies disaster messages, using data set containing real messages that were sent during disaster events, machine learning pipeline is created to categorize these events so that messages can be sent to appropriate disaster relief agency.

This project also includes a web app where an emergency worker can input a new message and get classification results in several categories.

So, this app can be used in disaster times where emergency worker can input message and get results and also visualization and forward it to corresponding relief agency.

## Project motivation

As a requirement for the nanodegree [become a data scientist](https://www.udacity.com/course/data-scientist-nanodegree--nd025) of [Udacity](https://www.udacity.com/).

## Interacting with this project

- To interact, clone the repo: `git clone https://github.com/JayaPrakas/udacity_disaster_pipeline.git` or fork this repository

## File descriptions

With this download you'll find the following files.

```text
    
- app
| - templates
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model after running

- README.md

- LICENSE

```

## Instructions

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`     
        
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/

## Acknowledgements

Thanks to [Udacity](https://www.udacity.com/) for data, starter code and instructions

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



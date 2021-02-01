# Udacity Disaster Response Pipeline

## Table of contents

- [Installation](#installation)
- [Project Description](#Project-Description)
- [Project motivation](#project-motivation)
- [Interacting with this project](#Interacting-with-this-project)
- [File descriptions](#file-descriptions)
- [Acknowledgements](#Acknowledgements)
- [License](#License)


## Installation

This notebook runs on **anaconda distribution** with **python**. All required libraries are already there in anaconda and don't require additional python libraries.

## Project Description

This project is analysis of [Airbnb](https://www.airbnb.com) data of Asheville, North Carolina, United States. 

## Project motivation

As a requirement for the nanodegree [become a data scientist](https://www.udacity.com/course/data-scientist-nanodegree--nd025) of [Udacity](https://www.udacity.com/).

## Interacting with this project

- To interact, clone the repo: `git clone https://github.com/JayaPrakas/Asheville_Airbnb_Analysis.git` or fork this repository

## File descriptions

With this download you'll find the following files.

```text

├── Asheville_airbnb.ipynb

├── data

    ├── calendar_.csv
    
    ├── listings_.csv

```

- Asheville_airbnb.ipynb ==> Notebook to investigate trends of bookings on Airbnb in Asheville.

- data consisting of following files

- calendar.csv           ==> Booking information of houses in Seattle.

- listings.csv           ==> data containing details of facilities available of houses, host information and lot of other information in Asheville.

## Instructions

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Acknowledgements

Thanks to [Udacity](https://www.udacity.com/) and [Airbnb](https://airbnb.com) for data

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



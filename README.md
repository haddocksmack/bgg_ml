# Predicting Board Game Ratings on Board Game Geek with Regression Model

Senior capstone project
Corey Reed
Bachelor of Science, Data Analytics
Western Governors University

## Research Question or Organizational Need

What are the features of a board game (player count, playtime, theme, mechanics, etc.),
and what combination of said features, that lead to a successful board game?
Can these features help predict future success?
Can combinations of features that typically lead to success be identified?

## Scope of Project

In this project, I developed a machine learning regression model capable of predicting
a board game’s rating score on BoardGameGeek.com accurately.
I also successfully interpreted the model to identify combinations of features
of a board game that lead to a game receiving a higher rating on the website.


## Source of Data

Data was downloaded from [Kaggle](https://www.kaggle.com/datasets/mattadamhouser/ranked-board-game-data-from-boardgamegeek)
The data includes the top 2000 rated games from Board Game Geek at the time of the scrapping, which was around July 2023.

The data is stored in the `\data\` directory

## How to Run Analysis

First, open and run all cells in `BGG_EDA.ipynb`. Then, open and run all cells in `BGG_ML_clean.ipynb`

## How to load model

    import pickle

    with open("production_model.pkl", "rb") as f:
    model = pickle.load(f)




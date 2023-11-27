import pandas as pd
from sklearn.ensemble import RandomForestClassifier # Import RandomForest Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import numpy as np
#make sure you have your labels correct
#some files have this in the file - others it is in the description
#if it is in the file you can copy them here then delete that line in the file
col_names = ['TEAM', 'MATCH UP', 'GAME DATE', 'W/L', 'MIN', 'PTS',
'FGM', 'FGA', 'FG%', '3PM', '3PA', '3P%', 'FTM', 'FTA', 'FT%',
'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'PF', '+/-']
# load dataset

s2021_2022 = pd.read_csv("2021-2022Season.csv", header=None, names=col_names)
s2022_2023 = pd.read_csv("2022-2023Season.csv", header=None, names=col_names)
s2023_2024 = pd.read_csv("2023-2024Season.csv", header=None, names=col_names)

#print(s2021_2022)
#print(s2022_2023)
#print(s2023_2024)

# Combining the three datasets to make one to train on
combined = pd.concat([s2021_2022, s2022_2023, s2023_2024])

# This takes out the vs. or @ so that we just have one team as the 'opposing team'
def get_opposing_team(row):
    team = row['TEAM']
    matchup = row['MATCH UP']
    # Splitting the matchup string on the known delimiters
    teams_in_matchup = matchup.replace(' @ ', '/').replace(' vs. ', '/').split('/')
    # Return the team that is not equal to the team in the 'TEAM' column
    return [x for x in teams_in_matchup if x != team][0]

combined['MATCH UP'] = combined.apply(get_opposing_team, axis=1)

# Dropping the first row that is a copy of the column names
combined = combined.drop(0)

# Maps all the teams to a number so the algorithms can work with them
team_mapping = {'ATL':1, 'BOS':2, 'BKN':3, 'CHA':4, 'CHI':5, 
                'CLE':6, 'DAL':7, 'DEN':8, 'DET':9, 'GSW':10,
                'HOU':11, 'IND':12, 'LAC':13, 'LAL':14, 'MEM':15,
                'MIA':16, 'MIL':17, 'MIN':18, 'NOP':19, 'NYK':20,
                'OKC':21, 'ORL':22, 'PHI':23, 'PHX':24, 'POR':25,
                'SAC':26, 'SAS':27, 'TOR':28, 'UTA':29, 'WAS':20}

combined['TEAM'] = combined['TEAM'].map(team_mapping)
combined['MATCH UP'] = combined['MATCH UP'].map(team_mapping)

win_loss_mapping = {'W':1, 'L':0}
combined['W/L'] = combined['W/L'].map(win_loss_mapping)
# Parse the 'GAME DATE' column to datetime if it's not already in that format
combined['GAME DATE'] = pd.to_datetime(combined['GAME DATE'])

# Extract features from the 'GAME DATE' column
combined['YEAR'] = combined['GAME DATE'].dt.year
combined['MONTH'] = combined['GAME DATE'].dt.month
combined['DAY'] = combined['GAME DATE'].dt.day

combined.drop('GAME DATE', axis=1, inplace=True)

print(combined)

feature_cols = ['TEAM', 'MATCH UP', 'MIN', 'PTS',
'FGM', 'FGA', 'FG%', '3PM', '3PA', '3P%', 'FTM', 'FTA', 'FT%',
'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'PF', '+/-', 'YEAR', 'MONTH', 'DAY']
X = combined[feature_cols] # Features
y = combined['W/L'] # Target variable


# Split dataset into training set and test set
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)
y_test_pred=forest.predict(X_test)
y_train_pred=forest.predict(X_train)
forest_train = metrics.accuracy_score(y_train, y_train_pred)
forest_test = metrics.accuracy_score(y_test, y_test_pred)

# Model Accuracy, how often is the classifier correct?
print("Accuracy of testing:",metrics.accuracy_score(y_test, y_test_pred))
print(f"Random forest train / test accuracies: {forest_train} / {forest_test}")
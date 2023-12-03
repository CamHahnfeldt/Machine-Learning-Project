import matplotlib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier # Import RandomForest Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn import model_selection

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

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

#print(combined)

feature_cols = ['TEAM', 'MATCH UP', #'PTS', #removing points gave slightly better accuracy
'FGM', 'FGA', 'FG%', '3PM', '3PA', '3P%', 'FTM', 'FTA', 'FT%',
'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'PF', 'YEAR', 'MONTH', 'DAY']
X = combined[feature_cols] # Features
y = combined['W/L'] # Target variable

# Split dataset into training set and test set
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

forest = RandomForestClassifier(n_estimators=100, max_depth = 11, random_state=42)
forest.fit(X_train, y_train)
y_test_pred=forest.predict(X_test)
y_train_pred=forest.predict(X_train)
forest_train = metrics.accuracy_score(y_train, y_train_pred)
forest_test = metrics.accuracy_score(y_test, y_test_pred)

# # Model Accuracy, how often is the classifier correct?
# print("Accuracy of Random Forest testing:",metrics.accuracy_score(y_test, y_test_pred))
# print(f"Random forest train / test accuracies: {forest_train} / {forest_test}")

## Determining the best tree depth for each classifier
def lookAtModels(models):
    results=[]
    names=[]
    for name, model in models:
        skfold = StratifiedKFold(n_splits=10)
        cv_results= cross_val_score(model, X_train, y_train, cv=skfold,scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print(f'{name}: {cv_results.mean()} ({cv_results.std()})')
    return names, results

#models =[]
#for d in range(2,20):
#    models.append((f'Tree depth{d}',GradientBoostingClassifier(max_depth=d)))

#names,results = lookAtModels(models)

## Gradient Boosting algorithm

# Initialize Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(n_estimators=70, learning_rate=0.1, max_depth=4, random_state=42)

# Train the model
gb_clf.fit(X_train, y_train)

# Make predictions
y_pred_test = gb_clf.predict(X_test)
y_pred_train = gb_clf.predict(X_train)
boost_train = metrics.accuracy_score(y_train, y_pred_train)
boost_test = metrics.accuracy_score(y_test, y_pred_test)

# Evaluate the model
print("Accuracy of Gradient Boosting testing:",metrics.accuracy_score(y_test, y_pred_test))
print(f"Gradient Boosting train / test accuracies: {boost_train} / {boost_test}")
#
# accuracy = accuracy_score(y_test, y_pred_test)
# print(f"Model Accuracy: {accuracy:.2f}")

# #k-fold
# kfold = model_selection.KFold(n_splits=10)
# model_kfold = GradientBoostingClassifier()
# results_kfold = model_selection.cross_val_score(model_kfold, X, y, cv=kfold)
# print(f"K-Fold Accuracy: {results_kfold.mean()} ({results_kfold.std()})")
#
# #stratified k-fold
# skfold = StratifiedKFold(n_splits=10)
# model_skfold = GradientBoostingClassifier()
# results_skfold = model_selection.cross_val_score(model_skfold, X, y, cv=skfold)
# print(f"Stratified K-Fold Accuracy: {results_skfold.mean()} ({results_skfold.std()})")

# #LOOCV
# loocv = model_selection.LeaveOneOut()
# model_loocv = GradientBoostingClassifier()
# results_loocv = model_selection.cross_val_score(model_loocv, X, y, cv=loocv)
# print(f"LOOCV Accuracy: {results_loocv.mean()} ({results_loocv.std()})")

#Repeated Random Test-Train splits
kfold2 = model_selection.ShuffleSplit(n_splits=10, test_size=0.30, random_state=42)
model_shufflecv = GradientBoostingClassifier()
results_shufflecv = model_selection.cross_val_score(model_shufflecv, X, y, cv=kfold2)
print(f"Repeated Random Accuracy: {results_shufflecv.mean()} ({results_shufflecv.std()})")
#
# def lookAtModels(models):
#     results=[]
#     names=[]
#     for name, model in models:
#         kfold2 = model_selection.ShuffleSplit(n_splits=10, test_size=0.30,
#                                               random_state=42)
#         model_shufflecv = GradientBoostingClassifier()
#         results_shufflecv = model_selection.cross_val_score(model_shufflecv, X,
#                                                             y, cv=kfold2)
#         results.append(results_shufflecv)
#         names.append(name)
#         print(f'{name}: {results_shufflecv.mean()} ({results_shufflecv.std()})')
#     return names, results
#
# #make models
# models =[]
# for d in range(2,9):
#     models.append((f'Tree depth {d}',
#                    GradientBoostingClassifier(criterion="entropy",max_depth=d)))
#
# #see your results
# names,results = lookAtModels(models)

#Create Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred_test)
#Display Confusion Matrix
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
cm_display.plot(cmap='Blues', values_format='d')

plt.title('Confusion Matrix')
plt.show()

#Look at the other results
print(metrics.classification_report(y_test,y_pred_test))

#Calculate sensitivity and specificity for each class
n_classes = conf_mat.shape[0]
for i in range(n_classes):
    tp = conf_mat[i, i]
    fn = sum(conf_mat[i, :]) - tp
    fp = sum(conf_mat[:, i]) - tp
    tn = sum(sum(conf_mat)) - tp - fn - fp

    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    print(f"Class {i}: TPR = {tpr:.2f}, TNR = {tnr:.2f}")

#ROC curve
from sklearn.metrics import RocCurveDisplay
fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred_test)
roc_auc = metrics.auc(fpr, tpr)
roc_displayDT = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Gradient Boosting')

# Precision-Recall Curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
prec, recall, _ = precision_recall_curve(y_test, y_pred_test)
pr_displayDT = PrecisionRecallDisplay(precision=prec, recall=recall)

#side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
roc_displayDT.plot(ax=ax1)
pr_displayDT.plot(ax=ax2)
plt.show()
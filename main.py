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

feature_cols = ['TEAM', 'MATCH UP',
'FGM', 'FGA', 'FG%', '3PM', '3PA', '3P%', 'FTM', 'FTA', 'FT%',
'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'PF', 'YEAR', 'MONTH', 'DAY']
X = combined[feature_cols] # Features
y = combined['W/L'] # Target variable

# Split dataset into training set and test set
# 50% training and 50% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Finding the best value for the max_depth
def lookAtModels(models):
    results=[]
    names=[]
    for name, model in models:
        skfold = StratifiedKFold(n_splits=10)
        cv_results= cross_val_score(model, X_train, y_train, cv=skfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print(f'{name}: {cv_results.mean()} ({cv_results.std()})')
    return names, results#not sure why I want lists yet

models =[]
for d in range(2,16):
    models.append((f'Tree depth {d}',RandomForestClassifier(max_depth=d)))
    #see your results
names,results = lookAtModels(models)

D2_mean = np.mean(results[0])
D2_std= np.std(results[0])
D3_mean = np.mean(results[1])
D3_std = np.std(results[1])
D4_mean = np.mean(results[2])
D4_std = np.std(results[2])
D5_mean = np.mean(results[3])
D5_std = np.std(results[3])
D6_mean = np.mean(results[4])
D6_std = np.std(results[4])
D7_mean = np.mean(results[5])
D7_std = np.std(results[5])
D8_mean = np.mean(results[6])
D8_std = np.std(results[6])
D9_mean = np.mean(results[7])
D9_std = np.std(results[7])
D10_mean = np.mean(results[8])
D10_std = np.std(results[8])
D11_mean = np.mean(results[9])
D11_std = np.std(results[9])
D12_mean = np.mean(results[10])
D12_std = np.std(results[10])
D13_mean = np.mean(results[11])
D13_std = np.std(results[11])
D14_mean = np.mean(results[12])
D14_std = np.std(results[12])
D15_mean = np.mean(results[13])
D15_std = np.std(results[13])

# Plotting the depth results
depth =['2','3','4','5','6','7','8', '9', '10', '11', '12', '13', '14', '15']
x_pos = np.arange(len(depth))
#print(x_pos)
Means = [D2_mean,D3_mean,D4_mean,D5_mean,D6_mean,D7_mean,D8_mean, D9_mean, 
         D10_mean, D11_mean, D12_mean, D13_mean, D14_mean, D15_mean]
error = [D2_std,D3_std,D4_std,D5_std,D6_std,D7_std,D8_std, D9_std, D10_std, D11_std,
        D12_std, D13_std, D14_std, D15_std]
# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, Means, yerr=error, align='center', alpha=0.5, ecolor='black',
capsize=10)
ax.set_ylabel('Accuracy')
ax.set_xticks(x_pos)
ax.set_xticklabels(depth)
ax.set_title('Accuracy scores of different depths (Random Forest)')
ax.yaxis.grid(True)

# Finding the best value for the number of estimators
def getForestEstimators(n):
    results = []
    for i in range(1, n+1):
        model = RandomForestClassifier(n_estimators=i,max_depth = 11, random_state=42)
        model.fit(X_train,y_train)
        y_test_pred = model.predict(X_test)
        results.append(metrics.accuracy_score(y_test, y_test_pred))
    return results

getAccuracy = getForestEstimators(101)

#for count, value in enumerate(getAccuracy):
    #print(f'k: {count} accuracy: {value}')
print(f'Best n: {getAccuracy.index(max(getAccuracy))}'
     f' accuracy: {getAccuracy[getAccuracy.index(max(getAccuracy))]}')

forest = RandomForestClassifier(n_estimators=44, max_depth = 11, random_state=42)
forest.fit(X_train, y_train)
y_test_pred=forest.predict(X_test)
y_train_pred=forest.predict(X_train)
forest_train = metrics.accuracy_score(y_train, y_train_pred)
forest_test = metrics.accuracy_score(y_test, y_test_pred)

print("Random Forest")

# Model Accuracy, how often is the classifier correct?
print("Accuracy of Random Forest testing:",metrics.accuracy_score(y_test, y_test_pred))
print(f"Random forest train / test accuracies: {forest_train} / {forest_test}")

#Create Confusion Matrix
conf_mat = confusion_matrix(y_test, y_test_pred)
#Display Confusion Matrix
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
cm_display.plot(cmap='Blues', values_format='d')
plt.title('Random Forest Confusion Matrix')
plt.show()

#Look at the other results
print(metrics.classification_report(y_test,y_test_pred))

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
# from sklearn.metrics import RocCurveDisplay
# fpr, tpr, thresholds = metrics.roc_curve(y_test,y_test_pred)
# roc_auc = metrics.auc(fpr, tpr)
# roc_displayDT = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Random Forest')

# Precision-Recall Curve
# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import PrecisionRecallDisplay
# prec, recall, _ = precision_recall_curve(y_test, y_test_pred)
# pr_displayDT = PrecisionRecallDisplay(precision=prec, recall=recall)

#Curves side by side
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
# roc_displayDT.plot(ax=ax1)
# pr_displayDT.plot(ax=ax2)
# plt.show()

#k-fold cross validation
kfold = model_selection.KFold(n_splits=10)
model_kfold = RandomForestClassifier()
results_kfold = model_selection.cross_val_score(model_kfold, X, y, cv=kfold)
print(f"K-Fold Accuracy: {results_kfold.mean()} ({results_kfold.std()})")

#stratified k-fold
skfold = StratifiedKFold(n_splits=10)
model_skfold = RandomForestClassifier()
results_skfold = model_selection.cross_val_score(model_skfold, X, y, cv=skfold)
print(f"Stratified K-Fold Accuracy: {results_skfold.mean()} ({results_skfold.std()})")

#LOOCV
'''
Takes too long to run

loocv = model_selection.LeaveOneOut()
model_loocv = RandomForestClassifier()
results_loocv = model_selection.cross_val_score(model_loocv, X, y, cv=loocv)
print(f"LOOCV Accuracy: {results_loocv.mean()} ({results_loocv.std()})")
'''

#Repeated Random Test-Train splits
kfold2 = model_selection.ShuffleSplit(n_splits=10, test_size=0.30, random_state=42)
model_shufflecv = RandomForestClassifier()
results_shufflecv = model_selection.cross_val_score(model_shufflecv, X, y, cv=kfold2)
print(f"Repeated Random Test-Train Splits Accuracy: {results_shufflecv.mean()} ({results_shufflecv.std()})")


## Gradient Boosting algorithm

# Finding best value for max_depth
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

#make all your models
models =[]
for d in range(2,9):
    models.append((f'Tree depth {d}',GradientBoostingClassifier(max_depth=d)))
    #see your results
names,results = lookAtModels(models)

D2_mean = np.mean(results[0])
D2_std= np.std(results[0])
D3_mean = np.mean(results[1])
D3_std = np.std(results[1])
D4_mean = np.mean(results[2])
D4_std = np.std(results[2])
D5_mean = np.mean(results[3])
D5_std = np.std(results[3])
D6_mean = np.mean(results[4])
D6_std = np.std(results[4])
D7_mean = np.mean(results[5])
D7_std = np.std(results[5])
D8_mean = np.mean(results[6])
D8_std = np.std(results[6])

#Plotting the depth results
depth =['2','3','4','5','6','7','8']
x_pos = np.arange(len(depth))
#print(x_pos)
Means = [D2_mean,D3_mean,D4_mean,D5_mean,D6_mean,D7_mean,D8_mean]
error = [D2_std,D3_std,D4_std,D5_std,D6_std,D7_std,D8_std]
# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, Means, yerr=error, align='center', alpha=0.5, ecolor='black',
capsize=10)
ax.set_ylabel('Accuracy')
ax.set_xticks(x_pos)
ax.set_xticklabels(depth)
ax.set_title('Accuracy scores of different depths (Gradient Boosting)')
ax.yaxis.grid(True)

# Finding the best value for the number of estimators
def getGradientEstimators(n):
    results = []
    for i in range(1, n+1):
        model = GradientBoostingClassifier(n_estimators=i, learning_rate = 0.1, max_depth = 4, random_state=42)
        model.fit(X_train,y_train)
        y_test_pred = model.predict(X_test)
        results.append(metrics.accuracy_score(y_test, y_test_pred))
    return results

getAccuracy = getGradientEstimators(101)

#for count, value in enumerate(getAccuracy):
    #print(f'k: {count} accuracy: {value}')
print(f'Best n: {getAccuracy.index(max(getAccuracy))}'
     f' accuracy: {getAccuracy[getAccuracy.index(max(getAccuracy))]}')

# Initialize Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(n_estimators=86, learning_rate=0.1, max_depth=4, random_state=42)

# Train the model
gb_clf.fit(X_train, y_train)

# Make predictions
y_pred_test = gb_clf.predict(X_test)
y_pred_train = gb_clf.predict(X_train)
boost_train = metrics.accuracy_score(y_train, y_pred_train)
boost_test = metrics.accuracy_score(y_test, y_pred_test)

print("\nGradient Boosting")

# Evaluate the model
print("Accuracy of Gradient Boosting testing:",metrics.accuracy_score(y_test, y_pred_test))
print(f"Gradient Boosting train / test accuracies: {boost_train} / {boost_test}")

accuracy = accuracy_score(y_test, y_pred_test)
print(f"Model Accuracy: {accuracy:.2f}")

#Create Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred_test)
#Display Confusion Matrix
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
cm_display.plot(cmap='Blues', values_format='d')
plt.title('Gradient Boosting Confusion Matrix')
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
# from sklearn.metrics import RocCurveDisplay
# fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred_test)
# roc_auc = metrics.auc(fpr, tpr)
# roc_displayDT = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Gradient Boosting')

# Precision-Recall Curve
# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import PrecisionRecallDisplay
# prec, recall, _ = precision_recall_curve(y_test, y_pred_test)
# pr_displayDT = PrecisionRecallDisplay(precision=prec, recall=recall)

#Curves side by side
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
# roc_displayDT.plot(ax=ax1)
# pr_displayDT.plot(ax=ax2)
# plt.show()

#k-fold cross validation
kfold = model_selection.KFold(n_splits=10)
model_kfold = GradientBoostingClassifier()
results_kfold = model_selection.cross_val_score(model_kfold, X, y, cv=kfold)
print(f"K-Fold Accuracy: {results_kfold.mean()} ({results_kfold.std()})")

#stratified k-fold
skfold = StratifiedKFold(n_splits=10)
model_skfold = GradientBoostingClassifier()
results_skfold = model_selection.cross_val_score(model_skfold, X, y, cv=skfold)
print(f"Stratified K-Fold Accuracy: {results_skfold.mean()} ({results_skfold.std()})")

#LOOCV
'''
Takes too long to run

loocv = model_selection.LeaveOneOut()
model_loocv = GradientBoostingClassifier()
results_loocv = model_selection.cross_val_score(model_loocv, X, y, cv=loocv)
print(f"LOOCV Accuracy: {results_loocv.mean()} ({results_loocv.std()})")
'''

#Repeated Random Test-Train splits
kfold2 = model_selection.ShuffleSplit(n_splits=10, test_size=0.30, random_state=42)
model_shufflecv = GradientBoostingClassifier()
results_shufflecv = model_selection.cross_val_score(model_shufflecv, X, y, cv=kfold2)
print(f"Repeated Random Test-Train Splits Accuracy: {results_shufflecv.mean()} ({results_shufflecv.std()})")

#ROC curves on same plot
#set up plotting area
plt.figure()
#Random Forest plot ROC curve
fpr, tpr, _ = metrics.roc_curve(y_test,y_test_pred)
roc_auc = round(metrics.roc_auc_score(y_test, y_test_pred), 4)
plt.plot(fpr, tpr, label="Random Forest, AUC="+str(roc_auc))
#Gradient Boosting plot ROC curve
fpr, tpr, _ = metrics.roc_curve(y_test,y_pred_test)
roc_auc = round(metrics.roc_auc_score(y_test, y_pred_test), 4)
plt.plot(fpr, tpr, label="Gradient Boosting, AUC="+str(roc_auc))
#add legend
plt.legend()
plt.show()

#Precision-Recall curves on same plot
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
plt.figure()
#Random Forest
prec, recall, _ = precision_recall_curve(y_test, y_test_pred)
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall, estimator_name="Random Forest")
pr_display.plot(ax=plt.gca(), color='b')
#Gradient Boosting
prec, recall, _ = precision_recall_curve(y_test, y_pred_test)
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall, estimator_name="Gradient Boosting")
pr_display.plot(ax=plt.gca(), color='r')
#plot
plt.show()





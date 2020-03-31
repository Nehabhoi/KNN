## IMPORTS ##
import re
import csv
import pandas as pd
import numpy as np
import sys

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

### CONSTANTS ###
DATA_LINE = 15
ORIGINAL_FILENAME = "wine_quality.csv"
CLEANED_FILENAME = "wine_quality_cleaned2.csv"
K_DEFAULT = 5
T_DEFAULT = 5
R1_DEFAULT = 1
R2_DEFAULT = 25

### CLEAN CSV FILE - ONLY RUN ONCE TO GET CLEANED CSV ###
def clean_csv(original_file_path, clean_file_path):
    file = open('{}'.format(ORIGINAL_FILENAME), "r")

    fieldnames = []
    for i in range(0, DATA_LINE-1):
        line = file.readline()
        m = re.search('(?<=@attribute )(.+)(?= \w+)', line)
        if (m):
            fieldnames.append(m.group(0))

    with open('{}'.format(CLEANED_FILENAME), "w", newline="") as clean_csv:
        clean_csv.write(','.join(fieldnames) + "\n")
        for line in file:
            clean_csv.write(file.readline())
    file.close()

### UTILITY FUNCTION ###
def display_group_by(data, col_name):
    z = data.groupby(col_name).count()
    print("Grouped by '{}' :\n".format(col_name), z)

def separate_data_from_label(data):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, 11].values
    return [X, y]

### KNN MODELS ###
def knn(data, K, test_ratio=0.20):
    [X, y] = separate_data_from_label(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0, test_size=test_ratio)

    classifier = KNeighborsClassifier(n_neighbors=K)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print("Result of KNN using K={} is: \n".format(K))
    print("Confusion matrix: \n", confusion_matrix(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))
    print("Training accuracy: ", metrics.accuracy_score(y_test, y_pred))

def knn_with_cross_validation(data, K, T, test_ratio=0.2):
    [X, y] = separate_data_from_label(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0, test_size=test_ratio)

    classifier = KNeighborsClassifier(n_neighbors=K)
    scores = cross_val_score(classifier, X, y, cv=T)
    print("scores: ", scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def knn_with_gridSearchCV(data, K, T, R1, R2,test_ratio=0.2):
    [X, y] = separate_data_from_label(data)
    param_grid = dict(n_neighbors=np.arange(R1, R2+1))  # inclusive

    clf = GridSearchCV(KNeighborsClassifier(), param_grid,
                    cv=T, scoring="accuracy")
    clf.fit(X, y)

    grid_df = pd.DataFrame(clf.cv_results_).sort_values(
        by=['rank_test_score']).set_index('rank_test_score')
    display_cols = ['param_n_neighbors', "mean_test_score", "std_test_score"]
    
    print("Grid result by descending rank test score is: \n{}".format(grid_df[display_cols]))
    print("Best score: ", clf.best_score_)
    print("Best param: ", clf.best_params_)
    print("Best estimator: ", clf.best_estimator_)

    best_line = { key : clf.cv_results_[key][clf.best_index_] for key in clf.cv_results_.keys() }
    print("Best accuracy: %0.2f (+/- %0.2f) when k = %.0f" %
          (best_line['mean_test_score'], best_line['std_test_score'] * 2, best_line['param_n_neighbors']))

if __name__ == "__main__":
    # Clean CSV - only run once to create a clean CSV file.
    # clean_csv(ORIGINAL_FILENAME, CLEANED_FILENAME) 

    # Parse arguments
    if (len(sys.argv) != 5):
        print(">>>>> Not enough arguments, required K T R1 R2")
        print(">>>>> Use default value: K=5, T=5, R1=1, R2=25")
        [K, T, R1, R2] = [K_DEFAULT, T_DEFAULT, R1_DEFAULT, R2_DEFAULT] 
    else:
        [K, T, R1, R2] = [int(arg) for arg in sys.argv[1:]] 

    ## WITH FULL DATASETS ##
    print("\n== == == == == == == == READ CSV == == == == == == == ==\n")
    data = pd.read_csv("wine_quality_cleaned.csv")
    print(data)
    # display_group_by(data, "quality")

    # KNN Models
    print("\n== == == == == == == == KNN - Predefined K = {}== == == == == == == ==\n".format(K))
    knn(data, K=K)

    print("\n== == == == == == == == KNN - Predefined K = {} + Cross Validation == == == == == == == ==\n".format(K))
    knn_with_cross_validation(data, K=K, T=T)

    print("\n== == == == == == == == KNN - GridSearchCV with K = [{},...,{}] == == == == == == == ==\n".format(R1, R2))
    knn_with_gridSearchCV(data, K=K, T=T, R1=R1, R2=R2)

    ## WITH FILTERED DATASET - REMOVE LABEL WITH ONLY 1 MEMBER ##
    data = pd.read_csv("wine_quality_cleaned.csv")
    data = data[data.groupby('quality').quality.transform(
        'count') > 2].copy()  # Remove quality that has count < 2
    display_group_by(data, "quality")

    # KNN Models
    print("\n== == == == == == == == KNN - Predefined K = {} == == == == == == == ==\n".format(K))
    knn(data, K=K)

    print("\n== == == == == == == == KNN - Predefined K = {} + Cross Validation == == == == == == == ==\n".format(K))
    knn_with_cross_validation(data, K=K, T=T)

    print(
        "\n== == == == == == == == KNN - GridSearchCV with K = [{},...,{}] == == == == == == == ==\n".format(R1, R2))
    knn_with_gridSearchCV(data, K=K, T=T, R1=R1, R2=R2)


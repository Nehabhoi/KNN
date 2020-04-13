## IMPORTS ##
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, KMeansSMOTE, SVMSMOTE, ADASYN
from sklearn.preprocessing import StandardScaler, QuantileTransformer, RobustScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics, preprocessing, pipeline
import re
import csv
import pandas as pd
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt

### CONSTANTS ###
DATA_LINE = 15
ORIGINAL_FILENAME = "wine_quality.csv"
CLEANED_FILENAME = "wine_quality_cleaned2.csv"
K_DEFAULT = 4
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
def knn(X, y, K, test_ratio=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0, test_size=test_ratio)

    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.fit_transform(X_test)

    pipeline = make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=K))
    model = pipeline.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    print("Result of KNN WITH PREPROCESSING using K={} is: \n".format(K))
    labels = [0, 3, 4, 5, 6, 7, 8, 9, 11]
    cm = confusion_matrix(y_test, y_pred, labels)
    print("Confusion Matrix: \n", cm)
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=labels, yticklabels=labels)

    print("Classification report: \n", classification_report(y_test, y_pred))
    print("Training accuracy: ", metrics.accuracy_score(y_test, y_pred))


def knn_untouched(X, y, K, test_ratio=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0, test_size=test_ratio)

    knn = KNeighborsClassifier(n_neighbors=K, n_jobs=-1)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    print("Result of KNN with NO PREPROCESSING using K={} is: \n".format(K))
    labels = [0, 3, 4, 5, 6, 7, 8, 9, 11]
    cm = confusion_matrix(y_test, y_pred, labels)
    print("Confusion Matrix: \n", cm)
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=labels, yticklabels=labels)

    print("Classification report: \n", classification_report(y_test, y_pred))
    print("Training accuracy: ", metrics.accuracy_score(y_test, y_pred))

def knn_with_cross_validation(X, y, K, T, test_ratio=0.2):
    pipeline = make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=K, n_jobs=-1))
    scores = cross_val_score(pipeline, X, y, cv=T)

    print("\n== == == == == == == == KNN - Predefined K = {} + Cross Validation == == == == == == == ==\n".format(K))
    print("scores: ", scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def knn_with_gridSearchCV(X, y, K, T, R1, R2, test_ratio=0.2):
    search_space = [{'knn__n_neighbors': range(R1, R2+1)}]

    sc = StandardScaler()
    X_std = sc.fit_transform(X)

    knn = KNeighborsClassifier(n_jobs=-1)

    pipeline = Pipeline([('sc', sc), ('knn', knn)])
    clf = GridSearchCV(pipeline, search_space, cv=T, scoring="accuracy")
    clf.fit(X_std, y)

    grid_df = pd.DataFrame(clf.cv_results_).sort_values(
        by=['rank_test_score']).set_index('rank_test_score')
    display_cols = ['param_knn__n_neighbors',
                    "mean_test_score", "std_test_score"]

    print(
        "\n== == == == == == == == KNN - GridSearchCV with K = [{},...,{}] == == == == == == == ==\n".format(R1, R2))
    plt.plot(list(range(R1, R2+1)), clf.cv_results_["mean_test_score"])
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated Accuracy')
    print("Grid result by descending rank test score is: \n{}".format(
        grid_df[display_cols]))
    print("Best score: ", clf.best_score_)
    print("Best param: ", clf.best_params_)
    print("Best estimator: ", clf.best_estimator_)

    best_line = {key: clf.cv_results_[key][clf.best_index_]
                 for key in clf.cv_results_.keys()}
    print("Best accuracy: %0.2f (+/- %0.2f) when k = %.0f" %
          (best_line['mean_test_score'], best_line['std_test_score'] * 2, best_line['param_knn__n_neighbors']))

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

    ###### KEEP UNBALANCED DATASET WITH SCALING ######
    [X, y] = separate_data_from_label(data)
    knn(X, y, K=K)
    knn_with_cross_validation(X, y, K=K, T=T)
    knn_with_gridSearchCV(X, y, K=K, T=T, R1=R1, R2=R2)

    ###### BALANCE DATASET WITH SCALING ######
    data = pd.read_csv("wine_quality_cleaned.csv", delimiter=',')
    data = data[data.groupby('quality').quality.transform(
        'count') >= 2].copy()  # Remove any label with members < 2
    print("GROUP BY QUALITY")
    print("We removed quality 0 and 11")
    data.groupby(['quality']).count()   

    ## Fix imbalance dataset
    [X, y] = separate_data_from_label(data)
    X_res, y_res = SMOTE(k_neighbors=3, random_state=0).fit_resample(X, y)
    print("Original data shape: X.shape = ", X.shape, "y.shape = ", y.shape)
    print("New data shape: X.shape = ", X_res.shape, "y.shape = ", y_res.shape)

    ## SPLIT TRAIN-TEST ##
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, random_state=0, test_size=0.2)

    knn(X_res, y_res, K=K)
    knn_with_cross_validation(X_res, y_res, K=K, T=T)
    knn_with_gridSearchCV(X_res, y_res, K=K, T=T, R1=R1, R2=R2)


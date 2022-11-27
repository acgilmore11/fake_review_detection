import sklearn
import numpy as np
import os.path
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split



def svm_feature_selection(features, labels, feature_names):
    """
    Trains svm model with inputted features. Visualizes and returns top n coeff feature names
    """
    # check if trained model already exists
    if not os.path.exists('lin_svm.joblib'):
        train_lin_model(features, labels)

    # uncomment this line to predict classification for test set
    # y_pred = clf.predict(X_test)
    clf = load('lin_svm.joblib')
    return feature_selection(clf, feature_names)


def train_lin_model(features, labels):
    clf = svm.SVC(kernel='linear')
    clf.fit(features, labels)

    # save svm model
    dump(clf, 'lin_svm.joblib')

def feature_selection(classifier, feature_names, n=10):
    coef = classifier.coef_.ravel()
    
    #get top 5 positive and negative coefficients
    top_coefficients = np.hstack([np.argsort(coef)[-int(n/2):], np.argsort(coef)[:int(n/2)]])
    print(top_coefficients)

    # create plot
    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(n), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + n), feature_names[top_coefficients], rotation=60, ha='right')
    plt.show()

    return [feature_names[i] for i in top_coefficients]



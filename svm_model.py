import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split



def svm_feature_selection(table):
    """
    Trains svm model with inputted features. Visualizes and returns top n coeff feature names
    """
    # Note: if features are added/removed in the future, this list needs to be modified to reflect change
    feature_names = ['rating', 'singleton','RationOfCapL', 'RatioOfFirstPerson', 'RatioOfExclamation', 'density', 'MRD', 'DFTLM', 
                        'MNR', 'PPR', 'PNR', 'RL', 'rating_dev', 'reviewer_dev', 'rating_variance', 'rating_entropy', 'avg_dev_from_entity_avg']
    # normalizes features
    features = preprocessing.normalize(table[feature_names])
    labels = table['label'].to_numpy()

    # 70% training and 30% test. Can always change this split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3,random_state=42069)
    
    # will use linear for faster execution time, may implement rbf later for better accuracy
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)

    # uncomment this line to predict classification for test set
    # y_pred = clf.predict(X_test)
    return plot_coefficients(clf, feature_names)


def plot_coefficients(classifier, feature_names, n=10):
    coef = classifier.coef_.ravel()
    
    #get top 5 positive and negative coefficients
    top_coefficients = np.hstack([np.argsort(coef)[-int(n/2):], np.argsort(coef)[:int(n/2)]])

    # create plot
    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(n), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + n), feature_names[top_coefficients], rotation=60, ha='right')
    plt.show()

    return [feature_names[i] for i in top_coefficients]



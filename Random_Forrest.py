from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import nltk
#from feature_extraction import *
import pandas as pd
import numpy as np
import string
import re
import collections
from statistics import mean
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats import entropy
from imblearn.under_sampling import NearMiss
from sklearn.preprocessing import Normalizer
import joblib

# perform undersampling

def train_rf(train_features, test_features, train_labels, test_labels,table):
    

    #labels = np.array(table['label'])
    #table = table.drop('label', axis = 1)
    scale= StandardScaler()
    scaled_train = scale.fit_transform(train_features) 
    # Split the data into training and testing sets
    #train_features, test_features, train_labels, test_labels = train_test_split(scaled_train, labels, test_size = 0.25, random_state = 42)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(scaled_train, train_labels)
    y_pred=clf.predict(test_features)
    print("Accuracy:",metrics.accuracy_score(test_labels, y_pred))
    joblib.dump(clf, "rf.joblib")
    clf = joblib.load("rf.joblib")
    return feature_importance(clf,table)

def feature_importance(clf,datatset):
    

    feats = {}
    for feature, importance in zip(datatset.columns, clf.feature_importances_):
        feats[feature] = importance
    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-Importance'})
    importances = importances.sort_values(by='Gini-Importance', ascending=False)
    importances = importances.reset_index()
    importances = importances.rename(columns={'index': 'Features'})
    sns.set(font_scale = 5)
    sns.set(style="whitegrid", color_codes=True, font_scale = 1.7)
    fig, ax = plt.subplots()
    fig.set_size_inches(30,15)
    sns.barplot(x=importances['Gini-Importance'], y=importances['Features'], data=importances, color='skyblue')
    plt.xlabel('Importance', fontsize=25, weight = 'bold')
    plt.ylabel('Features', fontsize=25, weight = 'bold')
    plt.title('Feature Importance', fontsize=25, weight = 'bold')
    display(plt.show())
    display(importances)
    return [importances['Features'][i] for i in range(10)]
    
def pca_visualization(table):
    scale= StandardScaler()
    labels = np.array(table['label'])
    table = table.drop('label', axis = 1)
    scaled_train = scale.fit_transform(table)
    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled_train)
    fig = px.scatter(components, x=0, y=1, color=labels)
    fig.show()
    

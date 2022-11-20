import pandas as pd
import numpy as np
from feature_extraction import undersample
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def preprocess_rf(table):
    #dataset = pd.read_csv("test.csv")
    dataset = undersample(table)
    labels = np.array(dataset['label'])
    dataset= dataset.drop('label', axis = 1)
    dataset= dataset.drop('date', axis = 1)
    dataset= dataset.drop('review', axis = 1)

    # Saving feature names for later use
    feature_list = list(dataset.columns)
    # Convert to numpy array
    train = np.array(dataset)
    train_features, test_features, train_labels, test_labels = train_test_split(dataset, labels, test_size = 0.25, random_state = 42)
    return [train_features,test_features,train_labels,test_labels]

def pca_rf(train_features,test_features):
    pca = PCA(n_components=6)
    pca.fit(train_features)
    X_train_scaled_pca = pca.transform(train_features)
    X_test_scaled_pca = pca.transform(test_features)
    return [ X_train_scaled_pca, X_test_scaled_pca]

def train_rf(train,label,X_train_scaled_pca, train_labels,test_labels):
    #train, labels = make_classification(n_samples=1000, n_features=4,
    #                        n_informative=2, n_redundant=0,
    #                        random_state=0, shuffle=False)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train_scaled_pca, train_labels)
    y_pred=clf.predict(X_test_scaled_pca)
    from sklearn import metrics
    print("Accuracy:",metrics.accuracy_score(test_labels, y_pred))
    joblib.dump(clf, "rf.joblib")

def feature_importance(datatset):
    clf = joblib.load("rf.joblib")

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

def pca_visualization(train_features):
    pca_test = PCA(n_components=6)
    pca_test.fit(train_features)
    sns.set(style='whitegrid')
    plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.axvline(linewidth=4, color='r', linestyle = '--', x=1, ymin=0, ymax=1)
    display(plt.show())
    evr = pca_test.explained_variance_ratio_
    cvr = np.cumsum(pca_test.explained_variance_ratio_)
    pca_df = pd.DataFrame()
    pca_df['Cumulative Variance Ratio'] = cvr
    pca_df['Explained Variance Ratio'] = evr
    display(pca_df.head(10))



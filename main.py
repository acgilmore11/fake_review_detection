import pandas as pd
import numpy as np
import nltk
from pathlib import Path 
from svm_model import *
from feature_extraction import *
from Random_Forrest import *


def main():
    # check if sampled dataset already in repo
    sampled_filepath = Path('dataset_v1.csv')
    if not os.path.exists(sampled_filepath):
        # check if feature engineering has already been completed and consolidated into table
        filepath = Path('review_feature_table.csv')
        if not os.path.exists(filepath):
            # download punkt package
            print("shouldn't be here")
            nltk.download('punkt')

            data_path = "YelpCSV"
            cols_meta = ["user_id", "prod_id", "rating", "label", "date"]
            meta_data = pd.read_csv(data_path+"/metadata.csv", names=cols_meta)
            cols_reviewContent = ["user_id", "prod_id", "date", "review"]
            reviewContent = pd.read_csv(
            data_path+"/reviewContent.csv", names=cols_reviewContent)
            table = pd.concat([meta_data, reviewContent["review"]], axis=1).dropna()

            # consolidates all features into one table
            table=pd.concat([table, review_metadata(table), review_textual(table), reviewer_burst(table), behavioral_features(table), rating_features(table), temporal(table)], axis=1)  

            # writes dataframe containing all features to csv file
            filepath.parent.mkdir(parents=True, exist_ok=True)  
            table.to_csv(filepath, index=False)
    
        table = pd.read_csv(filepath)

        sampled_filepath.parent.mkdir(parents=True, exist_ok=True)
        #undersample
        undersampled_table = undersample(table)
        undersampled_table.to_csv(sampled_filepath, index=False)

    sample = pd.read_csv(sampled_filepath)
    features, labels, feature_names = pre_process(sample)

    # remove
     

    # split data set into 80% training data and 20% test data
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2,random_state=573)
    
    # random forest and feature selection
    # # random forest and feature selection
    
    # will return accuracy score plot for different parameters
    compare_rf(train_features,train_labels,test_features,test_labels)
    # # will return names of top 10 features and return accuracy of the model
    rf_top_features = train_rf(train_features, test_features, train_labels, test_labels,feature_names)
    print(rf_top_features)
    pca_visualization(train_features,train_labels)
    # will return names of top 10 features and return accuracy of the model
    rf_top_features = train_rf(train_features, test_features, train_labels, test_labels,table)
    print(rf_top_features)

    # SVM and feature selection
    # will return names of top 10 features
    svm_top_features = svm_feature_selection(train_features, train_labels, test_features, test_labels, feature_names)
    print(svm_top_features)
    

    # deep learning approach with selected features



if __name__ == "__main__":
    main()
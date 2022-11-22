import pandas as pd
import numpy as np
import nltk
from svm_model import *
from feature_extraction import *


def main():
    # download punkt package
    nltk.download('punkt')

    data_path = "YelpCSV"
    cols_meta = ["user_id", "prod_id", "rating", "label", "date"]
    meta_data = pd.read_csv(data_path+"/metadata.csv", names=cols_meta)
    cols_reviewContent = ["user_id", "prod_id", "date", "review"]
    reviewContent = pd.read_csv(
        data_path+"/reviewContent.csv", names=cols_reviewContent)
    table = pd.concat([meta_data, reviewContent["review"]], axis=1).dropna()

    # perform undersampling

    # uncomment this line to create sample by keeping fake reviews and obtaining random sample
    # of real reviews such that of each class (real and fake) are equal
    # table = undersample(table)

    # uncomment this line to create test sample of only 200 entries (100 fake, 100 real)
    # this can be used to test models with shorter execution time
    # table = undersample_test(table)

    # combines original sample dataframe with feature columns
    # TO ADD FEATURE: create new function in feature_extraction.py that returns Series/Dataframe object,
    #                add method call to pd.concat function
    table=pd.concat([table, review_metadata(table), review_textual(table), reviewer_burst(table), behavioral_features(table), rating_features(table)], axis=1)
    
    # if we want to do under sampling after feature engineering 
    # just uncomment the following line
    # table = undersample_v2(table)
    
    # random forest and feature selection
    # Note: if features are added in the future, this function needs to be modified to add new feature names
    svm_top_features = svm_feature_selection(table)
    print(svm_top_features)

    # SVM and feature selection
    

    # deep learning approach with selected features

    # prints first 10 rows to check
    # print(table[:100])


if __name__ == "__main__":
    main()

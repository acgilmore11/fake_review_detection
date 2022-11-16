import pandas as pd
import numpy as np
import nltk
from feature_extraction import *

def main():
    #download punkt package
    nltk.download('punkt')

    data_path = "YelpCSV"
    cols_meta = ["user_id", "prod_id", "rating", "label", "date"]
    meta_data = pd.read_csv(data_path+"/metadata.csv", names = cols_meta)
    cols_reviewContent = ["user_id", "prod_id", "date", "review"]
    reviewContent = pd.read_csv(data_path+"/reviewContent.csv", names = cols_reviewContent)
    table = pd.concat([meta_data, reviewContent["review"]], axis = 1).dropna()

    #perform undersampling
    table = undersample(table)

    #combines original sample dataframe with feature columns
    #TO ADD FEATURE: create new function in feature_extraction.py that returns Series object,
    #                add method call to pd.concat function
    table = pd.concat([table, rating_deviation(table), singleton(table), review_centric_textual(table)], axis=1)

    #prints first 10 rows to check
    print(table)

if __name__ == "__main__":
    main()
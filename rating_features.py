import pandas as pd
import numpy as np
import collections
from scipy.stats import entropy
data_path = "YelpCSV"
cols_meta = ["user_id", "prod_id", "rating", "label", "date"]
meta_data = pd.read_csv(data_path+"/metadata.csv", names=cols_meta)
cols_reviewContent = ["user_id", "prod_id", "date", "review"]
reviewContent = pd.read_csv(
    data_path+"/reviewContent.csv", names=cols_reviewContent)
table = pd.concat([meta_data, reviewContent["review"]], axis=1).dropna()
# print(table)
rating_features = table[["user_id", "rating"]]
"""
Ratios , i.e., the ratio of negative, positive, and extreme reviews 
(i.e., whose rating corresponds to the extremes of the considered rating interval);
"""
# filter 
user_ids = rating_features['rating'].unique().tolist()
# get rating > =3
positive_rating = rating_features[rating_features['rating'] >=3]
positive_counts = positive_rating['user_id'].value_counts()
# get rating < 3
negative_rating = rating_features[rating_features['rating'] < 3]
negative_counts = negative_rating['user_id'].value_counts()
# get rating == 1
extreme_negative_rating = rating_features[rating_features['rating'] == 1]
extreme_negative_counts = extreme_negative_rating['user_id'].value_counts()
# get rating == 5
extreme_positive_rating = rating_features[rating_features['rating'] == 5]
extreme_positive_counts = extreme_positive_rating['user_id'].value_counts()
# simply iterate through the original dataframe to add values
"""
Average deviation from entity's average, 
i.e., the evaluation if a user's ratings assigned in her/his reviews are 
often very different from the mean of an entity's rating(far lower for instance);
"""
avg_rating_of_prods = table[["prod_id", "rating"]].groupby('prod_id').mean()
# simply using the rating minus the avegerage in the original table
grouped_users = rating_features.groupby('user_id')
"""
Rating entropy , 
i.e., the entropy of rating distribution of user's reviews;
"""
user_rating_entropy = collections.defaultdict(int)

for name, group in grouped_users:
    rating_peruser = list(group["rating"])
    user_rating_entropy[name] = entropy(rating_peruser)
# search for user id and add the entropy value to the original table
"""
Rating variance , i.e., the squared deviation of the rating assigned by a user with respect to the ratings mean. 
The variance as a rating feature has been added to further describe how the ratings for a particular user are distributed.
"""
avg_rating_of_users = rating_features.groupby('user_id').mean()
# simply subtract the rating for each row in the original table


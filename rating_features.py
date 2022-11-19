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
table = table.sample(n = 1000)
# print(table)
# def geneate_rating_features(table):
rating_features = table[["user_id", "rating"]]
"""
Ratios , i.e., the ratio of negative, positive, and extreme reviews 
(i.e., whose rating corresponds to the extremes of the considered rating interval);
"""
# filter 
user_ids = rating_features['rating'].unique().tolist()
numer_of_reviews_of_each_user = rating_features['user_id'].value_counts()
# get rating > =3
positive_rating = rating_features[rating_features['rating'] >=3]
positive_counts = positive_rating['user_id'].value_counts()
positive_users = set(positive_rating["user_id"].unique().tolist())

# get rating < 3
negative_rating = rating_features[rating_features['rating'] < 3]
negative_counts = negative_rating['user_id'].value_counts()
negative_users = set(negative_rating["user_id"].unique().tolist())
# get rating == 1
extreme_negative_rating = rating_features[rating_features['rating'] == 1]
extreme_negative_counts = extreme_negative_rating['user_id'].value_counts()
extreme_negative_users = set(extreme_negative_rating["user_id"].unique().tolist())
# get rating == 5
extreme_positive_rating = rating_features[rating_features['rating'] == 5]
extreme_positive_counts = extreme_positive_rating['user_id'].value_counts()
extreme_positive_users = set(extreme_positive_rating["user_id"].unique().tolist())
# simply iterate through the original dataframe to add values

"""
Average deviation from entity's average, 
i.e., the evaluation if a user's ratings assigned in her/his reviews are 
often very different from the mean of an entity's rating(far lower for instance);
"""
avg_rating_of_prods = table[["prod_id", "rating"]].groupby('prod_id').mean()
# simply using the rating minus the avegerage in the original table
"""
Rating entropy , 
i.e., the entropy of rating distribution of user's reviews;
"""
grouped_users = rating_features.groupby('user_id')
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
cols_reviewer_rating = ["positive", "negative", "extreme_positive", "extreme_negative","rating_variance","rating_entropy", "avg_dev_from_entity_avg"]
rating_features_output = collections.defaultdict(list)
for index, row in table.iterrows():
    # ratio calculation 
    user_id = row["user_id"]
    total_num_review = numer_of_reviews_of_each_user.loc[user_id]
    if user_id in positive_users:
        
        rating_features_output["positive"].append(positive_counts.loc[user_id]/total_num_review)
    else:
        rating_features_output["positive"].append(0)

    if user_id in negative_users:
        rating_features_output["negative"].append(negative_counts.loc[user_id]/total_num_review)
    else:
        rating_features_output["negative"].append(0)

    if user_id in extreme_positive_users:
        rating_features_output["extreme_positive"].append(extreme_positive_counts.loc[user_id]/total_num_review)
    else:
        rating_features_output["extreme_positive"].append(0)

    if user_id in extreme_negative_users:
        rating_features_output["extreme_negative"].append(extreme_negative_counts.loc[user_id]/total_num_review)
    else:
        rating_features_output["extreme_negative"].append(0)

    # rating variance 
    rating_features_output["rating_variance"].append((row["rating"] - avg_rating_of_users.loc[user_id]["rating"])**2)

    # rating_entropy
    rating_features_output["rating_entropy"].append(user_rating_entropy[user_id])

    # Average deviation from entity's average
    rating_features_output["avg_dev_from_entity_avg"].append(avg_rating_of_prods.loc[row["prod_id"]]["rating"])
rating_features_df = pd.DataFrame.from_dict(rating_features_output)
table = pd.concat([table, rating_features_df],ignore_index=True,  axis=1)
print(table)
# print(rating_features_df)

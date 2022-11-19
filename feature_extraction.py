import pandas as pd
import numpy as np
import string
import re
from statistics import mean
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
import collections
from scipy.stats import entropy

def undersample(table):
    """
    Performs undersampling on data by keeping fake reviews and obtaining random sample
    of real reviews such that # of each class (real and fake) are equal
    """
    fake_reviews = table[table['label'] == -1]
    real_reviews = table[table['label'] == 1].sample(n=fake_reviews.shape[0])
    sample = pd.concat([fake_reviews, real_reviews], ignore_index=True)
    return sample


def review_metadata(table):
    """
    Metadata features: 
    Rating: rating(1-5) given in review (no calculation needed)
    Singleton: 1 if review is only one written by user on date, 0 otherwise
    Rating Deviation: |curr_rating - avg_prod_rating| / # of review for product
    """
    # rating deviation
    avg_rating = table[['prod_id', 'rating']].groupby(['prod_id']).agg(avg=pd.NamedAgg(column='rating', aggfunc='mean'),
                                                                       count=pd.NamedAgg(column='rating', aggfunc='count'))
    table = pd.merge(table, avg_rating, on='prod_id', how='inner')
    table['rating_deviation'] = abs(
        table['rating'] - table['avg']) / table['count']

    # singleton
    date_counts = table.groupby(['user_id', 'date']).size().to_frame('size')
    table = pd.merge(table, date_counts, on=['user_id', 'date'], how='left')
    table['singleton'] = table['size'] == 1
    table['singleton'] = table['singleton'].astype('int')

    return table[['singleton', 'rating_deviation']]


def review_textual(table):
    """
    Text statistics: 
    Number of words, i.e., the length of the review in terms of words;
    Ratio of capital letters, i.e., the number of words containing capital letters with respect to the total number of words in the review;
    Ratio of capital words, i.e., considering the words where all the letters are uppercase;
    Ratio of first person pronouns,e.g.,‘I’,‘mine’,‘my’, etc.;
    Ratio of ‘exclamation’ sentences, i.e., ending with the symbol ‘!’. 
    """
    statistics_table = {"RationOfCapL": [], "RatioOfCapW": [
    ], "RatioOfFirstPerson": [], "RatioOfExclamation": []}  # , "sentiment":[]
    first_person_pronouns = set(
        ["i", "mine", "my", "me", "we", "our", "us", "ourselves", "ours"])

    for i, row in table.iterrows():
        sentences = sent_tokenize(row["review"])
        countExclamation = 0
        wordCountinAReview = 0
        countCapL = 0
        countCapW = 0
        countFirstP = 0
        for sentence in sentences:
            if sentence[-1] == "!":
                countExclamation += 1
            sentence = sentence.translate(
                str.maketrans('', '', string.punctuation))
            sentence = sentence.split(" ")

            wordCountinAReview += len(sentence)

            for word in sentence:
                if word.isupper():
                    countCapW += 1
                if word.lower() in first_person_pronouns:
                    countFirstP += 1
                for w in word:
                    if w.isupper():
                        countCapL += 1
                        break

        RatioOfExclamation = countExclamation/len(sentences)
        RationOfCapL = countCapL/wordCountinAReview
        RatioOfCapW = countCapW/wordCountinAReview
        RatioOfFirstPerson = countFirstP/wordCountinAReview
        statistics_table["RatioOfExclamation"].append(RatioOfExclamation)
        statistics_table["RationOfCapL"].append(RationOfCapL)
        statistics_table["RatioOfCapW"].append(RatioOfCapW)
        statistics_table["RatioOfFirstPerson"].append(RatioOfFirstPerson)

    text_statistics = pd.DataFrame.from_dict(statistics_table)
    return text_statistics


def reviewer_burst(table):
    """
    Burst features: 
    Density: # reviews for entity on given day
    Mean Rating Deviation(MRD): |avg_prod_rating_on_date - avg_prod_rating|
    Deviation From Local Mean(DFTLM):  |prod_rating - avg_prod_rating_on_date| / # of reviews on date
    """
    # Density
    df1 = table.groupby(['prod_id', 'date'], as_index=False)[
        'review'].agg('count')
    df1.rename(columns={'review': 'density'}, inplace=True)
    table = pd.merge(table, df1, left_on=['prod_id', 'date'], right_on=[
                     'prod_id', 'date'], validate='m:1')

    # Mean Rating Deviation
    df4 = table.groupby(['prod_id', 'date'], as_index=False).agg(avg_date=pd.NamedAgg(column='rating', aggfunc='mean'),
                                                                 count_date=pd.NamedAgg(column='rating', aggfunc='count'))
    table = pd.merge(table, df4, left_on=['prod_id', 'date'], right_on=[
                     'prod_id', 'date'], validate='m:1')

    # Deviation From The Local Mean
    df3 = table.groupby(['prod_id'], as_index=False).agg(
        avg=pd.NamedAgg(column='rating', aggfunc=np.mean))
    table = pd.merge(table, df3, left_on=['prod_id'], right_on=[
                     'prod_id'], validate='m:1')
    table['DFTLM'] = abs(table['rating'] - table['avg_date']
                         ) / table['count_date']
    table['MRD'] = abs(table['avg_date'] - table['avg'])

    return table[['density', 'MRD', 'DFTLM']]


#i dont think this is working properly
def reviewer_temporal(table):

    df=table[['user_id','date']]
    df['Date_pd'] = pd.to_datetime(df['date'])
    df['Date_int'] = abs(pd.to_datetime(df['Date_pd']).astype(np.int64))

    df=df[['user_id','Date_int']]
    res = df.groupby('user_id').agg(['mean','var'])
    res.columns = ['_'.join(c) for c in res.columns.values]
    #res['Date_var'] = pd.to_timedelta(res['Date_int_var'])/np.timedelta64(1,'D')
    res['Date_mean'] = pd.to_datetime(res['Date_int_mean'])
    res['Date_var'] = pd.to_timedelta(res['Date_int_var'])
    #res['Date_var']=res['Date_var'].fillna(0).astype(int)
    # res = res[['Date_mean','Date_var']]
    print(res)


# this function is producing an array memory error, needs to be fixed
def reviewer_textual(table):
    df = table[['user_id', 'review']]

    sentences = [sent.lower() for sent in df['review']]
    processed_sentences = [re.sub('[^a-zA-Z]', ' ', sent)
                           for sent in sentences]
    processed_article = [re.sub(r'\s+', ' ', sent)
                         for sent in processed_sentences]

    df['Word_number_average'] = df['review'].str.split(" ").str.len()

    tfidfvectorizer = TfidfVectorizer(
        min_df=0, analyzer='word', stop_words='english')
    tfidf_wm = tfidfvectorizer.fit_transform(processed_article)
    tfidf_tokens = tfidfvectorizer.get_feature_names_out()
    #df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),index = df['id'],columns = tfidf_tokens)
    df_tfidfvect = tfidf_wm.toarray()

    cosine = 1-pairwise_distances(df_tfidfvect, metric='cosine')
    np.fill_diagonal(cosine, 0)

    max_content_similarity = []
    for i in range(0, len(cosine)):
        max_content_similarity.append(round(max(cosine[i]), 3))

    df['Maximum_Content_Similarity'] = max_content_similarity

    avg_content_similarity = []
    for i in range(0, len(cosine)):
        avg_content_similarity.append(round(mean(cosine[i]), 3))

    df['Average_Content_Similarity'] = avg_content_similarity

    return df[['Maximum_Content_Similarity', 'Average_Content_Similarity', 'Word_number_average']]


def max_num_reviews(table):
    count_table = table[['user_id', 'date', 'rating']].groupby(['user_id', 'date']).agg(count=pd.NamedAgg(column='rating', aggfunc='count'))
    res = count_table.groupby(['user_id']).agg(MNR=pd.NamedAgg(column='count', aggfunc='max'))
    table = pd.merge(table, res, on='user_id', how='left')
    return table['MNR']

def geneate_rating_features(table):
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
    return rating_features_df



# def percent_pos_reviews(table):
#     # temp = table['user_id']
    
#     temp = pd.concat([table['user_id'], table['rating'] >= 4], axis=1)
#     print(temp['rating'])
#     temp2 = temp.groupby(['user_id', 'rating']).size().unstack().fillna(0)
#     print(temp2)
#     # temp['PPR'] = temp[True] / (temp[True] + temp[False])
#     # table = pd.merge(table, temp[['user_id', 'PPR' ]], on='user_id', how='left')
#     # print(temp.iloc[:, 0])
    



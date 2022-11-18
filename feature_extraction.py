import pandas as pd
import numpy as np
import string
import re
from statistics import mean
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances


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

# def percent_pos_reviews(table):
#     # temp = table['user_id']
    
#     temp = pd.concat([table['user_id'], table['rating'] >= 4], axis=1)
#     print(temp['rating'])
#     temp2 = temp.groupby(['user_id', 'rating']).size().unstack().fillna(0)
#     print(temp2)
#     # temp['PPR'] = temp[True] / (temp[True] + temp[False])
#     # table = pd.merge(table, temp[['user_id', 'PPR' ]], on='user_id', how='left')
#     # print(temp.iloc[:, 0])
    



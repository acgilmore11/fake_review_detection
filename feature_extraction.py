import pandas as pd
import numpy as np
import string
import re
from statistics import mean
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances


#performs undersampling on data
#keeps fake reviews and obtains random sample of real reviews such that # of each class are equal
def undersample(table):
    fake_reviews = table[table['label'] == -1]
    real_reviews = table[table['label'] == 1].sample(n=fake_reviews.shape[0])
    sample = pd.concat([fake_reviews,real_reviews], ignore_index=True)
    return sample

#rating_deviation feature:  the deviation of the evaluation provided in the review with respect to the entity’s average rating
#abs(product_rating - avg_product_rating)/4
def rating_deviation(table):
    avg_rating = table[['prod_id', 'rating']].groupby(['prod_id']).mean().rename(columns={'rating': 'avg_rating'})
    table = pd.merge(table, avg_rating, on='prod_id', how='inner')
    table['rating_deviation'] = abs(table['rating'] - table['avg_rating']) / 4
    return table['rating_deviation']

#singleton feature: 1 if review is the only review written that day by user, 0 otherwise
def singleton(table):
    date_counts = table.groupby(['user_id', 'date']).size().to_frame('size')
    table = pd.merge(table, date_counts, on=['user_id', 'date'], how='left')
    table['singleton'] = table['size'] == 1
    table['singleton'] = table['singleton'].astype('int')
    return table['singleton']

#textual features
def review_centric_textual(table):
    """
    Text statistics: 
    Number of words, i.e., the length of the review in terms of words;
    Ratio of capital letters, i.e., the number of words containing capital letters with respect to the total number of words in the review;
    Ratio of capital words, i.e., considering the words where all the letters are uppercase;
    Ratio of first person pronouns,e.g.,‘I’,‘mine’,‘my’, etc.;
    Ratio of ‘exclamation’ sentences, i.e., ending with the symbol ‘!’. 
    """
    statistics_table = {"RationOfCapL":[], "RatioOfCapW":[], "RatioOfFirstPerson":[], "RatioOfExclamation":[]} #, "sentiment":[]
    first_person_pronouns = set(["i", "mine", "my", "me", "we", "our", "us", "ourselves", "ours"])


    for i , row in table.iterrows():
        sentences = sent_tokenize(row["review"])
        countExclamation = 0
        wordCountinAReview = 0
        countCapL = 0
        countCapW = 0
        countFirstP = 0
        for sentence in sentences:
            if sentence[-1] == "!":
                countExclamation += 1
            sentence = sentence.translate(str.maketrans('', '', string.punctuation))
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

        RatioOfExclamation =  countExclamation/len(sentences) 
        RationOfCapL = countCapL/wordCountinAReview
        RatioOfCapW = countCapW/wordCountinAReview
        RatioOfFirstPerson = countFirstP/wordCountinAReview
        statistics_table["RatioOfExclamation"].append(RatioOfExclamation)
        statistics_table["RationOfCapL"].append(RationOfCapL)
        statistics_table["RatioOfCapW"].append(RatioOfCapW)
        statistics_table["RatioOfFirstPerson"].append(RatioOfFirstPerson)



    text_statistics = pd.DataFrame.from_dict(statistics_table)
    return text_statistics

def reviewer_burst_features(table):
    ## Density
    df1 = table.groupby([ 'prod_id', 'date'], as_index=False)['review'].agg('count')
    df1.rename(columns={'review': 'density'}, inplace=True)
    table = pd.merge(table,df1, left_on=['prod_id', 'date'],right_on=['prod_id', 'date'], validate = 'm:1')
    ## Mean Rating Deviation
    df2 = table.groupby([ 'prod_id', 'date'], as_index=False).agg(MRD=pd.NamedAgg(column ='rating', aggfunc='mad'))
    table = pd.merge(table,df2, left_on=['prod_id', 'date'],right_on=['prod_id', 'date'], validate = 'm:1')
    ## Mean Rating Deviation
    df4 = table.groupby([ 'prod_id','date'], as_index=False).agg(avg_date=pd.NamedAgg(column ='rating', aggfunc=np.mean))
    table = pd.merge(table,df4, left_on=['prod_id','date'],right_on=['prod_id','date'], validate = 'm:1')
    ## Deviation From The Local Mean
    df3 = table.groupby([ 'prod_id'], as_index=False).agg(avg=pd.NamedAgg(column ='rating', aggfunc=np.mean))
    table = pd.merge(table,df3, left_on=['prod_id'],right_on=['prod_id'], validate = 'm:1')
    table['DFTLM'] = table['rating'] - table['avg']
    table['MRD'] = (table['avg_date'] - table['avg'])

    table = table.drop(['avg'], axis=1)
    table = table.drop(['MRD'], axis=1)

    return table


#this function is producing an array memory error, needs to be fixed
def reviewer_centric_textual(table):
    df=table[['user_id','review']]

    sentences = [sent.lower() for sent in df['review']]
    processed_sentences = [re.sub('[^a-zA-Z]', ' ',sent) for sent in sentences]
    processed_article = [re.sub(r'\s+', ' ', sent) for sent in processed_sentences]

    df['Word_number_average'] = df['review'].str.split(" ").str.len()

    tfidfvectorizer  = TfidfVectorizer(min_df=0,analyzer='word',stop_words= 'english')
    tfidf_wm=tfidfvectorizer.fit_transform(processed_article)
    tfidf_tokens = tfidfvectorizer.get_feature_names_out()
    #df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),index = df['id'],columns = tfidf_tokens)
    df_tfidfvect = tfidf_wm.toarray()

    cosine=1-pairwise_distances(df_tfidfvect, metric='cosine')
    np.fill_diagonal(cosine,0)

    max_content_similarity=[]
    for i in range(0,len(cosine)):
        max_content_similarity.append(round(max(cosine[i]),3))

    df['Maximum_Content_Similarity']=max_content_similarity    

    avg_content_similarity=[]
    for i in range(0,len(cosine)):
        avg_content_similarity.append(round(mean(cosine[i]),3))

    df['Average_Content_Similarity']=avg_content_similarity

    return df[['Maximum_Content_Similarity','Average_Content_Similarity','Word_number_average']]
    
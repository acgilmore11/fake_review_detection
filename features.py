import pandas as pd
import numpy as np
import string
from nltk.tokenize import sent_tokenize



#rating_deviation feature:  the deviation of the evaluation provided in the review with respect to the entity’s average rating
#abs(product_rating - avg_product_rating)/4
def get_rating_deviation(table):
    avg_rating = table[['prod_id', 'rating']].groupby(['prod_id']).mean().rename(columns={'rating': 'avg_rating'})
    table = pd.merge(table, avg_rating, on='prod_id', how='inner')
    table['rating_deviation'] = abs(table['rating'] - table['avg_rating']) / 4
    return table['rating_deviation']

#singleton feature: 1 if review is the only review written that day by user, 0 otherwise
def get_singleton(table):
    date_counts = table.groupby(['user_id', 'date']).size().to_frame('size')
    table = pd.merge(table, date_counts, on=['user_id', 'date'], how='left')
    table['singleton'] = table['size'] == 1
    table['singleton'] = table['singleton'].astype('int')
    return table['singleton']

#textual features
def GenerateTextStastitics(table):
    """
    Text statistics: 
    Number of words, i.e., the length of the review in terms of words;
    Ratio of capital letters, i.e., the number of words containing capital letters with respect to the total number of words in the review;
    Ratio of capital words, i.e., considering the words where all the letters are uppercase;
    Ratio of first person pronouns,e.g.,‘I’,‘mine’,‘my’, etc.;
    Ratio of ‘exclamation’ sentences, i.e., ending with the symbol ‘!’. 
    """
    # table = pd.concat([meta_data, reviewContent["review"]], axis = 1).dropna()
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
    # table.drop("review", axis = 1)
    # table = pd.concat([table, text_statistics], axis = 1)
    # table.to_csv("text_statistic.csv", index=False)
    # return table

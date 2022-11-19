import pandas as pd
import numpy as np
import string
import nltk
from nltk.tokenize import sent_tokenize

#download punkt package
nltk.download('punkt')

# data_path = "YelpCSV"
# cols_meta = ["user_id", "prod_id", "rating", "label", "date"]
# meta_data = pd.read_csv(data_path+"/metadata.csv", names = cols_meta)
# cols_reviewContent = ["user_id", "prod_id", "date", "review"]
# reviewContent = pd.read_csv(data_path+"/reviewContent.csv", names = cols_reviewContent)




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
    table.drop("review", axis = 1)
    table = pd.concat([table, text_statistics], axis = 1)
    table.to_csv("text_statistic.csv", index=False)
    return table

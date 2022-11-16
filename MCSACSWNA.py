#Using TfidfVectorizer

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from math import sqrt, pow, exp
import numpy as np
from statistics import mean
from sklearn.metrics.pairwise import pairwise_distances
sentences=[]

def squared_sum(x):
  """ return 3 rounded square rooted value """ 
  return round(sqrt(sum([a*a for a in x])),3)

def cos_similarity(x,y):
  """ return cosine similarity between two lists """ 
  numerator = sum(a*b for a,b in zip(x,y))
  denominator = squared_sum(x)*squared_sum(y)
  return round(numerator/float(denominator),3)

df = pd.read_csv("reviewContent.csv")

df.columns = ['id', 'label', 'date', 'review']
df=df[['id','review']]

sentences = [sent.lower() for sent in df['review']]
processed_sentences = [re.sub('[^a-zA-Z]', ' ',sent) for sent in sentences]
processed_article = [re.sub(r'\s+', ' ', sent) for sent in processed_sentences]

df['Word_number_average'] = df['review'].str.split(" ").str.len()

'''
word_number_average=[]
for i in range(0,len(processed_article)):
    word_num_avg=len(processed_article[i].replace(" ", ""))/len(processed_article[i].split())
    word_number_average.append(round(word_num_avg,3))

df['Word_number_average']=word_number_average    

'''

tfidfvectorizer  = TfidfVectorizer(min_df=0,analyzer='word',stop_words= 'english')
tfidf_wm=tfidfvectorizer.fit_transform(processed_article)
tfidf_tokens = tfidfvectorizer.get_feature_names_out()
#df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),index = df['id'],columns = tfidf_tokens)
df_tfidfvect = tfidf_wm.toarray()
#print(df_tfidfvect)

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

df = df[['id','Maximum_Content_Similarity','Average_Content_Similarity','Word_number_average']]
df=df.groupby(['id']).mean()
print(df)



#Using CountVectorizer

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from math import sqrt, pow, exp
from statistics import mean
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

sentences=[]

def squared_sum(x):
  """ return 3 rounded square rooted value """ 
  return round(sqrt(sum([a*a for a in x])),3)

def cos_similarity(x,y):
  """ return cosine similarity between two lists """ 
  numerator = sum(a*b for a,b in zip(x,y))
  denominator = squared_sum(x)*squared_sum(y)
  return round(numerator/float(denominator),3)

df = pd.read_csv("reviewContent.csv")

df.columns = ['id', 'label', 'date', 'review']
df=df[['id','review']]

sentences = [sent.lower() for sent in df['review']]
processed_sentences = [re.sub('[^a-zA-Z]', ' ',sent) for sent in sentences]
processed_article = [re.sub(r'\s+', ' ', sent) for sent in processed_sentences]

df['Word_number_average'] = df['review'].str.split(" ").str.len()

'''
word_number_average=[]

for i in range(0,len(processed_article)):
    word_num_avg=len(processed_article[i].replace(" ", ""))/len(processed_article[i].split())
    word_number_average.append(round(word_num_avg,3))

df['Word_number_average']=word_number_average
'''
countvectorizer=CountVectorizer(min_df=0,analyzer= 'word',stop_words='english',ngram_range=(1, 2))
count_wm=countvectorizer.fit_transform(processed_article)
count_tokens=countvectorizer.get_feature_names_out()
#df_countvect = pd.DataFrame(data = count_wm.toarray(),index = df['id'],columns = count_tokens)
df_countvect = count_wm.toarray()

cosine=1-pairwise_distances(df_countvect, metric='cosine')
np.fill_diagonal(cosine,0)

max_content_similarity=[]
for i in range(0,len(cosine)):
    max_content_similarity.append(round(max(cosine[i]),3))

df['Maximum_Content_Similarity']=max_content_similarity    

avg_content_similarity=[]
for i in range(0,len(cosine)):
    avg_content_similarity.append(round(mean(cosine[i]),3))

df['Average_Content_Similarity']=avg_content_similarity

df = df[['id','Maximum_Content_Similarity','Average_Content_Similarity','Word_number_average']]
df=df.groupby(['id']).mean()
print(df)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 10:48:40 2022

@author: rrgoyal
"""

#%% Libraries
import pandas as pd
import json
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import re
from copy import deepcopy
from collections import Counter
from flair.models import TextClassifier
from flair.data import Sentence
import pickle

#%% Global Variables

classifier = TextClassifier.load('en-sentiment')
stopwords = set(STOPWORDS)


#%% Extracting the data out of the dataset

json_data = []
for line in open(r"/home/rrgoyal/Ram/Sem_III/Research Methods/Implementation/train_pair_data.jsonlist",'r'):
    json_data.append(json.loads(line))


#%% Creating a dataframe from the dataset

df = pd.DataFrame().from_records(json_data)


#%% Extracting the positive and the negative comments

positiveComments=[]
for comment_box in [comms['comments'] for comms in df['positive']]:
    for comment in comment_box:
        positiveComments.append(comment['body'])


negativeComments=[]
for comment_box in [comms['comments'] for comms in df['negative']]:
    for comment in comment_box:
        negativeComments.append(comment['body'])


#%% Sentimanting on positive Comments

positive_comments_copy = deepcopy(positiveComments)
for i in range(len(positive_comments_copy)):
    re.sub("\[", "", positive_comments_copy[i])
    positive_comments_copy[i] = Sentence(positive_comments_copy[i])
    classifier.predict(positive_comments_copy[i])
    
#%% Sentimanting on Negative Comments

negative_comments_copy = deepcopy(negativeComments)
for i in range(len(negative_comments_copy)):
    negative_comments_copy[i] = Sentence(negative_comments_copy[i])
    classifier.predict(negative_comments_copy[i])


#%% Exporting the objects

positiveFile = open("Positive_Comments_Sentiments.obj",'wb')
negativeFile = open("Negative_Comments_Sentiments.obj",'wb')

pickle.dump(positive_comments_copy,positiveFile)
pickle.dump(negative_comments_copy,negativeFile)


#%% Extracting the labels out of Sentence Object

positives = [x.labels[0].value for x in positive_comments_copy]
negatives = [x.labels[0].value for x in negative_comments_copy]

neutrality_counts_delta = Counter(positives)
neutrality_counts_negatives = Counter(negatives)


#%% Plotting the composition of different semantics in the positive and the negative comments

fig,ax = plt.subplots(2,1,figsize=(5,5))

ax[0].pie(neutrality_counts_delta.values(),autopct="  %1.1f%%")
ax[0].legend(neutrality_counts_delta.keys())
ax[0].set_title("Delta Comments Composition")

ax[1].pie(neutrality_counts_negatives.values(),autopct="  %1.1f%%")
ax[1].legend(neutrality_counts_negatives.keys())
ax[1].set_title("Negative Comments Composition")















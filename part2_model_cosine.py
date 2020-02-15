#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import re
import string
import nltk
import pandas as pd


stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

print("Please enter the email to find whether it is an Actionable or not")
email=input('Email: ')
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    sen=' '.join(text)
    return sen

email=clean_text(email)


pickle_vec_off = open ("tfidf_vec.txt", "rb")
dd=pickle.load(pickle_vec_off)
Action_df=pd.DataFrame(dd.toarray())
pickle_frame_off = open ("tfidf_frame.txt", "rb")
tfidf_frame=pickle.load(pickle_frame_off)

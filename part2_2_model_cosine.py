#!/usr/bin/env python
# coding: utf-8

# In[ ]:

#importing necessary libraries
import pickle
import re
import string
import nltk
import pandas as pd


stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

#getting the email from CLI
print("Please enter the email to find whether it is an Actionable or not")
email=input('Email: ')
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    sen=' '.join(text)
    return sen

email=clean_text(email)

#loaing the tfidf vector pickeled models
pickle_vec_off = open ("tfidf_vec.txt", "rb")
dd=pickle.load(pickle_vec_off)
Action_df=pd.DataFrame(dd.toarray())
pickle_frame_off = open ("tfidf_frame.txt", "rb")
tfidf_frame=pickle.load(pickle_frame_off)


#compute cosine similarity
def cosin(para):
    #print(para)
    y=[]
    for x in range(Action_df.shape[0]+1):
        try:
            
            dataSetI = Action_df.iloc[x].values
            #print(type(dataSetI),'11 values if dataset 1 is ',dataSetI,dataSetI.shape)
            dataSetII = tfidf_frame.transform([para]).toarray()[0]
            #print(type(dataSetII),'12values if dataset 1 is ',dataSetII, dataSetII.shape)
            result =spatial.distance.cosine(list(dataSetI), list(dataSetII))
            #print('Result',result)
            if result == 1:
                return 'Action'
            
        except:
            pass
    return 'Dummy'


print('%%%The state of the entered email is%%% :{:*^20}'.format(cosin(email)))


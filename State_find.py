#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial


# ## Reading the data actionalble sentences 



df=pd.read_csv('actions.csv')
df.columns=['Dialogue']


# ## cleaning the Actionable dialogues

lemmatizer=WordNetLemmatizer()

def cleaning(sentences):
    words = []
    clean = re.sub(r'[^ a-z A-Z 0-9]', " ", sentences)
    w = word_tokenize(clean)
    #lemmatizing
    words.append([lemmatizer.lemmatize(i.lower()) for i in w])
    return words

def sen(para):
  return ' '.join(para)
  
df['tocken']=df['Dialogue'].apply(lambda x : cleaning(x))
df['tocken']=df['tocken'].apply(lambda x :x[0])
df['clean']=df['tocken'].apply(lambda x : sen(x))


# ## loading the email data set

import pandas as pd
import numpy as np
import re
import string
import nltk
emails=pd.read_csv('emails.csv',nrows=100)


def parse_into_emails(messages):
    emails = [parse_raw_message(message) for message in messages]
    return {
        'body': map_to_list(emails, 'body'),
        'to': map_to_list(emails, 'to'),
        'from_': map_to_list(emails, 'from')
    }

# cleaning 
def parse_raw_message(raw_message):
    lines = raw_message.split('\n')
    email = {}
    message = ''
    keys_to_extract = ['from', 'to']
    for line in lines:
        if ':' not in line:
            message += line.strip()
            email['body'] = message
        else:
            pairs = line.split(':')
            key = pairs[0].lower()
            val = pairs[1].strip()
            if key in keys_to_extract:
                email[key] = val
    return email

def map_to_list(emails, key):
    results = []
    for email in emails:
        if key not in email:
            results.append('')
        else:
            results.append(email[key])
    return results
    
email_df = pd.DataFrame(parse_into_emails(emails.message))
email_data=pd.concat([email_df['body']],axis=1)




#data cleaning
stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text

email_data['clean_body']=email_data['body'].apply(lambda x : clean_text(x))
email_data['clean_senntence']=email_data['clean_body'].apply(lambda x : sen(x))

'''a=email_data['body'][1]
print(a.split())
v=tfidf.transform([a])
list(v.toarray()[0])'''


# ## preparing the TF-IDF vector



tfidf = TfidfVectorizer()
dd=tfidf.fit_transform(df['clsen'])
Action_df=pd.DataFrame(dd.toarray())


# ## Cosine similarity comparision

# this method will append a 'Action' tag for a email if it's cosine similarity with the Actionalable sectence maches 1.

def cosin(para):
    #print(para)
    y=[]
    for x in range(Action_df.shape[0]+1):
        try:
            
            dataSetI = Action_df.iloc[x].values
            #print(type(dataSetI),'11 values if dataset 1 is ',dataSetI,dataSetI.shape)
            dataSetII = tfidf.transform([para]).toarray()[0]
            #print(type(dataSetII),'12values if dataset 1 is ',dataSetII, dataSetII.shape)
            result =spatial.distance.cosine(list(dataSetI), list(dataSetII))
            #print('Result',result)
            if result == 1:
                return 'Action'
            
        except:
            pass
    return 'Dummy'


email_data['state']=email_data['body'].apply(lambda x : cosin(x))
email_data.to_csv('Result.csv')


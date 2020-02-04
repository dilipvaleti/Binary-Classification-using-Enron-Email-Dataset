# importing necessary libraries
import pandas as pd
import numpy as np
import re
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, accuracy_score

# loading the date into data frame from the classified email data set which was generated using main,py module
emails = pd.read_csv("email_classify.csv")

# seggregate the dataset
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
email_data=pd.concat([email_df['body'],emails['State']],axis=1)

#data cleaning
stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text

email_data['clean_body']=email_data['body'].apply(lambda x : clean_text(x))
email_data

# Intializing tfidf vector
tfidf_vect = TfidfVectorizer(analyzer=clean_text)
tfidf_vect_fit = tfidf_vect.fit_transform(email_data['clean_body'])
tfidf_vect_fit


#split the data into train and test data set by 4:1 ratio
X_train, X_test, y_train, y_test = train_test_split(tfidf_vect_fit.toarray(), email_data['State'], test_size=0.2)

#applying the Gaussian model
gnb=GaussianNB()
y_pred2 = gnb.fit(X_train, y_train).predict(X_test) 

#Metrics calculation
print("Accuracy score :", accuracy_score(y_test,y_pred2))
print("Confussion matrix :\n",confusion_matrix(y_test,y_pred2))
print("Classification report :\n", classification_report(y_test,y_pred2))


# Binary-Classification-using-Enron-Email-Dataset
we are going to prepare a binary classification model using Enron Email Dataset.

# Enron Email Dataset

Kaggle Dataset Containing Emails and other meta-information
[Kaggle Link]https://www.kaggle.com/wcukierski/enron-email-dataset/download

## Functionalities (Part 1)

Aim to Create a heuristics-based linguistic model for detecting actionable items from the email. A rule-based model to classify sentences to actionable sentence and non-actionable sentence

## Functionalities (Part 2)

Train a model to detect whether a given sentence is an actionable item or not. 

```
Actionable item => A sentence which asks someone to do something
example: "Please create an assignment and forward it by EOD"
```

### Directory Structure and Important Files

```
.email.csv : Email Data set is not uploading on github. You can download directly from Kaggle Link provided above.
./outputs/ : Save the output in this folder
./email_classify.csv : save a new CSV file having a classified email data set
main.py : Python Code to find out the list of actionable sentences and classify the emails under Action/Dummy classes.
```

##    1) Explain your project pipeline.
      By using main.py module we are going to findout the actionalbe sentences in the given enron data set. Export teh result data set  into one excel file which will be used as a input data set for model preparation model.py
    2) Explain in detail the process of feature extraction.  
      Once the dataset ready with two clases (Action, Dummy) which was generated from main.py module, we just clean teh data by removing the punctuatuions, stopwords, tokenization, stemming and finnaly each email was converted to a verctor by using TF-IDF. 
    3) Report Recall, precision, and F1 measure
    Classification report :
               precision    recall  f1-score   

      Action       0.86      0.79      0.82       
       Dummy       0.81      0.89      0.85      


    4) Explain the challenges you've faced while building the model.
I faced much more challenges while finding the actionable sentences in the email data set by using heuristics-based linguistic model, i.e identifying verb patterns for sentence maching and pos tagging etc. 

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

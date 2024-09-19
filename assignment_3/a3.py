import numpy as np
import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB

# takes a directory of files, and a label, and returns a
# list of 2-tuples of each file's contents, and the given label
def get_email_content(dir, label):
    files = []
    for filename in os.listdir(dir):
        path = os.path.join(dir, filename)
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    files += [(clean_email(f.read()), label)]
            except:
                pass
    return files

# if we want to clean the emails we can modify this function
# however, our proposed way of removing the headers actually
# decreased the accuracy of the models
def clean_email(content):
    # remove_header = '\\n\\n'
    # email_parts = re.split(remove_header, content)
    # if (len(email_parts) == 1):
    #     return email_parts[0]
    # else:
    #     return email_parts[1]
    return content

def get_confusion_values(y_test_inv, y_pred_inv, label):
    tp = ((y_test_inv == label) & (y_pred_inv == label)).sum()
    fp = ((y_test_inv != label) & (y_pred_inv == label)).sum()
    fn = ((y_test_inv == label) & (y_pred_inv != label)).sum()
    tn = ((y_test_inv != label) & (y_pred_inv != label)).sum()
    return tp, fp, fn, tn

def get_apr_values(tp, fp, fn, tn):
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return accuracy, precision, recall

SEED = 12345678
easy_ham_dir = 'assignment_3/easy_ham'
hard_ham_dir = 'assignment_3/hard_ham'
spam_dir = 'assignment_3/spam'

easy_ham_files = get_email_content(easy_ham_dir, "Ham")
hard_ham_files = get_email_content(hard_ham_dir, "Ham")
spam_files = get_email_content(spam_dir, "Spam")

# to switch between the easy_ham and hard_ham dataset,
# swap out either easy_ham_files or hard_ham_files for the
# other here
records = hard_ham_files + spam_files

df = pd.DataFrame.from_records(records, columns=["Email", "Type"])
df_train, df_test = train_test_split(df, random_state=SEED, test_size=0.25)

cv = CountVectorizer()
x_train = cv.fit_transform(df_train["Email"])
x_test = cv.transform(df_test["Email"])

le = LabelEncoder()
y_train = le.fit_transform(df_train["Type"])
y_test = le.transform(df_test["Type"])
y_test_inv = le.inverse_transform(y_test)

bnb = BernoulliNB()
bnb.fit(x_train, y_train)
y_pred_bnb = bnb.predict(x_test)
y_pred_bnb_inv = le.inverse_transform(y_pred_bnb)
acc_bnb, pre_bnb, rec_bnb = get_apr_values(*get_confusion_values(y_test_inv, y_pred_bnb_inv, "Ham"))

mnb = MultinomialNB()
mnb.fit(x_train, y_train)
y_pred_mnb = mnb.predict(x_test)
y_pred_mnb_inv = le.inverse_transform(y_pred_mnb)
acc_mnb, pre_mnb, rec_mnb = get_apr_values(*get_confusion_values(y_test_inv, y_pred_mnb_inv, "Ham"))

print(f'BNB: {get_confusion_values(y_test_inv, y_pred_bnb_inv, "Ham")}')
print(f'MNB: {get_confusion_values(y_test_inv, y_pred_mnb_inv, "Ham")}')
print(f'BNB: {acc_bnb}, {pre_bnb}, {rec_bnb}')
print(f'MNB: {acc_mnb}, {pre_mnb}, {rec_mnb}')

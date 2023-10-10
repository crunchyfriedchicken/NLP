#--------------------import packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
#nltk.download_shell()

#--------------------open and make file into a df
[line for line in open('data/SMSSpamCollection')][3]
#separated by \t
messages = pd.read_csv('data/SMSSpamCollection',sep="\t",names=["label","message"])
messages

#make a new column file length
messages["length"] = messages["message"].apply(len)
messages

#--------------------exploratory data analysis and data visualisation
messages.describe()
#there is a message that is 910 characters long
messages.groupby(["label"]).describe()

sns.set_style("whitegrid")
sns.histplot(messages["length"], bins=100)
plt.show()

#--------------------test data split
from sklearn.model_selection import train_test_split
X = messages["message"]
y = messages["label"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=101)

#--------------------create a function to preprocess the data [remove punctuation and stopwords]
import string
string.punctuation
stopwords.words("english")

#try with test message
testmess = messages["message"][3]
#remove punctuation
letters = [char for char in testmess if char not in string.punctuation]
#rejoin the characters and then split the words into a list of words
nopunc = ''.join(letters).split()
#remove stopwords and also change to lowercase
finalmess = [word.lower() for word in nopunc if word.lower() not in stopwords.words("english")]

#create function to apply to each document in a corpus
def token(document):
    letters = [char for char in document if char not in string.punctuation]
    nopunc = ''.join(letters).split()
    finalmess = [word.lower() for word in nopunc if word.lower() not in stopwords.words("english")]
    return finalmess

#--------------------create a pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

pipeline = Pipeline([
    ('bag of words', CountVectorizer(analyzer=token)),
    ('tfidf', TfidfTransformer()),
    ('classification model', MultinomialNB())
])

#--------------------create a pipeline
pipeline.fit(X_train,y_train)
predictions = pipeline.predict(X_test)

#--------------------classification report
from sklearn.metrics import classification_report
print(classification_report(predictions,y_test))



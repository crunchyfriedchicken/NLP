#--------------------import packages
import pandas as pd
from nltk.corpus import stopwords

#--------------------open and make file into a df
[line for line in open('data/SMSSpamCollection')][3]
#separated by \t
messages = pd.read_csv('data/SMSSpamCollection',sep="\t",names=["label","message"])
messages

#make a new column file length
messages["length"] = messages["message"].apply(len)
messages

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
def text_processor(document):
    letters = [char for char in document if char not in string.punctuation]
    nopunc = ''.join(letters).split()
    finalmess = [word.lower() for word in nopunc if word.lower() not in stopwords.words("english")]
    return finalmess

#--------------------create bow from CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#create instance and apply the text processing
vectorizer = CountVectorizer(analyzer=text_processor)

#fit on X_train and transform for X_train and X_test
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

#--------------------apply TFIDF transformation
from sklearn.feature_extraction.text import TfidfTransformer

#create instance (note: normalizing is defaulr)
tfidf_transformer = TfidfTransformer()

#fit on X_train and transform for X_train and X_test
X_train_tfidf = tfidf_transformer.fit_transform(X_train_bow)
X_test_tfidf = tfidf_transformer.transform(X_test_bow)

#--------------------fit onto model and predict

from sklearn.naive_bayes import MultinomialNB

multi = MultinomialNB()
multi.fit(X_train_tfidf, y_train)
predictions = multi.predict(X_test_tfidf)

#--------------------classification report & confusion matrix
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test,predictions))
print("\n")
print(classification_report(y_test, predictions))

#--------------------import packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
nltk.download_shell()

#--------------------open and make file into a df
[line for line in open('smsspamcollection/SMSSpamCollection')][3]
#separated by \t
messages = pd.read_csv('smsspamcollection/SMSSpamCollection',sep="\t",names=["label","message"])
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
X_train, X_test, y_train, y_test = train_test_split(messages["message"],messages["label"],test_size=0.2)

#--------------------create a function to preprocess the data [remove punctuation and stopwords]
import string
string.punctuation

stopwords.words("english")
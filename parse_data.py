import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from os import listdir
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn import neighbors, datasets
from sklearn import svm
from sklearn.model_selection import train_test_split


df = pd.read_csv('./final_result.csv')


texts = df['TEXT']
types = df['TYPE']


#print types

def transform_row(row):
    #print (row)
    row = re.sub(r"^[0-9\.]+", "", str(row))
    row = re.sub(r"[\.,\?]+$", "", row)
    row = row.replace(",", " ").replace(".", " ").replace(";", " ").replace('"', " ").replace(":", " ").replace('"', "").replace("!", "").replace("?", "").replace("(", "").replace(")", "").replace("\n", " ").replace("\/","").replace("\\","").replace("\[","").replace("\]","").replace("&", "").replace("-","").replace('*','').replace('/','').replace('#','').replace('>','').replace('<','')
    row = row.strip()
    return row
texts = texts.apply(transform_row)
count_vect = TfidfVectorizer()
count_vect.fit(texts)
clf = joblib.load('./lnl.pkl')


its = [ './it/' + i for i in listdir('./it')]
noits = ['./non-it/'+ i for i in listdir('./non-it')]
x = []
y = []
print ('it')
for i in its:
    print(i)
    person = pd.read_csv(i)
    FOLLOWER =float(person['FOLLOWER'][0])
    if FOLLOWER > 1000.0:
        FOLLOWER = 1000.0
    LIKE =float(person['LIKE'][0])
    if LIKE > 1000.0:
        LIKE = 1000.0
    TOTAL_LIKE = float(person['TOTAL_LIKE'][0])
    if TOTAL_LIKE > 1000.0:
        TOTAL_LIKE = 1000.0
    TOTAL_RETWEET = float(person['TOTAL RETWEET'][0])
    if TOTAL_RETWEET > 1000.0:
        TOTAL_RETWEET = 1000.0
    TWEET =person['TWEET']
    TWEET = [str(k) for k in TWEET]
    vec = count_vect.transform(TWEET)
    result = clf.predict(vec)
    count_it =float(len([i for i in result if i == 'it']))
    x.append([FOLLOWER/1000.0, LIKE/1000.0, TOTAL_LIKE/1000.0, TOTAL_RETWEET/1000.0, count_it/200.0])
    y.append('it')
    print(FOLLOWER, LIKE, TOTAL_LIKE, TOTAL_RETWEET, count_it)

print('non-it')
for i in noits:
    person = pd.read_csv(i)
    print(i)
    FOLLOWER =float(person['FOLLOWER'][0])
    if FOLLOWER > 1000.0:
        FOLLOWER = 1000.0
    LIKE =float(person['LIKE'][0])
    if LIKE > 1000.0:
        LIKE = 1000.0
    TOTAL_LIKE = float(person['TOTAL_LIKE'][0])
    if TOTAL_LIKE > 1000.0:
        TOTAL_LIKE = 1000.0
    TOTAL_RETWEET = float(person['TOTAL RETWEET'][0])
    if TOTAL_RETWEET > 1000.0:
        TOTAL_RETWEET = 1000.0
    TWEET =person['TWEET']
    TWEET = [str(k) for k in TWEET]
    vec = count_vect.transform(TWEET)
    result = clf.predict(vec)
    count_it =float(len([i for i in result if i == 'it']))
    x.append([FOLLOWER/1000.0, LIKE/1000.0, TOTAL_LIKE/1000.0, TOTAL_RETWEET/1000.0, count_it/200.0])
    y.append('non-it')
    print(FOLLOWER, LIKE, TOTAL_LIKE, TOTAL_RETWEET, count_it)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

def predict_lnl(x,y,x_test,y_test):
    logreg = linear_model.LogisticRegression(C=1.0)
    logreg.fit(x, y)
    joblib.dump(logreg, './test.pkl')
    result = logreg.predict(x_test)
    print(result)
    print ( accuracy_score(y_test, result))
def predict_with_knn(x,y,x_test, y_test ):
    knn = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights = 'distance')
    knn.fit(x, y)
    #joblib.dump(knn, './test.pkl')
    result = knn.predict(x_test)
    print(result)
    print ( accuracy_score(y_test, result))
def predict_svm(x,y,x_test,y_test):
    linear_svc = svm.SVC(kernel='linear')
    linear_svc.fit(x, y)
    result = linear_svc.predict(x_test)
#    joblib.dump(linear_svc, './test.pkl')
    print(result)
    print (accuracy_score(y_test, result))
predict_lnl(X_train,y_train,X_test,y_test)
predict_with_knn(X_train,y_train,X_test,y_test)
#predict_svm(x,y,x,y)

person_predict = joblib.load('./test.pkl')

def predict(file):
    person = pd.read_csv(file)
    FOLLOWER =float(person['FOLLOWER'][0])
    _1 = FOLLOWER
    if FOLLOWER > 1000.0:
        FOLLOWER = 1000.0
    LIKE =float(person['LIKE'][0])
    _2 = LIKE
    if LIKE > 1000.0:
        LIKE = 1000.0
    TOTAL_LIKE = float(person['TOTAL_LIKE'][0])
    _3 = TOTAL_LIKE
    if TOTAL_LIKE > 1000.0:
        TOTAL_LIKE = 1000.0
    TOTAL_RETWEET = float(person['TOTAL RETWEET'][0])
    _4 = TOTAL_RETWEET
    if TOTAL_RETWEET > 1000.0:
        TOTAL_RETWEET = 1000.0
    TWEET =person['TWEET']
    TWEET = [str(k) for k in TWEET]
    vec = count_vect.transform(TWEET)
    result = clf.predict(vec)
    count_it =float(len([i for i in result if i == 'it']))
    print(FOLLOWER, LIKE, TOTAL_LIKE, TOTAL_RETWEET, count_it)
    return('https://twitter.com/'+person['NAME'][0],_1, _2, _3, _4, count_it, person_predict.predict([[FOLLOWER/1000.0, LIKE/1000.0, TOTAL_LIKE/1000.0, TOTAL_RETWEET/1000.0, count_it/200.0]]))
person = pd.read_csv('./User_Info_Rework/arxivblog.csv')
FOLLOWER = person['FOLLOWER'][0]
LIKE = person['LIKE'][0]
TOTAL_LIKE = person['TOTAL_LIKE'][0]
TOTAL_RETWEET = person['TOTAL RETWEET'][0]
TWEET = person['TWEET']
vec = count_vect.transform(TWEET)
result = clf.predict(vec)
count_it = len([i for i in result if i == 'it'])

print(FOLLOWER, LIKE, TOTAL_LIKE, TOTAL_RETWEET, count_it)
print(person_predict.predict([[FOLLOWER, LIKE, TOTAL_LIKE, TOTAL_RETWEET, count_it]]))
#vec = count_vect.transform(datadf['TWEET'])
#result = clf.predict(vec)

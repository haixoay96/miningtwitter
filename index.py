import pandas as pd
import numpy as np
import re
#import spacy

from sklearn import neighbors, datasets
from sklearn import linear_model
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
#nlp = spacy.load('en')
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

OPTIMIZER = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)


df = pd.read_csv('./final_result.csv')


texts = df['TEXT']
types = df['TYPE']
types_ = []
VALIDATION_SPLIT = 0.2
VERBOSE = 1
BATCH_SIZE = 128
NB_EPOCH = 5000



#print types

def transform_row(row):
    #print (row)
    row = re.sub(r"^[0-9\.]+", "", str(row))
    row = re.sub(r"[\.,\?]+$", "", row)
    row = row.replace(",", " ").replace(".", " ").replace(";", " ").replace('"', " ").replace(":", " ").replace('"', "").replace("!", "").replace("?", "").replace("(", "").replace(")", "").replace("\n", " ").replace("\/","").replace("\\","").replace("\[","").replace("\]","").replace("&", "").replace("-","").replace('*','').replace('/','').replace('#','').replace('>','').replace('<','')
    row = row.strip()
    return row
texts = texts.apply(transform_row)
print(len(texts))
vecs = []
count_vect = TfidfVectorizer()
count_vect.fit(texts)
#print(count_vect.vocabulary_)
for i,txt in enumerate(texts):
    vec = count_vect.transform([txt])
    #print(vec.shape)
    #print(type(vec))
    #print(vec.toarray())
    print(txt)
    print(i)
    print (types[i])
    if types[i] == 'it':
        types_.append(1)
    else:
        types_.append(0)
    if types[i] != 'it' and types[i] != 'non-it':
        print( 'error type')
        print (type(types[i]))
        break
    # vec = nlp(txt).vector
    # if len(vec) != 384:
    #     print('error')
    #     print(i)
    #     print(txt)
    #     break
    vecs.append(vec.toarray()[0])
    print(len(vec.toarray()[0]))

X_train, X_test, y_train, y_test = train_test_split(vecs, types_, test_size=0.2, random_state=42)
def predict_with_knn(x,y,x_test, y_test ):
    clf = neighbors.KNeighborsClassifier(n_neighbors = 1000, p = 2, weights = 'distance')
    clf.fit(x, y)
    result = clf.predict(x_test)
    joblib.dump(clf, './knn.pkl')
    print(result)
    print ( accuracy_score(y_test, result))
def predict_lnl(x,y,x_test,y_test):
    logreg = linear_model.LogisticRegression(C=1.0)
    logreg.fit(x, y)
    result = logreg.predict(x_test)
    joblib.dump(logreg, './lnl.pkl')
    print(result)
    print ( accuracy_score(y_test, result))
def predict_svm(x,y,x_test,y_test):
    linear_svc = svm.SVC(kernel='linear')
    linear_svc.fit(x, y)
    result = linear_svc.predict(x_test)
    joblib.dump(linear_svc, './svm.pkl')
    print(result)
    print ( accuracy_score(y_test, result))
def predict_nb(x,y,x_test,y_test):
    gnb = MultinomialNB()
    gnb.fit(x, y)
    result = gnb.predict(x_test)
    joblib.dump(gnb, './nb.pkl')
    print(result)
    print ( accuracy_score(y_test, result))
def neuralnetwork(x,y,x_test,y_test):
    model = Sequential()
    model.add(Dense(2,  input_dim=16449))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
    history = model.fit(x, y, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
    score = model.evaluate(x_test, y_test, verbose=VERBOSE)
    model.save('./neural3.h5')
    print('Test score:', score[0])
    print('Test accuracy', score[1])



y_train_ = ['noit' for i in y_train if i == 'non-it']
y_test_ = ['noit' for i in y_test if i == 'non-it']
neuralnetwork(np.array(X_train), np_utils.to_categorical(y_train, 2),np.array(X_test),np_utils.to_categorical(y_test,2))



#predict_lnl(X_train,y_train,X_test,y_test)
#predict_nb(X_train,y_train,X_test,y_test)
#predict_svm(X_train, y_train,X_test,y_test)
#predict_with_knn(X_train,y_train, X_test, y_test)

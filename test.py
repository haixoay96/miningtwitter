from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy

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
#nlp = spacy.load('en')

df = pd.read_csv('./final_result.csv')


texts = df['TEXT'][0:100]
types = df['TYPE'][0:100]


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
targets = []
count_vect = TfidfVectorizer()
count_vect.fit(texts)
#print(count_vect.vocabulary_)
for i,txt in enumerate(texts):
    vec = count_vect.transform([txt])
    #print(vec.shape)
    #print(type(vec))
    print('vec')
    print(vec.toarray()[0])
#    print(txt)
    print(i)
#    print (types[i])
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
    if types[i] == 'it':
        targets.append([1,0])
    else:
        targets.append([0,1])
    #print(vec.toarray()[0])
    #print( type(vec.toarray()[0]))

    print(len(vec.toarray()[0]))

X_train, X_test, y_train, y_test = train_test_split(vecs, targets, test_size=0.2, random_state=42)
# fix random seed for reproducibility
numpy.random.seed(7)

#print X_train

model = Sequential()
model.add(Dense(1000,input_dim=769, activation='sigmoid'))
model.add(Dense(500, activation='sigmoid'))
model.add(Dense(250, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(25, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(2, activation='sigmoid'))
model.add(Dense(2, activation='sigmoid'))
model.add(Activation('softmax'))
print y_train
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(np.array(X_train),np.array(y_train), epochs=1000000, batch_size=100,verbose=2)
model.save('./test.h5')
# evaluate the model
scores = model.evaluate(np.array(X_train), np.array(y_train))
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

scores = model.evaluate(np.array(X_test),np.array(y_test))
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
result = model.predict(np.array(X_test))

import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.models import model_from_json
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.isnull().sum()

train.columns

classes = ['toxic','severe_toxic','obscene', 'threat',
       'insult', 'identity_hate']

y = train[classes].values
sentences_train = train["comment_text"]
sentences_test = test["comment_text"]  

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
vector = cv.fit(list_sentences_train)

len(cv.get_feature_names())

#Total number of unique words are 189775
 
max_features = 20000
tokenizer = Tokenizer(num_words=max_features,lower=True)
tokenizer.fit_on_texts(list(sentences_train))
tokenized_train = tokenizer.texts_to_sequences(sentences_train)
tokeized_test = tokenizer.texts_to_sequences(sentences_test)
 
maxlen = 200
X_train = pad_sequences(list_tokenized_train,maxlen=maxlen)
X_test = pad_sequences(list_tokeized_test,maxlen=maxlen)
 
inp = Input(shape=(maxlen,))
embed_size = 128
x = Embedding(max_features,embed_size)(inp)
x = LSTM(60,return_sequences=True,name='lstm_layer')(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)

x = Dense(50,activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6,activation="sigmoid")(x)

model = Model(inputs=inp,outputs=x)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

batch_size = 32
epochs = 2
model.fit(X_train,y,batch_size=batch_size,epochs=epochs,validation_split=0.1)

model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")

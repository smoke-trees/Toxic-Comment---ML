import argparse
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="True"

ap = argparse.ArgumentParser()
ap.add_argument("-m","--message",required=True,help="Message to be tested")
args = vars(ap.parse_args())

train = pd.read_csv('train.csv')
sentences_train = train["comment_text"]
max_features = 20000
maxlen = 200
tokenizer = Tokenizer(num_words=max_features,lower=True)
tokenizer.fit_on_texts(list(sentences_train))


json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

message = args["message"]
message = message.split(',')


columns1 = ['toxic','severe_toxic','obscene', 'threat',
       'insult', 'identity_hate']


list_sentences = message
to_be_tested = tokenizer.texts_to_sequences(list_sentences)
to_be_tested = pad_sequences(to_be_tested,maxlen=maxlen)


predict = loaded_model.predict(to_be_tested)
print(predict)

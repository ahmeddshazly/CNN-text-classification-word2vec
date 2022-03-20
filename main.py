from __future__ import division, print_function

import pip
from gensim import models
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Conv1D, GlobalMaxPooling1D, Embedding ,LSTM ,MaxPooling1D
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from numpy.distutils.system_info import conda
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import collections
import re
import string
import matplotlib.pyplot as plt


data = pd.read_csv('imdb.csv', header = None )
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values


data.columns = ['review','sentiment']




print(data.head())



print(data.shape)



Y = data['sentiment']
Y = np.array(list(map(lambda x: 1 if x=="positive" else 0, Y)))
print(Y)
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 50001):
  review = re.sub('[^a-zA-Z]', ' ', data['review'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  #review = ''.join([i for i in review if not i.isdigit()])
  corpus.append(review)
data['review'] = corpus

words = corpus
print(len(words))
print(words[1])
from keras.preprocessing.text import Tokenizer
X=X.flatten()
num_words = 20000
tokenizer = Tokenizer(num_words)
tokenizer.fit_on_texts(X)



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)


X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

maxlen = 256
X_train = pad_sequences(X_train, maxlen )
X_test = pad_sequences(X_test, maxlen )

vocab_size = len(tokenizer.word_index) + 1
wordindex = tokenizer.word_index

print(vocab_size)


word2vec_path = 'GoogleNews-vectors-negative300.bin.gz'
word2vec = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)



embedding_dim =300
counter=0
embedding_matrix2 = np.zeros((vocab_size, embedding_dim))
for word, i in wordindex.items():

    try:
        embedding_vector2 = word2vec[word]
        embedding_matrix2[i] = embedding_vector2
        counter=counter+1
    except KeyError:
        embedding_matrix2[i]=np.random.normal(0,np.sqrt(0.25),embedding_dim)

print(counter)

print(embedding_matrix2[20001])
print(counter)
print(len(embedding_matrix2))

def ConvNet(embedding_matrix2, maxlen , vocab_size , embedding_dim):
    embedding_layer = Embedding(vocab_size,
                                 embedding_dim,
                                 weights= [embedding_matrix2],
                                 input_length=maxlen,
                                 trainable= False)

    sequence_input = Input(shape=(maxlen,), dtype='int32')

    embedded_sequences = embedding_layer(sequence_input)

    #convs = []

    l_conv = Conv1D(filters=100, kernel_size=3, activation='relu')(embedded_sequences)
    #l_pool = GlobalMaxPooling1D()(l_conv)
    #convs.append(l_pool)
    #l_conv2 = Conv1D(filters=10, kernel_size=3, activation='relu')(l_conv)
    l_pool = GlobalMaxPooling1D()(l_conv)
    #convs.append(l_pool)
   # l_conv = Conv1D(filters=64, kernel_size=3, activation='relu')(embedded_sequences)
   # l_pool = GlobalMaxPooling1D()(l_conv)
   # convs.append(l_pool)


   # l_merge = concatenate(convs, axis=1)

   # x = Dropout(0.5)
    x = Dense(512, activation='relu')(l_pool)
    x = Dropout(0.2)(x)
    preds = Dense(units=1,activation='sigmoid')(x)
    model = Model(sequence_input, preds)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    return model


def convnet2 (embedding_matrix2, maxlen , vocab_size, embedding_dim):
    embedding_layer = Embedding(vocab_size,
                                 embedding_dim,
                                 weights= [embedding_matrix2],
                                 input_length=maxlen,
                                 trainable= False)

    sequence_input = Input(shape=(maxlen,), dtype='int32')

    embedded_sequences = embedding_layer(sequence_input)



    l_conv = Conv1D(filters=128, kernel_size=3, activation='relu')(embedded_sequences)

    l_pool = GlobalMaxPooling1D()(l_conv)

    x = Dense(512, activation='relu')(l_pool)
    x = Dropout(0.2)(x)
    preds = Dense(units=1,activation='sigmoid')(x)
    model = Model(sequence_input, preds)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    return model







def convnet3 (embedding_matrix2, maxlen , vocab_size, embedding_dim):
    embedding_layer = Embedding(vocab_size,
                                 embedding_dim,
                                 weights= [embedding_matrix2],
                                 input_length=maxlen,
                                 trainable= False)

    sequence_input = Input(shape=(maxlen,), dtype='int32')

    embedded_sequences = embedding_layer(sequence_input)



    l_conv = Conv1D(filters=256, kernel_size=3, activation='relu')(embedded_sequences)


    l_pool = GlobalMaxPooling1D()(l_conv)








    x = Dense(512, activation='relu')(l_pool)
    x = Dropout(0.2)(x)
    preds = Dense(units=1,activation='sigmoid')(x)
    model = Model(sequence_input, preds)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    return model





def convnet4 (embedding_matrix2, maxlen , vocab_size, embedding_dim):
    embedding_layer = Embedding(vocab_size,
                                 embedding_dim,
                                 weights= [embedding_matrix2],
                                 input_length=maxlen,
                                 trainable= False)

    sequence_input = Input(shape=(maxlen,), dtype='int32')

    embedded_sequences = embedding_layer(sequence_input)



    l_conv = Conv1D(filters=10, kernel_size=3, activation='relu')(embedded_sequences)

    l_conv2 = Conv1D(filters=10, kernel_size=3, activation='relu')(l_conv)
    l_pool = GlobalMaxPooling1D()(l_conv2)




    x = Dense(512, activation='relu')(l_pool)
    x = Dropout(0.2)(x)
    preds = Dense(units=1,activation='sigmoid')(x)
    model = Model(sequence_input, preds)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    return model




def convnet5 (embedding_matrix2, maxlen , vocab_size, embedding_dim):
    embedding_layer = Embedding(vocab_size,
                                 embedding_dim,
                                 weights= [embedding_matrix2],
                                 input_length=maxlen,
                                 trainable= False)

    sequence_input = Input(shape=(maxlen,), dtype='int32')

    embedded_sequences = embedding_layer(sequence_input)



    l_conv = Conv1D(filters=50, kernel_size=3, activation='relu')(embedded_sequences)

    l_conv2 = Conv1D(filters=50, kernel_size=3, activation='relu')(l_conv)
    l_pool = GlobalMaxPooling1D()(l_conv2)







    x = Dense(512, activation='relu')(l_pool)
    x = Dropout(0.2)(x)
    preds = Dense(units=1,activation='sigmoid')(x)
    model = Model(sequence_input, preds)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    return model



def convnet6 (embedding_matrix2, maxlen , vocab_size, embedding_dim):
    embedding_layer = Embedding(vocab_size,
                                 embedding_dim,
                                 weights= [embedding_matrix2],
                                 input_length=maxlen,
                                 trainable= False)

    sequence_input = Input(shape=(maxlen,), dtype='int32')

    embedded_sequences = embedding_layer(sequence_input)



    l_conv = Conv1D(filters=100, kernel_size=3, activation='relu')(embedded_sequences)

    l_conv2 = Conv1D(filters=100, kernel_size=3, activation='relu')(l_conv)
    l_pool = GlobalMaxPooling1D()(l_conv2)



    x = Dense(512, activation='relu')(l_pool)
    x = Dropout(0.2)(x)
    preds = Dense(units=1,activation='sigmoid')(x)
    model = Model(sequence_input, preds)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    return model





def convnet7 (embedding_matrix2, maxlen , vocab_size, embedding_dim):
    embedding_layer = Embedding(vocab_size,
                                 embedding_dim,
                                 weights= [embedding_matrix2],
                                 input_length=maxlen,
                                 trainable= False)

    sequence_input = Input(shape=(maxlen,), dtype='int32')

    embedded_sequences = embedding_layer(sequence_input)



    l_conv = Conv1D(filters=128, kernel_size=3, activation='relu')(embedded_sequences)

    l_conv2 = Conv1D(filters=128, kernel_size=3, activation='relu')(l_conv)


    l_pool3 = GlobalMaxPooling1D()(l_conv2)


    x = Dense(512, activation='relu')(l_pool3)
    x = Dropout(0.2)(x)
    preds = Dense(units=1,activation='sigmoid')(x)
    model = Model(sequence_input, preds)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    return model



def convnet8 (embedding_matrix2, maxlen , vocab_size, embedding_dim):
    embedding_layer = Embedding(vocab_size,
                                 embedding_dim,
                                 weights= [embedding_matrix2],
                                 input_length=maxlen,
                                 trainable= False)

    sequence_input = Input(shape=(maxlen,), dtype='int32')

    embedded_sequences = embedding_layer(sequence_input)



    l_conv = Conv1D(filters=128, kernel_size=3, activation='relu')(embedded_sequences)

    l_conv2 = Conv1D(filters=128, kernel_size=5, activation='relu')(l_conv)


    l_pool = GlobalMaxPooling1D()(l_conv2)




    x = Dense(512, activation='relu')(l_pool)
    x = Dropout(0.2)(x)
    preds = Dense(units=1,activation='sigmoid')(x)
    model = Model(sequence_input, preds)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    return model







def convnet9 (embedding_matrix2, maxlen , vocab_size, embedding_dim):
    embedding_layer = Embedding(vocab_size,
                                 embedding_dim,
                                 weights= [embedding_matrix2],
                                 input_length=maxlen,
                                 trainable= False)

    sequence_input = Input(shape=(maxlen,), dtype='int32')

    embedded_sequences = embedding_layer(sequence_input)


    l_conv = Conv1D(filters=128, kernel_size=3, activation='relu')(embedded_sequences)

    l_conv2 = Conv1D(filters=64, kernel_size=5, activation='relu')(l_conv)


    l_pool = GlobalMaxPooling1D()(l_conv2)



    x = Dense(512, activation='relu')(l_pool)
    x = Dropout(0.2)(x)
    preds = Dense(units=1,activation='sigmoid')(x)
    model = Model(sequence_input, preds)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    return model





def convnet10 (embedding_matrix2, maxlen , vocab_size, embedding_dim):
    embedding_layer = Embedding(vocab_size,
                                 embedding_dim,
                                 weights= [embedding_matrix2],
                                 input_length=maxlen,
                                 trainable= False)

    sequence_input = Input(shape=(maxlen,), dtype='int32')

    embedded_sequences = embedding_layer(sequence_input)



    l_conv = Conv1D(filters=256, kernel_size=3, activation='relu')(embedded_sequences)

    l_conv2 = Conv1D(filters=256, kernel_size=3, activation='relu')(l_conv)


    l_pool = GlobalMaxPooling1D()(l_conv2)




    x = Dense(512, activation='relu')(l_pool)
    x = Dropout(0.2)(x)
    preds = Dense(units=1,activation='sigmoid')(x)
    model = Model(sequence_input, preds)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    return model


def convnet11(embedding_matrix2, maxlen, vocab_size, embedding_dim):
    embedding_layer = Embedding(vocab_size,
                                embedding_dim,
                                weights=[embedding_matrix2],
                                input_length=maxlen,
                                trainable=False)

    sequence_input = Input(shape=(maxlen,), dtype='int32')

    embedded_sequences = embedding_layer(sequence_input)

    l_conv = Conv1D(filters=256, kernel_size=3, activation='relu')(embedded_sequences)

    l_conv2 = Conv1D(filters=256, kernel_size=5, activation='relu')(l_conv)

    l_pool = GlobalMaxPooling1D()(l_conv2)

    x = Dense(512, activation='relu')(l_pool)
    x = Dropout(0.2)(x)
    preds = Dense(units=1, activation='sigmoid')(x)
    model = Model(sequence_input, preds)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def convnet12 (embedding_matrix2, maxlen , vocab_size, embedding_dim):
    embedding_layer = Embedding(vocab_size,
                                 embedding_dim,
                                 weights= [embedding_matrix2],
                                 input_length=maxlen,
                                 trainable= False)

    sequence_input = Input(shape=(maxlen,), dtype='int32')

    embedded_sequences = embedding_layer(sequence_input)


    l_conv = Conv1D(filters=100, kernel_size=3, activation='relu')(embedded_sequences)

    l_conv2 = Conv1D(filters=100, kernel_size=3, activation='relu')(l_conv)

    l_conv3 = Conv1D(filters=100, kernel_size=3, activation='relu')(l_conv2)
    l_pool = GlobalMaxPooling1D()(l_conv3)

    x = Dense(512, activation='relu')(l_pool)
    x = Dropout(0.2)(x)
    preds = Dense(units=1,activation='sigmoid')(x)
    model = Model(sequence_input, preds)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    return model




def convnet13 (embedding_matrix2, maxlen , vocab_size, embedding_dim):
    embedding_layer = Embedding(vocab_size,
                                 embedding_dim,
                                 weights= [embedding_matrix2],
                                 input_length=maxlen,
                                 trainable= False)

    sequence_input = Input(shape=(maxlen,), dtype='int32')

    embedded_sequences = embedding_layer(sequence_input)



    l_conv = Conv1D(filters=100, kernel_size=3, activation='relu')(embedded_sequences)
    #l_pool = MaxPooling1D()(l_conv)
    #convs.append(l_pool)
    l_conv2 = Conv1D(filters=100, kernel_size=5, activation='relu')(l_conv)
   # l_pool2 = MaxPooling1D()(l_conv2)
   # convs.append(l_pool2)
    l_conv3 = Conv1D(filters=100, kernel_size=7, activation='relu')(l_conv2)
    l_pool3 = GlobalMaxPooling1D()(l_conv3)
   # convs.append(l_pool3)
   # l_conv = Conv1D(filters=128, kernel_size=3, activation='relu')(embedded_sequences)
   # l_pool = GlobalMaxPooling1D()(l_conv)
   # convs.append(l_pool)
   # l_conv = Conv1D(filters=256, kernel_size=3, activation='relu')(embedded_sequences)
   # l_pool = GlobalMaxPooling1D()(l_conv)
   # convs.append(l_pool)


   # l_merge = concatenate(convs, axis=1)


    x = Dense(512, activation='relu')(l_pool3)
    x = Dropout(0.2)(x)
    preds = Dense(units=1,activation='sigmoid')(x)
    model = Model(sequence_input, preds)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    return model

def convnet14(embedding_matrix2, maxlen , vocab_size , embedding_dim):
    embedding_layer = Embedding(vocab_size,
                                 embedding_dim,
                                 weights= [embedding_matrix2],
                                 input_length=maxlen,
                                 trainable= False)

    sequence_input = Input(shape=(maxlen,), dtype='int32')

    embedded_sequences = embedding_layer(sequence_input)



    l_conv = Conv1D(filters=128, kernel_size=3, activation='relu')(embedded_sequences)

    l_conv2 = Conv1D(filters=128, kernel_size=3, activation='relu')(l_conv)

    l_conv3 = Conv1D(filters=128, kernel_size=3, activation='relu')(l_conv2)
    l_pool3 = GlobalMaxPooling1D()(l_conv3)



    x = Dense(512, activation='relu')(l_pool3)
    x = Dropout(0.2)(x)
    preds = Dense(units=1,activation='sigmoid')(x)
    model = Model(sequence_input, preds)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    return model

def convnet15(embedding_matrix2, maxlen , vocab_size , embedding_dim):
    embedding_layer = Embedding(vocab_size,
                                 embedding_dim,
                                 weights= [embedding_matrix2],
                                 input_length=maxlen,
                                 trainable= False)

    sequence_input = Input(shape=(maxlen,), dtype='int32')

    embedded_sequences = embedding_layer(sequence_input)

    l_conv = Conv1D(filters=128, kernel_size=3, activation='relu')(embedded_sequences)

    l_conv2 = Conv1D(filters=128, kernel_size=5, activation='relu')(l_conv)

    l_conv3 = Conv1D(filters=128, kernel_size=7, activation='relu')(l_conv2)
    l_pool3 = GlobalMaxPooling1D()(l_conv3)






    x = Dense(512, activation='relu')(l_pool3)
    x = Dropout(0.2)(x)
    preds = Dense(units=1,activation='sigmoid')(x)
    model = Model(sequence_input, preds)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    return model

def convnet16(embedding_matrix2, maxlen , vocab_size , embedding_dim):
    embedding_layer = Embedding(vocab_size,
                                 embedding_dim,
                                 weights= [embedding_matrix2],
                                 input_length=maxlen,
                                 trainable= False)

    sequence_input = Input(shape=(maxlen,), dtype='int32')

    embedded_sequences = embedding_layer(sequence_input)

    l_conv = Conv1D(filters=256, kernel_size=3, activation='relu')(embedded_sequences)

    l_conv2 = Conv1D(filters=256, kernel_size=3, activation='relu')(l_conv)

    l_conv3 = Conv1D(filters=256, kernel_size=3, activation='relu')(l_conv2)
    l_pool3 = GlobalMaxPooling1D()(l_conv3)






    x = Dense(512, activation='relu')(l_pool3)
    x = Dropout(0.2)(x)
    preds = Dense(units=1,activation='sigmoid')(x)
    model = Model(sequence_input, preds)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    return model

def convnet17(embedding_matrix2, maxlen , vocab_size , embedding_dim):
    embedding_layer = Embedding(vocab_size,
                                 embedding_dim,
                                 weights= [embedding_matrix2],
                                 input_length=maxlen,
                                 trainable= False)

    sequence_input = Input(shape=(maxlen,), dtype='int32')

    embedded_sequences = embedding_layer(sequence_input)


    l_conv = Conv1D(filters=256, kernel_size=3, activation='relu')(embedded_sequences)

    l_conv2 = Conv1D(filters=256, kernel_size=5, activation='relu')(l_conv)

    l_conv3 = Conv1D(filters=256, kernel_size=7, activation='relu')(l_conv2)

    l_pool = GlobalMaxPooling1D()(l_conv3)

    x = Dense(512, activation='relu')(l_pool)
    x = Dropout(0.2)(x)
    preds = Dense(units=1,activation='sigmoid')(x)
    model = Model(sequence_input, preds)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    return model






#model1 = ConvNet(embedding_matrix2, maxlen , vocab_size, embedding_dim)
#history1 = model1.fit(X_train, y_train, epochs= 3,batch_size=128, validation_data=(X_test, y_test))
#results1 = model1.evaluate(X_test, y_test, batch_size=128)

def plot_learningCurve1(history1, epochs):
  # Plot training & validation accuracy values
  epoch_range = range(1, epochs+1)
  plt.plot(epoch_range, history1.history['accuracy'])
  plt.plot(epoch_range, history1.history['val_accuracy'])
  plt.title('Model accuracy CNN1')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(epoch_range, history1.history['loss'])
  plt.plot(epoch_range, history1.history['val_loss'])
  plt.title('Model loss CNN1')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

#plot_learningCurve1(history1, 3)


#model2= convnet2(embedding_matrix2, maxlen , vocab_size, embedding_dim)
#history2 = model2.fit(X_train, y_train, epochs=3 ,batch_size=128, validation_data=(X_test, y_test))
#results2 = model2.evaluate(X_test, y_test, batch_size=128)
def plot_learningCurve2(history2, epochs):
  # Plot training & validation accuracy values
  epoch_range = range(1, epochs+1)
  plt.plot(epoch_range, history2.history['accuracy'])
  plt.plot(epoch_range, history2.history['val_accuracy'])
  plt.title('Model accuracy CNN2')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(epoch_range, history2.history['loss'])
  plt.plot(epoch_range, history2.history['val_loss'])
  plt.title('Model loss CNN2')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

#plot_learningCurve2(history2, 3)

#model3= convnet3(embedding_matrix2, maxlen , vocab_size, embedding_dim)
#history3 = model3.fit(X_train, y_train, epochs=3 ,batch_size=128, validation_data=(X_test, y_test))
#results3 = model3.evaluate(X_test, y_test, batch_size=128)

def plot_learningCurve3(history3, epochs):
  # Plot training & validation accuracy values
  epoch_range = range(1, epochs+1)
  plt.plot(epoch_range, history3.history['accuracy'])
  plt.plot(epoch_range, history3.history['val_accuracy'])
  plt.title('Model accuracy CNN3')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(epoch_range, history3.history['loss'])
  plt.plot(epoch_range, history3.history['val_loss'])
  plt.title('Model loss CNN3')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()




#plot_learningCurve3(history3, 3)








#model4= convnet4(embedding_matrix2, maxlen , vocab_size, embedding_dim)
#history4 = model4.fit(X_train, y_train, epochs=3 ,batch_size=128, validation_data=(X_test, y_test))
#results4 = model4.evaluate(X_test, y_test, batch_size=128)

def plot_learningCurve4(history4, epochs):
  # Plot training & validation accuracy values
  epoch_range = range(1, epochs+1)
  plt.plot(epoch_range, history4.history['accuracy'])
  plt.plot(epoch_range, history4.history['val_accuracy'])
  plt.title('Model accuracy CNN4')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(epoch_range, history4.history['loss'])
  plt.plot(epoch_range, history4.history['val_loss'])
  plt.title('Model loss CNN4')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

#plot_learningCurve4(history4, 3)

#model5= convnet5(embedding_matrix2, maxlen , vocab_size, embedding_dim)
#history5 = model5.fit(X_train, y_train, epochs=3 ,batch_size=128, validation_data=(X_test, y_test))
#results5 = model5.evaluate(X_test, y_test, batch_size=128)

def plot_learningCurve5(history5, epochs):
  # Plot training & validation accuracy values
  epoch_range = range(1, epochs+1)
  plt.plot(epoch_range, history5.history['accuracy'])
  plt.plot(epoch_range, history5.history['val_accuracy'])
  plt.title('Model accuracy CNN5')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(epoch_range, history5.history['loss'])
  plt.plot(epoch_range, history5.history['val_loss'])
  plt.title('Model loss CNN5')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

#plot_learningCurve5(history5, 3)






#model6= convnet6(embedding_matrix2, maxlen , vocab_size, embedding_dim)
#history6 = model6.fit(X_train, y_train, epochs=3 ,batch_size=128, validation_data=(X_test, y_test))
#results6 = model6.evaluate(X_test, y_test, batch_size=128)

def plot_learningCurve6(history6, epochs):
  # Plot training & validation accuracy values
  epoch_range = range(1, epochs+1)
  plt.plot(epoch_range, history6.history['accuracy'])
  plt.plot(epoch_range, history6.history['val_accuracy'])
  plt.title('Model accuracy CNN6')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(epoch_range, history6.history['loss'])
  plt.plot(epoch_range, history6.history['val_loss'])
  plt.title('Model loss CNN6')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

#plot_learningCurve6(history6, 3)












#model7= convnet7(embedding_matrix2, maxlen , vocab_size, embedding_dim)
#history7 = model7.fit(X_train, y_train, epochs=3,batch_size=128, validation_data=(X_test, y_test))
#results7 = model7.evaluate(X_test, y_test, batch_size=128)


def plot_learningCurve7(history7, epochs):
  #Plot training & validation accuracy values
  epoch_range = range(1, epochs+1)
  plt.plot(epoch_range, history7.history['accuracy'])
  plt.plot(epoch_range, history7.history['val_accuracy'])
  plt.title('Model accuracy CNN7')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

#Plot training & validation loss values
  plt.plot(epoch_range, history7.history['loss'])
  plt.plot(epoch_range, history7.history['val_loss'])
  plt.title('Model loss CNN7')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

#plot_learningCurve7(history7, 3)











#model8= convnet8(embedding_matrix2, maxlen , vocab_size, embedding_dim)
#history8 = model8.fit(X_train, y_train, epochs=3,batch_size=128, validation_data=(X_test, y_test))
#results8 = model8.evaluate(X_test, y_test, batch_size=128)


def plot_learningCurve8(history8, epochs):
  # Plot training & validation accuracy values
  epoch_range = range(1, epochs+1)
  plt.plot(epoch_range, history8.history['accuracy'])
  plt.plot(epoch_range, history8.history['val_accuracy'])
  plt.title('Model accuracy CNN8')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(epoch_range, history8.history['loss'])
  plt.plot(epoch_range, history8.history['val_loss'])
  plt.title('Model loss CNN8')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

#plot_learningCurve8(history8, 3)






#model9= convnet9(embedding_matrix2, maxlen , vocab_size, embedding_dim)
#history9 = model9.fit(X_train, y_train, epochs=3 ,batch_size=128, validation_data=(X_test, y_test))
#results9 = model9.evaluate(X_test, y_test, batch_size=128)


def plot_learningCurve9(history9, epochs):
  # Plot training & validation accuracy values
  epoch_range = range(1, epochs+1)
  plt.plot(epoch_range, history9.history['accuracy'])
  plt.plot(epoch_range, history9.history['val_accuracy'])
  plt.title('Model accuracy CNN9')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(epoch_range, history9.history['loss'])
  plt.plot(epoch_range, history9.history['val_loss'])
  plt.title('Model loss CNN9')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

#plot_learningCurve9(history9, 3)




model10= convnet10(embedding_matrix2, maxlen , vocab_size, embedding_dim)
history10 = model10.fit(X_train, y_train, epochs=3 ,batch_size=128, validation_data=(X_test, y_test))
results10= model10.evaluate(X_test, y_test, batch_size=128)
def plot_learningCurve10(history10, epochs):
  # Plot training & validation accuracy values
  epoch_range = range(1, epochs+1)
  plt.plot(epoch_range, history10.history['accuracy'])
  plt.plot(epoch_range, history10.history['val_accuracy'])
  plt.title('Model accuracy CNN10')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(epoch_range, history10.history['loss'])
  plt.plot(epoch_range, history10.history['val_loss'])
  plt.title('Model loss CNN10')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

plot_learningCurve10(history10, 3)


model11= convnet11(embedding_matrix2, maxlen , vocab_size, embedding_dim)
history11 = model11.fit(X_train, y_train, epochs=3 ,batch_size=128, validation_data=(X_test, y_test))
results11= model11.evaluate(X_test, y_test, batch_size=128)


def plot_learningCurve11(history11, epochs):
  # Plot training & validation accuracy values
  epoch_range = range(1, epochs+1)
  plt.plot(epoch_range, history11.history['accuracy'])
  plt.plot(epoch_range, history11.history['val_accuracy'])
  plt.title('Model accuracy CNN11')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(epoch_range, history11.history['loss'])
  plt.plot(epoch_range, history11.history['val_loss'])
  plt.title('Model loss CNN11')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

plot_learningCurve11(history11, 3)






model12= convnet12(embedding_matrix2, maxlen , vocab_size, embedding_dim)
history12 = model12.fit(X_train, y_train, epochs=3,batch_size=128, validation_data=(X_test, y_test))
results12= model12.evaluate(X_test, y_test, batch_size=128)


def plot_learningCurve12(history12, epochs):
  # Plot training & validation accuracy values
  epoch_range = range(1, epochs+1)
  plt.plot(epoch_range, history12.history['accuracy'])
  plt.plot(epoch_range, history12.history['val_accuracy'])
  plt.title('Model accuracy CNN12')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(epoch_range, history12.history['loss'])
  plt.plot(epoch_range, history12.history['val_loss'])
  plt.title('Model loss CNN12')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

plot_learningCurve12(history12, 3)



model13=convnet13(embedding_matrix2, maxlen , vocab_size, embedding_dim)
history13 = model13.fit(X_train, y_train, epochs=3 ,batch_size=128, validation_data=(X_test, y_test))
results13= model13.evaluate(X_test, y_test, batch_size=128)


def plot_learningCurve13(history13, epochs):
  # Plot training & validation accuracy values
  epoch_range = range(1, epochs+1)
  plt.plot(epoch_range, history13.history['accuracy'])
  plt.plot(epoch_range, history13.history['val_accuracy'])
  plt.title('Model accuracy CNN13')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(epoch_range, history13.history['loss'])
  plt.plot(epoch_range, history13.history['val_loss'])
  plt.title('Model loss CNN13')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

plot_learningCurve13(history13, 3)






model14=convnet14(embedding_matrix2, maxlen , vocab_size, embedding_dim)
history14= model14.fit(X_train, y_train, epochs=3 ,batch_size=128, validation_data=(X_test, y_test))
results14= model14.evaluate(X_test, y_test, batch_size=128)


def plot_learningCurve14(history14, epochs):
  # Plot training & validation accuracy values
  epoch_range = range(1, epochs+1)
  plt.plot(epoch_range, history14.history['accuracy'])
  plt.plot(epoch_range, history14.history['val_accuracy'])
  plt.title('Model accuracy CNN14')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(epoch_range, history14.history['loss'])
  plt.plot(epoch_range, history14.history['val_loss'])
  plt.title('Model loss CNN14')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

plot_learningCurve14(history14, 3)


model15=convnet15(embedding_matrix2, maxlen , vocab_size, embedding_dim)
history15 = model15.fit(X_train, y_train, epochs=3 ,batch_size=128, validation_data=(X_test, y_test))
results15= model15.evaluate(X_test, y_test, batch_size=128)


def plot_learningCurve15(history15, epochs):
  # Plot training & validation accuracy values
  epoch_range = range(1, epochs+1)
  plt.plot(epoch_range, history15.history['accuracy'])
  plt.plot(epoch_range, history15.history['val_accuracy'])
  plt.title('Model accuracy CNN15')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(epoch_range, history15.history['loss'])
  plt.plot(epoch_range, history15.history['val_loss'])
  plt.title('Model loss CNN15')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

plot_learningCurve15(history15, 3)



model16=convnet16(embedding_matrix2, maxlen , vocab_size, embedding_dim)
history16 = model16.fit(X_train, y_train, epochs=3 ,batch_size=128, validation_data=(X_test, y_test))
results16= model16.evaluate(X_test, y_test, batch_size=128)


def plot_learningCurve16(history16, epochs):
  # Plot training & validation accuracy values
  epoch_range = range(1, epochs+1)
  plt.plot(epoch_range, history16.history['accuracy'])
  plt.plot(epoch_range, history16.history['val_accuracy'])
  plt.title('Model accuracy CNN16')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(epoch_range, history16.history['loss'])
  plt.plot(epoch_range, history16.history['val_loss'])
  plt.title('Model loss CNN16')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

plot_learningCurve16(history16, 3)





model17=convnet17(embedding_matrix2, maxlen , vocab_size, embedding_dim)
history17 = model17.fit(X_train, y_train, epochs=3 ,batch_size=128, validation_data=(X_test, y_test))
results17= model17.evaluate(X_test, y_test, batch_size=128)


def plot_learningCurve17(history17, epochs):
  # Plot training & validation accuracy values
  epoch_range = range(1, epochs+1)
  plt.plot(epoch_range, history17.history['accuracy'])
  plt.plot(epoch_range, history17.history['val_accuracy'])
  plt.title('Model accuracy CNN17')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(epoch_range, history17.history['loss'])
  plt.plot(epoch_range, history17.history['val_loss'])
  plt.title('Model loss CNN17')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

plot_learningCurve17(history17, 3)
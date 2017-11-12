import os
import sys
import csv
import json
import numpy
import pandas
from collections import OrderedDict

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.preprocessing import sequence
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint

# fix random seed for reproducibility
# numpy.random.seed(7)

frame        =   80
features     = 4096
max_caption  =   40
max_sentence =   40
batch_       =   25
voca_filter  =    2
epochs_total =   80
epochs_same  =    5

# readin
# training_label.json  list.len=1450  
train_label_f = open( os.path.join( sys.argv[1], 'training_label.json'), 'r')
train_label   = json.load(train_label_f)
train_label_f.close()


# vocabulary = numpy.load('vocabulary.npy')
vocabulary = OrderedDict( { '' : 100 } )
for i in range( len(train_label) ):
    for cap in range(len(train_label[i]['caption'])):
        train_label[i]['caption'][cap] = train_label[i]['caption'][cap].lower().replace(",", "").replace(".", "").replace("  ", " ")
        for word in range( len(train_label[i]['caption'][cap].split(' ')) ):
            temp_word = train_label[i]['caption'][cap].split(' ')[word]
            vocabulary[temp_word] = vocabulary.get(temp_word, 0) + 1
vocabulary = numpy.array([ k for k,v in vocabulary.items() if v>voca_filter ])
print(vocabulary.shape)
print(vocabulary)
numpy.save('vocabulary.npy', vocabulary)

# training_data/feat/'id.avi.npy' shape=(80, 4096)=(frame, features)
X_train = numpy.zeros( ( len(train_label), frame, features), dtype=numpy.float)
y_data = numpy.zeros( ( max_caption, len(train_label), frame + max_sentence), dtype=numpy.int16)
y_train = numpy.zeros( ( len(train_label), frame + max_sentence, 1), dtype=numpy.float)
caption_choice = numpy.zeros( ( len(train_label), epochs_total), dtype=numpy.int16)
for i in range( len(train_label) ):
    X_train[i] = numpy.load( os.path.join( sys.argv[1], 'training_data', 'feat', ((train_label[i]['id'])+'.npy')))
    caption_choice[i] = numpy.random.choice(len(train_label[i]['caption']), epochs_total)
    for cap in range(len(train_label[i]['caption'])):
        train_label[i]['caption'][cap] = train_label[i]['caption'][cap].lower().replace(",", "").replace(".", "").replace("  ", " ")
        for word in range( len(train_label[i]['caption'][cap].split(' ')) ):
            temp_word = train_label[i]['caption'][cap].split(' ')[word]
            if len(numpy.where( vocabulary == temp_word )[0]) > 0:
                y_data[cap, i, frame + word] = numpy.where( vocabulary == temp_word )[0][0]
# for debugging
X_train = numpy.append( X_train, numpy.zeros( ( X_train.shape[0], max_sentence, features), dtype=numpy.float), axis=1)
y_data = numpy.expand_dims(y_data, -1)
print('X(samples, frame+max_sentence, features):', X_train.shape)
print('--------------------------------')
print('y(max_caption, samples, frame+max_sentence, vocabulary):', y_data.shape)
print('--------------------------------')

# Start training
model = Sequential()
model.add(LSTM( 512,
               # input_length=TIME_STEPS, input_dim=INPUT_SIZE
               input_shape=(X_train.shape[1], X_train.shape[2]),      
               batch_size=batch_,
               return_sequences=True,
               stateful=True))
model.add(Dropout(0.25))
model.add(LSTM( 512, return_sequences=True))
model.add(Dropout(0.25))
model.add(TimeDistributed(Dense( vocabulary.shape[0], activation='softmax')))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())
checkpointer = ModelCheckpoint(filepath='./temp.h5', verbose=1, save_best_only=False)
for epo in range(epochs_total):
    for s in range(y_train.shape[0]):
        y_train[s] = y_data[ caption_choice[s,epo], s]
    print(epo)
    model.fit(X_train, y_train, epochs=epochs_same, batch_size=batch_, callbacks=[checkpointer])
model.save(sys.argv[2])
import sys
import csv
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
# fix random seed for reproducibility
# numpy.random.seed(7)

features = 69
# readin
# train.ark (1124823, 70+39)  test.ark (180406, 70)
X_temp = pandas.read_csv(sys.argv[1]+'fbank/train.ark', sep=' ', header=None).values
# train.lab (1124823, 2)
y_temp = pandas.read_csv(sys.argv[1]+'label/train.lab', sep=',', header=None).values
map48phone_char = pandas.read_csv(sys.argv[1]+'48phone_char.map', sep='\t', header=None).values
d48tonum = OrderedDict( zip(map48phone_char[:,0], map48phone_char[:,1]) )

# aligning
d1 = OrderedDict( zip(X_temp[:,0], numpy.zeros(X_temp.shape[0])) )
d2 = OrderedDict( zip(y_temp[:,0], y_temp[:,1]) )
d1.update(d2)
y_temp = numpy.array( list( d1.values() ) )

# mapping
for i in range(y_temp.shape[0]):
    y_temp[i] = d48tonum.get(str(y_temp[i]))
y_temp = y_temp.astype(numpy.int16)

# reshape
wav_count = 1
for i in range(X_temp.shape[0]):
    X_temp[i, 0] = int(str(X_temp[i, 0]).split('_')[2])
for i in range(X_temp.shape[0]-1):
    if X_temp[i, 0] > X_temp[i+1, 0] :
        wav_count = wav_count + 1
max_time = int(numpy.amax(X_temp[:,0]))


X = numpy.zeros((wav_count, max_time, features), numpy.float)
y = numpy.zeros((wav_count, max_time, 1 ), numpy.int16)

count = 0
for i in range(X_temp.shape[0]-1):
    if X_temp[i, 0] > X_temp[i+1, 0] or i == (X_temp.shape[0]-2) :
        flame = X_temp[i, 0]
        X_resh = numpy.reshape(X_temp[( i+1- flame) : (i+1), 1: ], (1, flame, features))
        y_resh = numpy.reshape(y_temp[( i+1- flame) : (i+1) ] , (1, flame, 1))
        zerofeatures = numpy.zeros((1, max_time-flame, features), numpy.float)
        zero1  = numpy.ones((1, max_time-flame, 1 ), numpy.int16) * 37
        # numpy.repeat( numpy.reshape( y_resh[0, flame-1,:], (1, 1, 1)), max_time-flame, axis=1)
        X[count] = numpy.append( X_resh , zerofeatures, axis=1)
        y[count] = numpy.append( y_resh , zero1 , axis=1)
        count = count + 1


X_train = numpy.copy(X)
y_train = keras.utils.to_categorical( y , 48 )
y_train = numpy.reshape(y_train, (X_train.shape[0], X_train.shape[1], 48))

# for debugging
print('X(samples, timesteps, input_dim):', X_train.shape)
print('--------------------------------')
print('y(samples, timesteps, output_dim):', y_train.shape)
print('--------------------------------')
# Start training
model = Sequential()
# model.add(Embedding(features, output_dim=256))
model.add(LSTM(2048,
               # input_length=TIME_STEPS, input_dim=INPUT_SIZE
               input_shape=(X_train.shape[1], X_train.shape[2]),      
               batch_size=16,
               return_sequences=True,
               stateful=True))
model.add(Dropout(0.125))
model.add(LSTM(2048, return_sequences=True))
model.add(Dropout(0.125))
model.add(LSTM(2048, return_sequences=True))
model.add(Dropout(0.125))
model.add(TimeDistributed(Dense(48, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=16, batch_size=16)
model.save(sys.argv[2])
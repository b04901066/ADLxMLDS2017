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
# fix random seed for reproducibility
# numpy.random.seed(7)

frame        =   80
features     = 4096
max_caption  =   40
max_sentence =   40
batch_       =   25
voca_filter  =    2

# readin
'''
# testing_label.json  list.len=1450  
test_label_f = open( os.path.join( sys.argv[1], 'testing_label.json'), 'r')
test_label   = json.load(test_label_f)
test_label_f.close()
'''
# testing_data/feat/'id.avi.npy' shape=(80, 4096)=(frame, features)
npyfiles = os.listdir( os.path.join( sys.argv[1], sys.argv[4], 'feat') )
X_test = numpy.zeros( ( 5, frame, features), dtype=numpy.float)
X_ID = numpy.zeros( ( 5), dtype=str)
#y_train = numpy.zeros( ( len(train_label), max_caption, max_sentence), dtype=numpy.int16)
vocabulary = numpy.load('./vocabulary.npy')
counter = 0
for file in npyfiles:
    if file == 'klteYv1Uv9A_27_33.avi.npy' or file == '5YJaS2Eswg0_22_26.avi.npy' or file == 'UbmZAe5u5FI_132_141.avi.npy' or file == 'JntMAcTlOF0_50_70.avi.npy' or file == 'tJHUH9tpqPg_113_118.avi.npy':
        X_test[counter] = numpy.load( os.path.join( sys.argv[1], 'testing_data', 'feat', file))
        X_ID[counter] = file
        counter += 1

print('X(samples, frame, features):', X_test.shape)
print('--------------------------------')
X_test = numpy.append( X_test, numpy.zeros( ( X_test.shape[0], max_sentence, features), dtype=numpy.float), axis=1)

# Start testing
model = load_model(sys.argv[3])
with open(sys.argv[2], 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    input_X = numpy.append(X_test, numpy.zeros((batch_-X_test.shape[0], frame+max_sentence, features), numpy.float), axis=0)
    y = numpy.argmax( model.predict( input_X ), axis=2)
    # trimming
    for s in range(5):
        output = ''
        for out in range(y.shape[1]):
            output += vocabulary[y[s][out]]
            if vocabulary[y[s][out]] != '':
                output += ' '
        output = output[:-1]
        output += '.'
        output = output.capitalize()

        print(output)
        spamwriter.writerow([ X_ID[s], output])

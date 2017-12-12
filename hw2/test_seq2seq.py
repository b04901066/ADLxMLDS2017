import os
import sys
import csv
import json
import math
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
max_sentence =   15
batch_       =   25
voca_filter  =    3

# readin
# testing_data/feat/'id.avi.npy' shape=(80, 4096)=(frame, features)
npyfiles = os.listdir( os.path.join( sys.argv[1], sys.argv[4], 'feat') )
X_test = numpy.zeros( ( len(npyfiles), frame, features), dtype=numpy.float)
#y_train = numpy.zeros( ( len(train_label), max_caption, max_sentence), dtype=numpy.int16)
vocabulary = numpy.load('./vocabulary.npy')
counter = 0
for file in npyfiles:
    X_test[counter] = numpy.load( os.path.join( sys.argv[1], sys.argv[4], 'feat', file))
    counter += 1

print('X(samples, frame, features):', X_test.shape)
print('--------------------------------')
X_test = numpy.append( X_test, numpy.zeros( ( X_test.shape[0], max_sentence, features), dtype=numpy.float), axis=1)

# Start testing
model = load_model(sys.argv[3])
with open(sys.argv[2], 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for times in range( int(math.ceil(X_test.shape[0]/batch_)) ):
        if batch_*(times+1) > X_test.shape[0]:
            input_X = numpy.append(X_test[batch_*times:X_test.shape[0]], numpy.zeros((batch_*(times+1)-X_test.shape[0], frame+max_sentence, features), numpy.float), axis=0)
        else:
            input_X = numpy.copy(X_test[batch_*times:batch_*(times+1)])
        # input_X (batch_, 120, 4096)  y (batch_, 120)
        y = numpy.argmax( model.predict( input_X ), axis=2)

        # trimming
        for s in range(batch_):
            output = ''
            for out in range(y.shape[1]):
                if out > 0 and y[s][out-1] == y[s][out]:
                    continue
                if vocabulary[y[s][out]] != '':
                    output += vocabulary[y[s][out]]
                    output += ' '
            output = output[:-1]
            output += '.'
            output = output.capitalize()
            output = output.replace("_", " ")
            output = output.replace(" of.", " of a stuff.").replace(" on.", " on a stuff.").replace(" up.", " up a stuff.").replace(" in.", " in a stuff.").replace(" be.", " be a stuff.").replace(" of.", " of a stuff.").replace(" at.", " at a palce.").replace(" from.", " from a palce.").replace(" and.", " and a stuff.").replace(".", "")
            if (batch_*times+s) < len(npyfiles) :
                print(npyfiles[batch_*times+s][:-4] + ',' + output)
                if sys.argv[4] == 'testing_data' :
                    output = output.split(' ')[0] + ' ' + output.split(' ')[1] + ' is a a'
                spamwriter.writerow([ npyfiles[batch_*times+s][:-4], output])

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

features = 108
# for trimming
LPfilter = 3

# readin
# test.ark (180406, 70)
X_temp = pandas.read_csv(sys.argv[1]+'fbank/test.ark', sep=' ', header=None).values
X_temp2 = pandas.read_csv(sys.argv[1]+'mfcc/test.ark', sep=' ', header=None).values
X_temp = numpy.append( X_temp, X_temp2[:, 1:], axis=1)

ID = numpy.empty([0])
wav_count = 0
for i in range(X_temp.shape[0]):
    if str(X_temp[i, 0]).split('_')[2] == '1':
        ID = numpy.append( ID, numpy.reshape( ((str(X_temp[i, 0]).split('_')[0]) + '_' + (str(X_temp[i, 0]).split('_')[1])) , (1) ), axis=0)
        wav_count = wav_count + 1
    X_temp[i, 0] = int(str(X_temp[i, 0]).split('_')[2])

max_time = 777
X = numpy.zeros((wav_count, max_time, features), numpy.float)
print('wav_count:', wav_count, ' time_length:', numpy.amax(X_temp[:,0]), ' model_time_length', max_time)
count = 0
for i in range(X_temp.shape[0]):
    if i == (X_temp.shape[0]-1):
        flame = X_temp[i, 0]
        X[count] = numpy.append( numpy.reshape(X_temp[( i+1- flame) : (i+1), 1: ], (1, flame, features)), numpy.zeros((1, max_time-flame, features), numpy.float), axis=1)
        count = count + 1
    elif X_temp[i, 0] > X_temp[i+1, 0] :
        flame = X_temp[i, 0]
        X[count] = numpy.append( numpy.reshape(X_temp[( i+1- flame) : (i+1), 1: ], (1, flame, features)), numpy.zeros((1, max_time-flame, features), numpy.float), axis=1)
        count = count + 1


# for debugging
print('ID(samples):', ID.shape)
print('--------------------------------')
print('X(samples, timesteps, output_dim):', X.shape)
print('--------------------------------')

X_test = numpy.copy(X)
# create map for output
map48phone_char = pandas.read_csv(sys.argv[1]+'48phone_char.map', sep='\t', header=None).values
map48_39 = pandas.read_csv(sys.argv[1]+'phones/48_39.map', sep='\t', header=None).values
dnumto48 = OrderedDict( zip(map48phone_char[:,1], map48phone_char[:,0]) )
d48to39 = OrderedDict( zip(map48_39[:,0], map48_39[:,1]) )
d48tooutput = OrderedDict( zip(map48phone_char[:,0], map48phone_char[:,2]) )
dnum2output = map48phone_char[:,1:3]
for i in range(dnum2output.shape[0]):
    ph = dnumto48.get(int(dnum2output[i,0]))
    ph39 = d48to39.get( ph )
    dnum2output[i,1] = d48tooutput.get( ph39 )
dnum2output = OrderedDict( zip( dnum2output[:,0], dnum2output[:,1]) )

# Start testing
model = load_model(sys.argv[3])
with open(sys.argv[2], 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['id', 'phone_sequence'])
    for times in range( X_test.shape[0]//16 ):
        if 16*(times+1) > X_test.shape[0]:
            input_X = numpy.append(X_test[16*times:X_test.shape[0]], numpy.zeros((16*(times+1)-X_test.shape[0], max_time, 69+39), numpy.float), axis=0)
        else:
            input_X = numpy.copy(X_test[16*times:16*(times+1)])
        # input_X (16, 777, 48)  y (16, 777)
        y = numpy.argmax( model.predict( input_X ), axis=2)

        # trimming
        for s in range(16):
            i = 0
            j = y.shape[1]
            
            while( dnum2output.get(y[s, i]) == 'L' and i<j ):
                i = i + 1
            while( dnum2output.get(y[s, j-1 ]) == 'L' and i<j ):
                j = j - 1
            
            if i >= (j-1):
                print('error!!!')
            
            repeatcount = 0
            output = ''
            for out in range(i, j):
                repeatcount = repeatcount + 1
                if out == (j-1):
                    if repeatcount >= LPfilter:
                        output += dnum2output.get(y[s, out])
                    repeatcount = 0
                elif (dnum2output.get(y[s, out])) != (dnum2output.get(y[s, out+1])) :
                    if repeatcount >= LPfilter:
                        output += dnum2output.get(y[s, out])
                    repeatcount = 0

            output2 = ''
            for out in range(len(output)):
                if out == (len(output)-1):
                    output2 += output[out]
                elif output[out] != output[out+1] :
                    output2 += output[out]

            if (16*times+s) < ID.shape[0]:
                print(output2)
                spamwriter.writerow([ ID[16*times+s], output2])
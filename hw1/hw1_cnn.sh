#!/bin/bash
wget 'https://www.dropbox.com/s/e8smfipin3fesxw/cnn.h5?dl=1' -O './cnn.h5'
python ./cnn_LSTM_test.py $1 $2 ./cnn.h5
exit 0

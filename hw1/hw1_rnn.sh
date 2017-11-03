#!/bin/bash
wget 'https://www.dropbox.com/s/qw4jr3g19surfn6/rnn.h5?dl=1' -O './rnn.h5'
python ./rnn_LSTM_test.py $1 $2 ./rnn.h5
exit 0

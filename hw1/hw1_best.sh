#!/bin/bash
wget 'https://www.dropbox.com/s/3ajysb98vdepbi9/LSTM_16epo_3_2048.h5?dl=1' -O './LSTM_16epo_3_2048.h5'
python ./best_test.py $1 $2 ./LSTM_16epo_3_2048.h5
exit 0

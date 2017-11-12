#!/bin/bash
# wget 'https://www.dropbox.com/s/qw4jr3g19surfn6/rnn.h5?dl=1' -O './rnn.h5'
python ./test_special.py $1 $2 ./seq2seq_model_special.h5 testing_data
exit 0
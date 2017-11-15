#!/bin/bash
# wget 'https://www.dropbox.com/s/qw4jr3g19surfn6/rnn.h5?dl=1' -O './rnn.h5'
python ./test_seq2seq.py $1 $2 ./seq2seq_model.h5 testing_data
python ./test_seq2seq.py $1 $3 ./seq2seq_model.h5 peer_review
exit 0
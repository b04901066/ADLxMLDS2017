#!/bin/bash
wget 'https://www.dropbox.com/s/j6vcd425xw9xgzi/seq210.h5?dl=1' -O './seq2seq_model.h5'
python ./test_seq2seq.py $1 $2 ./seq2seq_model.h5 testing_data
python ./test_seq2seq.py $1 $3 ./seq2seq_model.h5 peer_review
exit 0

#!/bin/bash
wget '' -O './seq2seq_model.h5'
python ./test_special.py $1 $2 ./seq2seq_model.h5 testing_data
exit 0
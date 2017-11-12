#!/bin/bash
wget 'https://github.com/b04901066/ADLxMLDS2017/releases/download/MLDS_hw2/seq2seq_model_1.h5' -O './seq2seq_model.h5'
python ./test_special.py $1 $2 ./seq2seq_model.h5 testing_data
exit 0
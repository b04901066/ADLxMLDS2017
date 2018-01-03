## command (use python3.6)
 * train
   ```python
   python3.6 train.py
   ```
   
   > change path & settings in train.py Class ACGAN.__init__()   
   > change training settings in train.py Class ACGAN.train(args)
 * test
   ```python
   bash run.sh [testing_text.txt]
   ```
 * extra_test
   ```python
   bash extra_run.sh [testing_text.txt]
   ```   
   > This is same as run.sh
## library
* Keras (2.1.1) ( Tensorflow backend )  [ (2.0.7) should be fine ]
* h5py (2.7.1)
* numpy (1.13.3)
* pandas (0.21.0)
* scikit-image (0.13.1)
* scipy (1.0.0)
* Python Standard Lib

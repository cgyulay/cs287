# cs287
Psets and work related to CS 287.

----
####Primary contributors
[Alex Saich](https://github.com/asaich)

[Colton Gyulay](https://github.com/cgyulay)

----
####HW 1 Usage

Naive Bayes
```
th HW1.lua -datafile SST1.hdf5 -classifier nb
```

Logistic Regression
```
th HW1.lua -datafile SST1.hdf5 -classifier lr-cross
```

Limear SVM
```
th HW1.lua -datafile SST1.hdf5 -classifier lr-hinge
```

The following hyperparameters can be specified from the command line:
* datafile: the hdf5 file used for training/validation
* classifier: the specific model to run
* alpha: Laplace smoothing coefficient
* lr: learning rate for SGD
* lambda: l2 regularization coefficient
* n_epochs: number of training epochs
* m: mini-batch size
* kfold: number of k-folds for cross-validation

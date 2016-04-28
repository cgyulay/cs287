# cs287
Psets and work related to CS 287.

----
####Overview

This repo features a number of different language tasks and approaches. Most models are written with the help of the library [Torch](https://github.com/torch/torch7).
* HW1: Sentiment Analysis (naive Bayes, logistic regression, SVM)
* HW2: POS Tagging (naive Bayes, logistic regression, MLP)
* HW3: Language Modeling (n-gram, neural probabilistic LM)
* HW4: Word Segmentation (n-gram, neural probabilistic LM, RNN/LSTM)
* HW5: Named Entity Recognition (HMM, MEMM, structured perceptron + Viterbi)

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

----
####Primary contributors

[Colton Gyulay](https://github.com/cgyulay)

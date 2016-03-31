#!/usr/bin/env python

"""Space Prediction Preprocessing
"""

import numpy as np
import h5py
import argparse
import sys
import re
import codecs

SPACE = '<space>'
START = '<s>'
FILE_PATHS = {"PTB": ("data/train_chars.txt",
              "data/valid_chars.txt",
              "data/valid_chars_kaggle.txt",
              "data/test_chars.txt")}
args = {}
word_to_idx = {}

# For train/valid datasets
def build_ngrams(file_list, ngram):
  input_ngrams = {}
  output = {} # 0 or 1 depending on presence of space or not
  for filename in file_list:
    if filename:
      input_ngrams[filename] = []
      output[filename] = []
      with codecs.open(filename, "r", encoding="latin-1") as f:
        print('Building ngrams from ' + filename + '...')

        for line in f:
          words = [word_to_idx[str(w)] for w in line.split()]
          start = [word_to_idx[START]] * (ngram - 1)
          words = start + words
          
          for i in xrange(ngram, len(words)+1):
            context = words[i-ngram:i]
            inp = context[:-1]
            out = int(context[-1] == word_to_idx[SPACE]) # 1 = space, 0 ow
            input_ngrams[filename].append(inp)
            output[filename].append(out)
  return input_ngrams, output

def build_word_dict(file_list):
  last_idx = 3
  word_to_idx[SPACE] = 1
  word_to_idx[START] = 2
  for filename in file_list:
    if filename:
      with codecs.open(filename, "r", encoding="latin-1") as f:
        count = 0
        for line in f:
          count = count + 1
          letters = line.split(' ')
          for l in letters:
            l = l.rstrip() # Remove pesky line feed from </s>
            if l not in word_to_idx:
              word_to_idx[l] = last_idx
              last_idx = last_idx + 1

def main(arguments):
  global args
  parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('dataset', help="Data set",
            type=str)
  parser.add_argument('ngram', help="Length of ngram",
            type=int)
  args = parser.parse_args(arguments)
  dataset = args.dataset
  ngram = args.ngram
  train, valid, valid_kaggle, test = FILE_PATHS[dataset]

  build_word_dict([train, valid, valid_kaggle, test])
  input_dict, output_dict = build_ngrams([train, valid], ngram)

  train_input = np.array(input_dict[train], dtype=np.int32)
  train_output = np.array(output_dict[train], dtype=np.int32)
  valid_input = np.array(input_dict[valid], dtype=np.int32)
  valid_output = np.array(output_dict[valid], dtype=np.int32)

  filename = args.dataset + '_' + str(ngram) + 'gram.hdf5'
  with h5py.File(filename, "w") as f:
    f['train_input'] = train_input
    f['train_output'] = train_output
    if valid:
      f['valid_input'] = valid_input
      f['valid_output'] = valid_output
    # if test:
    #   f['test_input'] = test_input
    f['nclasses'] = np.array([2], dtype=np.int32) # space or not
    f['ngram'] = np.array([ngram], dtype=np.int32)


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))

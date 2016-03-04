#!/usr/bin/env python

"""Language modeling preprocessing
"""

import numpy as np
import h5py
import argparse
import sys
import re
import codecs

# Your preprocessing, features construction, and word2vec code.

START = '<s>'
END = '</s>'

FILE_PATHS = {"PTB": ("data/train.txt",
              "data/valid.txt",
              "data/test_blanks.txt",
              "data/words.dict"),
              "PTB1000": ("data/train.1000.txt",
              "data/valid.1000.txt",
              "data/test_blanks.txt",
              "data/words.1000.dict")}
args = {}
word_to_idx = {}
word_freq = {}

def build_ngrams(file_list, ngram):
  input_ngrams = {}
  output = {}
  for filename in file_list:
    if filename:
      input_ngrams[filename] = []
      output[filename] = []
      with codecs.open(filename, "r", encoding="latin-1") as f:
        print('Building ngrams from ' + filename + '...')

        iterlines = iter(f)
        next(iterlines) # Skip first line because it's nonsense
        for line in iterlines:
          words = [word_to_idx[str(w)] for w in line.split()]

          # Padding
          start = [word_to_idx[START]] * (ngram - 1)
          end = [word_to_idx[END]]
          words = start + words + end
          
          for i in xrange(ngram, len(words)+1):
            context = words[i-ngram:i]
            inp = context[:-1]
            out = context[-1]
            input_ngrams[filename].append(inp)
            output[filename].append(out)

          return

def build_word_dict(filename):
  last_idx = -1
  with codecs.open(filename, "r", encoding="latin-1") as f:
    for line in f:
      l = line.split()
      idx = int(l[0])
      word = str(l[1])
      freq = int(l[2])
      word_to_idx[word] = idx
      word_freq[idx] = freq
      last_idx = idx
  word_to_idx[START] = last_idx + 1
  word_to_idx[END] = last_idx + 2

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
  train, valid, test, words = FILE_PATHS[dataset]

  build_word_dict(words)
  input_dict, word_to_idx = build_ngrams([train, valid], ngram)
  # TODO: build contexts for test, but only for the ngram before the blank

  V = len(word_to_idx)
  C = np.max(train_output)

  filename = args.dataset + '.hdf5'
  with h5py.File(filename, "w") as f:
    f['train_input'] = train_input
    f['train_output'] = train_output
    if valid:
      f['valid_input'] = valid_input
      f['valid_output'] = valid_output
    if test:
      f['test_input'] = test_input

    f['nwords'] = np.array([V], dtype=np.int32)
    f['nclasses'] = np.array([C], dtype=np.int32)
    f['ngram'] = np.array(ngram, dtype=np.int32)


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))

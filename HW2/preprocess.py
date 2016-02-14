#!/usr/bin/env python

"""Part-Of-Speech Preprocessing
"""

import numpy as np
import h5py
import argparse
import sys
import re
import codecs

PADDING = '__PADDING__'
tag_dict = {}

# Your preprocessing, features construction, and word2vec code.

def clean_line(line):
  '''
  Extracts word and POS from line.
  '''
  info = line.split()

  if len(info) == 0:
    return None, None

  word, pos = str(info[2]), str(info[3])

  # Lower
  word = word.lower()

  # POS
  pos = tag_dict[pos]

  # Replace numbers


  return word, pos

def get_vocab(file_list):
  '''
  Locates top 100k words across dataset and replace rare words with RARE token.
  Reforms word indexes after removing rare words.
  '''
  # NB: there are fewer than 100k unique tokens in train + valid + test
  word_counts = {}
  for filename in file_list:
    if filename:
      with codecs.open(filename, "r", encoding="latin-1") as f:
        print('Extracting vocab from ' + filename + '...')
        for line in f:
          word, pos = clean_line(line)
          if word is None: continue
          if word not in word_counts:
            word_counts[word] = 0
          word_counts[word] += 1

  return word_counts


  # Build word to index dictionary
  for filename in file_list:
    if filename:
      with codecs.open(filename, "r", encoding="latin-1") as f:
        for line in f:
          word, pos = clean_line(line)
          if word not in word_to_idx:
            word_to_idx[word] = idx
            idx += 1
  return word_to_idx

def run_tests():
  '''
  Runs a simple set of tests to ensure vocab extraction and cleaning is working
  properly.
  '''

  # Case
  line = '1 1 Apparently RB'
  assert clean_line(line) == ('apparently', 17)
  
  # Numbers
  line = '1 1 PS4 NNP'
  assert clean_line(line) == ('psNUMBER', 1)

def build_tag_dict(filename):
  with codecs.open(filename, "r", encoding="latin-1") as f:
    for line in f:
      info = line.split()
      abbrev = str(info[0])
      idx = int(info[1])
      tag_dict[abbrev] = idx
      

FILE_PATHS = {"PTB": ("data/train.tags.txt",
            "data/dev.tags.txt",
            "data/test.tags.txt",
            "data/tags.dict")}
args = {}


def main(arguments):
  global args
  parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('dataset', help="Data set",
            type=str)
  args = parser.parse_args(arguments)
  dataset = args.dataset
  if dataset == 'test':
    build_tag_dict('data/tags.dict')
    return run_tests()

  train, valid, test, tags = FILE_PATHS[dataset]
  build_tag_dict(tags)
  vocab = get_vocab([train, valid, test])

  filename = args.dataset + '.hdf5'
  with h5py.File(filename, "w") as f:
    f['train_input'] = train_input
    f['train_output'] = train_output
    if valid:
      f['valid_input'] = valid_input
      f['valid_output'] = valid_output
    if test:
      f['test_input'] = test_input
    f['nfeatures'] = np.array([V], dtype=np.int32)
    f['nclasses'] = np.array([C], dtype=np.int32)


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))

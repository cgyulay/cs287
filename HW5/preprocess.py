#!/usr/bin/env python

'''NER Preprocessing
'''

import numpy as np
import h5py
import argparse
import sys
import re
import codecs

# Preprocessing, feature construction, and GloVe setup

FILE_PATHS = {'CONLL': ('data/train.num.txt',
            'data/dev.num.txt',
            'data/test.num.txt',
            'data/tags.txt')}
args = {}
tag_to_idx = {}
word_to_idx = {}

START = '<s>'
STOP = '</s>'
START_TAG = '<t>'
STOP_TAG = '</t>'
UNKNOWN = '<unk>'

def build_sentences(file_list):
  input_s = {}
  output_s = {}
  for filename in file_list:
    if filename:
      input_s[filename] = []
      output_s[filename] = []
      with codecs.open(filename, 'r', encoding='latin-1') as f:
        print('Building sentences from ' + filename + '...')

        sentence = [word_to_idx[START]]
        tags = [tag_to_idx[START_TAG]]
        longest = 0
        for line in f:
          line = line.split()
          if len(line) == 0: # EOS
            # Add closing word and tag
            sentence.append(word_to_idx[STOP])
            tags.append(tag_to_idx[STOP_TAG])

            input_s[filename].append(sentence)
            output_s[filename].append(tags)
            if len(sentence) > longest: longest = len(sentence)

            sentence = [word_to_idx[START]]
            tags = [tag_to_idx[START_TAG]]
            continue

          # If we got here, we have a valid word in the middle of a sentence
          word = word_to_idx[str(line[2])]
          sentence.append(word)
          if filename != 'data/test.num.txt':
            tag = tag_to_idx[str(line[3])]
            tags.append(tag)

        # Standardize sentence/tag length with padding
        for i in range(len(input_s[filename])):
          input_s[filename][i] = input_s[filename][i] + [word_to_idx[STOP]] \
            * (longest - len(input_s[filename][i]))
          output_s[filename][i] = output_s[filename][i] + [tag_to_idx[STOP_TAG]] \
            * (longest - len(output_s[filename][i]))
  return input_s, output_s

def build_tag_dict(filename):
  idx = -1
  with codecs.open(filename, 'r', encoding='latin-1') as f:
    for line in f:
      info = line.split()
      abbrev = str(info[0])
      idx = int(info[1])
      tag_to_idx[abbrev] = idx
  tag_to_idx[START_TAG] = idx + 1
  tag_to_idx[STOP_TAG] = idx + 2

def build_word_dict(file_list):
  last_idx = 3
  word_to_idx[START] = 1
  word_to_idx[STOP] = 2
  for filename in file_list:
    if filename:
      with codecs.open(filename, 'r', encoding='latin-1') as f:
        for line in f:
          line = line.split()
          if len(line) == 0: continue

          word = str(line[2])
          if word not in word_to_idx:
            word_to_idx[word] = last_idx
            last_idx = last_idx + 1
  print('Built word dict with ' + str(last_idx - 1) + ' entries.')

def build_vector(filename):
  vector_dict = {}

  with codecs.open(filename, 'r', encoding='latin-1') as f:
    for line in f:
      try:
        info = line.split()
        word = info[0]
        vec = [float(x) for x in info[1:]]
      except:
        continue
      vector_dict[word] = vec
  return vector_dict

def main(arguments):
  global args
  parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('dataset', help='Data set',
            type=str)
  args = parser.parse_args(arguments)
  dataset = args.dataset
  train, valid, test, tag_dict = FILE_PATHS[dataset]
  sets = [train, valid, test]

  build_word_dict(sets)
  build_tag_dict(tag_dict)
  V = len(word_to_idx)
  C = len(tag_to_idx)

  input_dict, output_dict = build_sentences(sets)
  train_input = np.array(input_dict[train], dtype=np.int32)
  train_output = np.array(output_dict[train], dtype=np.int32)
  valid_input = np.array(input_dict[valid], dtype=np.int32)
  valid_output = np.array(output_dict[valid], dtype=np.int32)
  test_input = np.array(input_dict[test], dtype=np.int32)

  filename = args.dataset + '.hdf5'
  with h5py.File(filename, 'w') as f:
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

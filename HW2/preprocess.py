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
  Extracts word, POS, and capitalization information from a line.

  Returns the following tuple: (lowered word, pos, case)
  '''
  info = line.split()

  if len(info) == 0:
    return None, None, None

  word, pos = str(info[2]), str(info[3])

  # Record data regarding capitalization
  all_upper = True
  first_upper = False
  any_upper = False
  for i, c in enumerate(word):
    if c.isupper():
      any_upper = True
      if i == 0:
        first_upper = True
    else:
      all_upper = False

  # 0 = all lower
  # 1 = first char upper
  # 2 = any other char upper
  # 3 = all upper
  case = 0
  if all_upper:
    case = 3
  elif any_upper:
    if first_upper:
      case = 1
    else:
      case = 2

  # Lower
  word = word.lower()

  # POS
  pos = tag_dict[pos]

  # Replace numbers
  # If the word is a number and can successfully be converted to float,
  # simply treat it as NUMBER
  if pos == 3:
    try:
      word = float(word)
      word = 'NUMBER'
    except ValueError:
      pass

  # If the word is partially composed of numbers, replace these number
  # substrings with NUMBER
  word = re.sub('(\d+)', 'NUMBER', word)

  return word, pos, case

def create_windows_for_sentences(sentences):
  '''
  Takes a list of sentences comprised of lists of cleaned words and creates a series of
  windows of length dwin.
  '''

  for s in sentences:
    # TODO

def get_vocab(file_list):
  '''
  Locates top 100k words across dataset and replace rare words with RARE token.
  Reforms word indexes after removing rare words.
  '''
  # NB: there are fewer than 100k unique tokens in train + valid + test
  # (< 40k), so we don't need to create a limited dictionary with RARE tokens
  word_to_idx = {}
  sentences = {}
  sentence = []
  idx = 2 # padding = idx 1

  for filename in file_list:
    if filename:
      with codecs.open(filename, "r", encoding="latin-1") as f:
        print('Extracting vocab from ' + filename + '...')
        sentences[filename] = []

        for line in f:
          word, pos, case = clean_line(line)
          if word is not None:
            sentence.append((word, pos, case))
          else:
            sentences[filename].append(sentence)
            sentence = []
            continue
          if word not in word_to_idx:
            word_to_idx[word] = idx
            idx += 1

  return word_to_idx, sentences

def run_tests():
  '''
  Runs a simple set of tests to ensure vocab extraction and cleaning is working
  properly.
  '''
  # tuple = word, pos, case

  # Capitalization
  line = '1 1 Apparently RB'
  assert clean_line(line) == ('apparently', 17, 1)

  line = '1 1 appaRently RB'
  assert clean_line(line) == ('apparently', 17, 2)

  line = '1 1 APPARENTLY RB'
  assert clean_line(line) == ('apparently', 17, 3)
  
  # Numbers
  line = '1 1 PS4 NNP'
  assert clean_line(line) == ('psNUMBER', 1, 1)

  line = '1 1 1.3 CD'
  assert clean_line(line) == ('NUMBER', 3, 0)

  line = '1 1 million CD'
  assert clean_line(line) == ('million', 3, 0)

  print('String cleaning tests pass.')

def build_tag_dict(filename):
  tag_dict['_'] = -1 # For test file
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

  # Dict for POS tags
  build_tag_dict(tags)

  # Get unique word dictionary and cleaned sentences
  word_to_idx, sentences = get_vocab([train, valid, test])

  # Convert sentences to input, input_cap, and output windows
  dwin = 5
  train_input_word_windows, train_input_cap_windows, train_output = \
    create_windows_for_sentences(sentences[train])

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

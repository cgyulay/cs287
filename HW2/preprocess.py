#!/usr/bin/env python

"""Part-Of-Speech Preprocessing
"""

import numpy as np
import h5py
import argparse
import sys
import re
import codecs
import math

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

  # 1 = all lower
  # 2 = first char upper
  # 3 = any other char upper
  # 4 = all upper
  case = 1
  if all_upper:
    case = 4
  elif any_upper:
    if first_upper:
      case = 2
    else:
      case = 3

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

def create_windows_for_sentences(sentences, dwin, word_to_idx):
  '''
  Takes a list of sentences comprised of lists of cleaned words and creates a series of
  windows of length dwin.
  '''

  num_padding = int(math.floor((dwin - 1) / 2))
  padding = [(PADDING, -1, 1)] * num_padding
  total_padding = 2 * num_padding

  input_word_windows = []
  input_cap_windows = []
  output = []

  for s in sentences:
    s = padding + s + padding
    for i in range(len(s) - total_padding):
      word_window = s[i:i + total_padding + 1]

      # word, pos, case
      input_word_windows.append([word_to_idx[w[0]] for w in word_window])
      input_cap_windows.append([w[2] for w in word_window])
      output.append(word_window[num_padding][1])

  return np.array(input_word_windows, dtype=np.int32), \
    np.array(input_cap_windows, dtype=np.int32), \
    np.array(output, dtype=np.int32)

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
  word_to_idx[PADDING] = 1

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
  assert clean_line(line) == ('apparently', 17, 2)

  line = '1 1 appaRently RB'
  assert clean_line(line) == ('apparently', 17, 3)

  line = '1 1 APPARENTLY RB'
  assert clean_line(line) == ('apparently', 17, 4)
  
  # Numbers
  line = '1 1 PS4 NNP'
  assert clean_line(line) == ('psNUMBER', 1, 2)

  line = '1 1 1.3 CD'
  assert clean_line(line) == ('NUMBER', 3, 1)

  line = '1 1 million CD'
  assert clean_line(line) == ('million', 3, 1)

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
    create_windows_for_sentences(sentences[train], dwin, word_to_idx)

  valid_input_word_windows, valid_input_cap_windows, valid_output = \
    create_windows_for_sentences(sentences[valid], dwin, word_to_idx)

  test_input_word_windows, test_input_cap_windows, test_output = \
    create_windows_for_sentences(sentences[test], dwin, word_to_idx)

  V = len(word_to_idx) + 1
  print('Vocab size: {0}'.format(V))

  C = np.max(train_output)

  filename = args.dataset + '.hdf5'
  with h5py.File(filename, "w") as f:
    f['train_input_word_windows'] = train_input_word_windows
    f['train_input_cap_windows'] = train_input_cap_windows
    f['train_output'] = train_output
    if valid:
      f['valid_input_word_windows'] = valid_input_word_windows
      f['valid_input_cap_windows'] = valid_input_cap_windows
      f['valid_output'] = valid_output
    if test:
      f['test_input_word_windows'] = test_input_word_windows
      f['test_input_cap_windows'] = test_input_cap_windows

    f['nwords'] = np.array([V], dtype=np.int32)
    f['nclasses'] = np.array([C], dtype=np.int32)

    # TODO: word embeddings from data/glove.6B.50d.txt.gz
    f['word_embeddings'] = np.array([0], dtype=np.int32)


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))

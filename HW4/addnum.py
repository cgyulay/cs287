# Reads space predictions, and reformats them to be indexed

import codecs

filename = 'training_output/kaggle_copy.txt'
preds = []
with codecs.open(filename, 'r', encoding='latin-1') as f:
  idx = 0
  for line in f:
    if idx > 0:
      preds.append(int(line))
    idx = idx + 1
f.close()

newfile = 'training_output/kaggle_fixed.txt'
with codecs.open(newfile, 'w', encoding='latin-1') as f:
  f.write('ID,Count')
  for i, p in enumerate(preds):
    f.write('\n' + str(i + 1) + ',' + str(p))
f.close()

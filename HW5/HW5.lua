-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-lm', 'hmm', 'classifier to use')
cmd:option('-alpha', 0.01, 'Laplace smoothing coefficient')
cmd:option('-eta', 0.1, 'learning rate')
cmd:option('-nepochs', 15, 'number of training epochs')
cmd:option('-mb', 32, 'minibatch size')

START = 1
STOP = 2
START_TAG = 8
STOP_TAG = 9

----------
-- Misc
----------

-- Helper function that finds the nth index of a given value in a tensor
function find_nth(t, val, n)
  local count = 0
  for i = 1, t:size(1) do
    if t[i] == val then
      count = count + 1
      if count == n then return i end
    end
  end
  return -1
end

-- Helper function that finds the first index of a given value in a tensor
function find_first(t, val)
  return find_nth(t, val, 1)
end

-- Removes excess repeated labels from end of a generic sequence
function chop(seq)
  local label = seq[seq:size(1)]
  local stop = find_first(seq, label)
  return seq[{{1, stop}}]
end

-- Removes start and stop labels from ends of a sequence
function slap(seq)
  return seq[{{2, seq:size(1) - 1}}]
end

-- Computes F score
function f_score(preds, y, beta)
  local cor_pos = 0
  local all_pos = 0
  local tru_pos = 0
  for i = 1, preds:size(1) do
    if preds[i] > 1 then
      all_pos = all_pos + 1
      if preds[i] == y[i] then cor_pos = cor_pos + 1 end
    end
    if y[i] > 1 then tru_pos = tru_pos + 1 end
  end

  local p = cor_pos / all_pos
  local r = cor_pos / tru_pos

  -- If all_pos or tru_pos are 0, then effectively we have 0/0
  -- which should be mapped to a precision or recall of 1
  if all_pos == 0 then p = 1 end
  if tru_pos == 0 then r = 1 end
  
  local b2 = beta * beta
  if b2 * p + r == 0 then return 0 end
  return (1 + b2) * ((p * r) / (b2 * p + r))
end

----------
-- Search
----------

-- Viterbi algorithm
-- observations: a sequence of observations, represented as integers
-- logscore: the edge scoring function over classes and observations in a
-- history-based model
function viterbi(observations, logscore, emission)
  local initial = torch.zeros(nclasses, 1)
  initial[START_TAG] = 1 -- Initial transition dist (always begins with start)
  initial:log()

  local n = observations:size(1)
  local max_table = torch.Tensor(n, nclasses)
  local backpointer_table = torch.Tensor(n, nclasses)

  -- First timestep
  -- Initial most likely paths are the initial state distribution
  local maxes, backpointers = (initial + emission[observations[1]]):max(2)
  max_table[1] = maxes

  -- Remaining timesteps
  for i = 2, n do
    local y = logscore(observations, i)
    local scores = y + maxes:view(1, nclasses):expand(nclasses, nclasses)
    maxes, backpointers = scores:max(2)
    max_table[i] = maxes
    backpointer_table[i] = backpointers
  end

  -- Use backpointers to recover optimal path
  local classes = torch.Tensor(n)
  maxes, classes[n] = maxes:max(1)
  for i = n, 2, -1 do
    classes[i-1] = backpointer_table[{i, classes[i]}]
  end

  return classes:int()
end

----------
-- HMM
----------

-- Build simple HMM using the class to class transition distribution combined
-- with the class to word emission distribution
function hmm()
  print('Building HMM...')
  local transition = torch.zeros(nclasses, nclasses) -- p(y_{i}|y_{i-1})
  local emission = torch.zeros(nwords, nclasses) -- p(x_{i}|y_{i})

  for i = 1, train_x:size(1) do
    local x = chop(train_x[i])
    local y = chop(train_y[i])

    for j = 1, x:size(1) do
      local word = x[j]
      local class = y[j]
      emission[word][class] = emission[word][class] + 1

      if j < x:size(1) then
        transition[y[j+1]][class] = transition[y[j+1]][class] + 1
      end
    end
  end

  -- Normalize and log() transition probabilities
  transition = transition + alpha
  local sums = torch.sum(transition, 1)
  sums = sums:expand(nclasses, nclasses)
  transition:cdiv(sums):log()

  -- Normalize and log() emission probabilities
  emission = emission + alpha
  sums = torch.sum(emission, 1)
  sums = sums:expand(nwords, nclasses)
  emission:cdiv(sums):log()

  -- Log-scores of transition and emission
  -- i: timestep for the computed score
  function hmm_score(observations, i)
    local observation_emission = emission[observations[i]]:view(nclasses, 1):
      expand(nclasses, nclasses)
    return observation_emission + transition
  end

  -- Predict optimal sequences in validation using viterbi
  print('Computing F1 score on validation...')
  local totalf = 0
  for i = 1, valid_x:size(1) do
    local x = chop(valid_x[i])
    local y = slap(chop(valid_y[i])) -- SLAP CHOP!!!
    local classes = slap(viterbi(x, hmm_score, emission))
    local f = f_score(classes, y, 1) -- F1
    totalf = totalf + f
  end

  local validf = totalf / valid_x:size(1)
  print('Validation F1 score: ' .. validf .. '.')
end

----------
-- MEMM
----------

function memm()
  
end

----------
-- Perceptron
----------

function perceptron()
  print('TODO perceptron')
end

function main() 
  -- Parse input params
  opt = cmd:parse(arg)
  datafile = opt.datafile
  lm = opt.lm
  alpha = opt.alpha
  eta = opt.eta
  nepochs = opt.nepochs
  batch_size = opt.mb

  local f = hdf5.open(opt.datafile, 'r')
  nclasses = f:read('nclasses'):all():long()[1]
  nfeatures = f:read('nfeatures'):all():long()[1]
  nwords = f:read('nwords'):all():long()[1]

   -- Split training, validation, test data
  train_x = f:read('train_input'):all()
  train_y = f:read('train_output'):all()
  valid_x = f:read('valid_input'):all()
  valid_y = f:read('valid_output'):all()
  test_x = f:read('test_input'):all()

  if lm == 'hmm' then
    hmm()
  elseif lm == 'memm' then
    memm()
  else
    perceptron()
  end
end

main()

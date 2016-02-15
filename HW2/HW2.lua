-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'PTB.hdf5', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use')
cmd:option('-alpha', 1.0, 'Laplace smoothing coefficient')
cmd:option('-lr', 0.01, 'learning rate')
cmd:option('-lambda', 0.05, 'l2 regularization coefficient')
cmd:option('-n_epochs', 3, 'number of training epochs')
cmd:option('-m', 20, 'minibatch size')

function naive_bayes()
  print('Building naive bayes model...')

  -- Generate prior for each part of speech
  local double_output = train_output:double()
  local p_y = torch.histc(double_output, nclasses)
  p_y:div(torch.sum(p_y))

  -- Build multinomial distribution for each relative window position
  -- ie: p(word at position 1|y)
  --     p(capitalization at position 1|y)

  -- Record number occurrences of each word at each position given a class
  print('Building p(word at i|y), p(cap at i|y)...')
  ncaps = train_input_cap_windows:max()
  local word_occurrences = torch.zeros(nclasses, dwin, nwords)
  local cap_occurrences = torch.zeros(nclasses, dwin, ncaps)

  for i = 1, train_output:size(1) do
    local y = train_output[i]

    local word_window = train_input_word_windows[i]
    local cap_window = train_input_cap_windows[i]
    for j = 1, dwin do
      local w = word_window[j]
      local c = cap_window[j]

      word_occurrences[y][j][w] = word_occurrences[y][j][w] + 1
      cap_occurrences[y][j][c] = cap_occurrences[y][j][c] + 1
    end
  end

  -- Add smoothing to account for words/caps not appearing in a position/class
  word_occurrences:add(alpha)
  cap_occurrences:add(alpha)

  -- Normalize to 1
  for y = 1, nclasses do
    for p = 1, dwin do
      -- All word/cap occurrences at position p in class y
      local w_sum = word_occurrences[y][p]:sum()
      local c_sum = word_occurrences[y][p]:sum()

      -- Divide by sum across nwords/ncaps
      word_occurrences:select(1, y):select(1, p):div(w_sum)
      cap_occurrences:select(1, y):select(1, p):div(c_sum)
    end
  end

  print('Running naive bayes on validation set...')

  function nb_predict(word_windows, cap_windows)
    local pred = torch.IntTensor(word_windows:size(1))
    for i = 1, word_windows:size(1) do
      local word_window = word_windows[i]
      local cap_window = cap_windows[i]

      local p_y_hat = torch.zeros(nclasses)
      for y = 1, nclasses do
        p_y_hat[y] = p_y[y]

        -- Multiply p_y_hat by p(word at j|y), p(cap at j|y)
        for j = 1, dwin do
          w = word_window[j]
          c = cap_window[j]

          p_y_hat:mul(word_occurrences[y][j][w])
          p_y_hat:mul(cap_occurrences[y][j][c])
        end
      end

      p_y_hat:div(p_y_hat:sum())
      val, prediction = torch.max(p_y_hat, 1)

      pred[i] = prediction
    end
    return pred
  end

  -- Generate predictions on validation
  local pred = nb_predict(valid_input_word_windows, valid_input_cap_windows)
  pred = pred:eq(valid_output):double()
  print(pred)
  local accuracy = torch.mean(pred) -- torch.histc(pred,2)[2] / pred:size(1)

  print('Validation accuracy: ' .. accuracy .. '.')

  -- print('Running naive bayes on test set...')
  -- pred = nb_predict(test_x, word_occurrences, p_y)
  -- writeToFile(pred)
end

function main() 
   -- Parse input params
   opt = cmd:parse(arg)
   classifier = opt.classifier
   alpha = opt.alpha
   lr = opt.lr
   lambda = opt.lambda
   n_epochs = opt.n_epochs
   batch_size = opt.m

   local f = hdf5.open(opt.datafile, 'r')
   nwords = f:read('nwords'):all():long()[1]
   nclasses = f:read('nclasses'):all():long()[1]
   dwin = f:read('dwin'):all():long()[1]

   -- Training, valid, and test windows
   train_input_word_windows = f:read('train_input_word_windows'):all()
   train_input_cap_windows = f:read('train_input_cap_windows'):all()
   train_output = f:read('train_output'):all()

   valid_input_word_windows = f:read('valid_input_word_windows'):all()
   valid_input_cap_windows = f:read('valid_input_cap_windows'):all()
   valid_output = f:read('valid_output'):all()

   test_input_word_windows = f:read('test_input_word_windows'):all()
   test_input_cap_windows = f:read('test_input_cap_windows'):all()

   -- local W = torch.DoubleTensor(nclasses, nwords)
   -- local b = torch.DoubleTensor(nclasses)

   -- Train.
  if opt.classifier == 'nb' then
    naive_bayes()
  elseif opt.classifier == 'lr' then
    print('run logistic regression')
  end

   -- Test.
end

main()

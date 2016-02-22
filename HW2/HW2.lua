-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'PTB.hdf5', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use')
cmd:option('-alpha', 1.0, 'Laplace smoothing coefficient')
cmd:option('-eta', 0.01, 'learning rate')
cmd:option('-lambda', 0.05, 'l2 regularization coefficient')
cmd:option('-n_epochs', 3, 'number of training epochs')
cmd:option('-m', 32, 'minibatch size')

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
      local c_sum = cap_occurrences[y][p]:sum()

      -- Divide by sum across nwords/ncaps
      word_occurrences:select(1, y):select(1, p):div(w_sum)
      cap_occurrences:select(1, y):select(1, p):div(c_sum)
    end
  end

  print('Running naive bayes on validation set...')

  function predict(word_windows, cap_windows)
    local pred = torch.IntTensor(word_windows:size(1))
    for i = 1, word_windows:size(1) do
      local word_window = word_windows[i]
      local cap_window = cap_windows[i]

      local p_y_hat = torch.zeros(nclasses)
      for y = 1, nclasses do
        p_y_hat[y] = p_y[y]

        -- Multiply p_y_hat by p(word at j|y) and p(cap at j|y)
        for j = 1, dwin do
          w = word_window[j]
          c = cap_window[j]

          p_y_hat[y] = p_y_hat[y] *  word_occurrences[y][j][w]
          p_y_hat[y] = p_y_hat[y] *  cap_occurrences[y][j][c]
        end
      end

      p_y_hat:div(p_y_hat:sum())
      val, prediction = torch.max(p_y_hat, 1)

      pred[i] = prediction
    end
    return pred
  end

  -- Generate predictions on validation
  local pred = predict(valid_input)

  pred = pred:eq(valid_output):double()
  local accuracy = torch.mean(pred) * 100

  print('Validation accuracy: ' .. accuracy .. '.')

  -- print('Running naive bayes on test set...')
  -- pred = predict(test_x, word_occurrences, p_y)
  -- writeToFile(pred)
end

function model(structure)
  local embedding_size = 50
  local din = dwin * (embedding_size + ncaps)
  local dout = nclasses
  local dhid = 200

  local model = nn.Sequential()

  if structure == 'lr' then
    print('Building logistic regression model...')

    local sparseW_word = nn.LookupTable(nwords, nclasses)
    local W_word = nn.Sequential():add(sparseW_word):add(nn.Sum(2))

    local sparseW_cap = nn.LookupTable(ncaps, nclasses)
    local W_cap = nn.Sequential():add(sparseW_cap):add(nn.Sum(2))
    
    local par = nn.ParallelTable()
    par:add(W_word) -- first child
    par:add(W_cap) -- second child

    local logsoftmax = nn.LogSoftMax()

    model:add(par):add(nn.CAddTable()):add(logsoftmax)
  elseif structure == 'mlp' then
    print('Building multilayer perceptron model...')

    -- Use two parallel sequentials to support LookupTables with Reshape
    -- Word LookupTable
    local word_lookup = nn.Sequential()
    local w = nn.LookupTable(nwords, embedding_size) -- Random embed init (types x embedding size)
    local w_reshape = nn.Reshape(dwin * embedding_size)
    word_lookup:add(w):add(w_reshape)

    -- Cap LookupTable
    local cap_lookup = nn.Sequential()
    local c = nn.LookupTable(dwin, ncaps)
    local c_reshape = nn.Reshape(dwin * ncaps)
    cap_lookup:add(c):add(c_reshape)

    local par = nn.ParallelTable()
    par:add(word_lookup)
    par:add(cap_lookup)
    model:add(par)
    model:add(nn.JoinTable(2))

    model:add(nn.Linear(din, dhid))
    model:add(nn.HardTanh())
    model:add(nn.Linear(dhid, dout))
    model:add(nn.LogSoftMax())
  else
    print('Classifier incorrectly specified, bailing out.')
    return
  end

  local nll = nn.ClassNLLCriterion()
  nll.sizeAverage = false

  local params, gradParams = model:getParameters()

  function train(e)
    -- Package selected dataset into minibatches
    local selected_x_ww = valid_input_word_windows
    local selected_x_cw = valid_input_cap_windows
    local selected_y = valid_output
    local n_train_batches = math.floor(selected_x_ww:size(1) / batch_size) - 1

    print('\nBeginning epoch ' .. e .. ' training: ' .. n_train_batches .. ' minibatches of size ' .. batch_size .. '.')
    for i = 1, n_train_batches do
      -- local input = torch.DoubleTensor(batch_size, dwin)
      -- local output = torch.IntTensor(batch_size)
      -- local batch_start = torch.random(1, selected_x:size(1) - batch_size)
      local batch_start = (i - 1) * batch_size + 1
      local batch_end = batch_start + batch_size - 1

      local range = torch.range(batch_start, batch_end):long()
      local x_ww = selected_x_ww:index(1, range)
      local x_cw = selected_x_cw:index(1, range)
      local y = selected_y:index(1, range)

      -- Compute forward and backward pass (predictions + gradient updates)
      function run_minibatch(p)
        if params ~= p then
          params:copy(p)
        end

        -- Accumulate gradients from scratch each minibatch
        gradParams:zero()

        -- Forward pass
        local preds = model:forward({x_ww, x_cw})
        local loss = nll:forward(preds, y)

        if i % 2000 == 0 then
          print('Completed ' .. i .. ' minibatches.')
          -- print('Loss after ' .. batch_end .. ' examples: ' .. loss)
        end

        -- Backward pass
        local dLdpreds = nll:backward(preds, y)
        model:backward(preds, dLdpreds)

        return loss, gradParams
      end

      options = {
        learningRate = eta
      }

      -- Use optim package for minibatch sgd
      optim.sgd(run_minibatch, params, options)
    end
  end

  function test(x_ww, x_cw, y)
    local preds = model:forward({x_ww, x_cw})
    local max, yhat = preds:max(2)
    
    local correct = yhat:int():eq(y):double()
    return torch.mean(correct) * 100
  end

  function valid_acc()
    return test(valid_input_word_windows, valid_input_cap_windows, valid_output)
  end

  function train_acc()
    return test(train_input_word_windows, train_input_cap_windows, train_output)
  end

  print('Validation accuracy before training: ' .. valid_acc() .. ' %.')
  print('Beginning training...')
  for i = 1, n_epochs do
    local timer = torch.Timer()
    train(i)
    print('Epoch ' .. i .. ' training completed in ' .. timer:time().real .. ' seconds.')
    print('Validation accuracy after epoch ' .. i .. ': ' .. valid_acc() .. ' %.')
  end
end

function main() 
  -- Parse input params
  opt = cmd:parse(arg)
  classifier = opt.classifier
  alpha = opt.alpha
  eta = opt.eta
  lambda = opt.lambda
  n_epochs = opt.n_epochs
  batch_size = opt.m

  local f = hdf5.open(opt.datafile, 'r')

  -- Training, valid, and test windows
  train_input_word_windows = f:read('train_input_word_windows'):all()
  train_input_cap_windows = f:read('train_input_cap_windows'):all()
  train_output = f:read('train_output'):all()

  valid_input_word_windows = f:read('valid_input_word_windows'):all()
  valid_input_cap_windows = f:read('valid_input_cap_windows'):all()
  valid_output = f:read('valid_output'):all()

  test_input_word_windows = f:read('test_input_word_windows'):all()
  test_input_cap_windows = f:read('test_input_cap_windows'):all()

  -- Useful values across models
  nwords = f:read('nwords'):all():long()[1]
  nclasses = f:read('nclasses'):all():long()[1]
  dwin = f:read('dwin'):all():long()[1]
  ncaps = train_input_cap_windows:max()

  function combine(ww, cw)
    -- Concat word and cap windows into single tensor
    -- d = torch.DoubleTensor(ww:size(1), ww:size(2) + cw:size(2))
    -- d:narrow(2, 1, ww:size(2)):copy(ww)
    -- d:narrow(2, ww:size(2) + 1, cw:size(2)):copy(cw)

    -- Prep word and cap windows for parallel table
    d = {}
    for i = 1, ww:size(1) do
      d[i] = {ww[i], cw[i]}
    end

    return d
  end

  -- train_input = combine(train_input_word_windows, train_input_cap_windows)
  -- valid_input = combine(valid_input_word_windows, valid_input_cap_windows)
  -- test_input = combine(test_input_word_windows, test_input_cap_windows)

   -- Run models
  if opt.classifier == 'nb' then
    naive_bayes()
  else
    model(opt.classifier)
  end
end

main()

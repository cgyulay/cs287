-- Only requirement allowed
require("hdf5")

cmd = torch.CmdLine()

-- Cmd Args / Hyperparameters
cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use')
cmd:option('-alpha', 1.0, 'Laplace smoothing coefficient')
cmd:option('-lr', 0.01, 'learning rate')
cmd:option('-lambda', 0.01, 'l2 regularization coefficient')
cmd:option('-n_epochs', 3, 'number of training epochs')
cmd:option('-m', 20, 'minibatch size')
-- Flag, default, description

function nb_predict(set, word_occurrences, p_y)
  local pred = torch.IntTensor(set:size(1))
  for e = 1, set:size(1) do
    local example = set[e]
    local p_y_hat = torch.zeros(nclasses)
    for i = 1, nclasses do
      p_y_hat[i] = p_y[i]

      example:apply(function(w)
        if w ~= 1 then
          p_y_hat[i] = p_y_hat[i] * word_occurrences[i][w]
        end
      end)
    end
    p_y_hat = p_y_hat / torch.sum(p_y_hat)
    val, prediction = torch.max(p_y_hat, 1)

    pred[e] = prediction
  end

  return pred
end

function naive_bayes()
  print('Building naive bayes model...')

  -- Generate prior for each class
  local double_output = train_y:double()
  local p_y = torch.histc(double_output, nclasses)
  p_y = p_y / torch.sum(p_y)
  print(p_y)

  -- Record number occurrences of each word in each class
  print('Building p(word|class)...')
  local word_occurrences = torch.zeros(nclasses, nfeatures)
  for i = 1, train_x:size(1) do
    local class = train_y[i]

    for j = 1, train_x:size(2) do
      local w = train_x[i][j]
      word_occurrences[class][w] = word_occurrences[class][w] + 1
    end
  end

  -- Add smoothing to account for words not appearing in a class
  word_occurrences:add(alpha)

  -- Maintain indexes but ignore padding, and renormalize
  for i = 1, nclasses do
    word_occurrences[i][1] = 0
  end
  word_occurrences:renorm(1, 1, 1)

  print('Running naive bayes on validation set...')

  -- Generate predictions on validation
  local pred = nb_predict(valid_x, word_occurrences, p_y)
  pred = pred:eq(valid_y):double()
  local accuracy = torch.histc(pred,2)[2] / pred:size(1)

  print('Validation accuracy: ' .. accuracy .. '.')

  print('Running naive bayes on test set...')
  pred = nb_predict(test_x, word_occurrences, p_y)
  writeToFile(pred)
end

function sparsify(dense)
  -- Loops are prohibitively slow...unusuable for whole dataset
  -- TODO: redo with torch:scatter
  sparse = torch.zeros(dense:size(1), nfeatures)

  for i = 1, dense:size(1) do
    for j = 1, dense:size(2) do
      idx = dense[i][j]
      -- print(idx)
      if idx ~= 1 then
        sparse[i][idx] = sparse[i][idx] + 1
      end
    end

    -- if i % 5000 == 0 then
    --   print('Sparsified ' .. i .. ' examples.')
    -- end
  end

  return sparse
end

function onehot(y, nclasses)
  local y_onehot = torch.zeros(nclasses, y:size(2))
  return y_onehot:scatter(1, y:long(), 1)
end

function softmax(x)
  s1 = x:size(1)
  s2 = x:size(2)
  max = torch.max(x, 1):expand(s1, s2)
  e_x = torch.exp(torch.csub(x, max)) -- + max
  log_exp = torch.expand(torch.log(torch.sum(e_x, 1)), s1, s2) + max
  -- soft = torch.exp(torch.csub(x, log_exp)) -- p(y|x)
  soft = torch.csub(x, log_exp) -- LogSoftMax

  -- Enforce normalization
  -- soft = torch.cdiv(soft, torch.sum(soft, 1):expand(s1, s2))

  return soft
end

function sigmoid(x)
  return torch.ones(x:size()):cdiv(torch.exp(-x) + 1.0)
end

function cross_entropy_loss(py_x, y_onehot)
  -- Negative log probability of each correct class
  y = torch.cmul(py_x, y_onehot):sum(1)
  return -torch.log(y):sum()
end

function logistic_regression()
  print('Building logistic regression model...')

  -- Create sparse representation of training/validation data
  -- For now, select only first n examples
  -- print('Converting training data to sparse format...')
  -- local n_examples = 75000 -- train_x:size(1)
  -- local sparse_train_x = sparsify(train_x:index(1, torch.range(1, n_examples):long()))

  local n_train_batches = math.floor(train_x:size(1) / batch_size) - 1

  print('Beginning training...')

  local W = torch.DoubleTensor(nfeatures, nclasses)
  local b = torch.DoubleTensor(nclasses)
  -- b = torch.expand(b:resize(nclasses,1), nclasses, batch_size) -- For minibatch

  -- Calculates accuracy on (pred, label) pairs
  function accuracy(x, y)
    local z = W:t() * x
    local py_x = softmax(z)
    local max, pred = py_x:max(1)
    return pred:int():resize(pred:size(2)):eq(y):double():mean()
  end

  -- Start with an initial benchmark for validation accuracy
  local sparse_valid_x = sparsify(valid_x):t()
  acc = accuracy(sparse_valid_x, valid_y)
  print('Untrained validation accuracy: ' .. acc)


  -- TODO:
  -- Fix bias weights and gradient
  -- Implement training over n_epochs
  -- Timing check each epoch on train and valid
  -- Figure out how loss could go below 0 (hint: it shouldn't)
  -- Implement hinge loss and separate out sgd code
  
  -- Pass over full training dataset n_epochs times
  for i = 1, n_epochs do
    -- Minibatch forward pass and gradient calculation
    print('Beginning epoch ' .. i .. ' training...')
    for j = 0, n_train_batches do

      -- Dim 2 is batch
      -- Sample in order from prepopulated sparse dataset
      -- local batch_start = j * batch_size + 1
      -- local batch_end = (j + 1) * batch_size
      -- local x = sparse_train_x:index(1, torch.range(batch_start, batch_end):long()):t()
      -- local y = train_y:index(1, torch.range(batch_start, batch_end):long()):resize(1, batch_size)
      -- local y_onehot = onehot(y, nclasses)

      -- Sample random batch and sparsify on the fly
      local batch_start = torch.random(1, train_x:size(1) - batch_size)
      local batch_end = batch_start + batch_size - 1
      local x = sparsify(train_x:index(1, torch.range(batch_start, batch_end):long())):t()
      local y = train_y:index(1, torch.range(batch_start, batch_end):long()):resize(1, batch_size)
      local y_onehot = onehot(y, nclasses)

      local z = W:t() * x -- + b -- + torch.expand(b:resize(nclasses, 1), nclasses, batch_size)
      local log_soft = softmax(z)
      local py_x = torch.exp(log_soft)
      
      -- l2 regularization
      local l2 = torch.cmul(W, W):sum() * lambda / 2.0

      -- Loss fn
      local loss = cross_entropy_loss(py_x + l2, y_onehot)

      if j % 1000 == 0 then
        print('Loss after ' .. j .. ' minibatches: ' .. loss)
      end

      -- Calculate grads and update weights
      -- py_x but subtract 1 from correct class
      -- size nclasses x batch_size
      local dL_dz = torch.csub(torch.DoubleTensor(py_x:size()):copy(py_x), y_onehot)
      
      local W_grad = x * dL_dz:t()
      -- local b_grad = torch.expand(torch.mean(dL_dz, 2), nclasses, batch_size)

      assert(W_grad:size(1) == W:size(1))
      assert(W_grad:size(2) == W:size(2))
      -- assert(b_grad:size(1) == b:size(1))
      -- assert(b_grad:size(2) == b:size(2))

      -- Update weights
      local decay = (1 - (lr * lambda) / 10.0)
      W = (W * decay) - (W_grad * lr)
      -- b = (b * decay) - (b_grad * lr)
    end -- End minibatch sgd

    print('Epoch ' .. i .. ' training complete!')
    print('Evaluating accuracy on training subset and validation set...')

    local subset_size = 50000
    local subset_start = torch.random(1, train_x:size(1) - subset_size)
    local subset_end = subset_start + subset_size - 1
    local sparse_train_subset_x = sparsify(train_x:index(1, torch.range(subset_start, subset_end):long())):t()
    local train_subset_y = train_y:index(1, torch.range(subset_start, subset_end):long())
    local acc = accuracy(sparse_train_subset_x, train_subset_y)
    print('Training subset accuracy: ' .. acc)

    local sparse_valid_x = sparsify(valid_x):t()
    acc = accuracy(sparse_valid_x, valid_y)
    print('Validation accuracy: ' .. acc)
  end -- End epoch evaluation

  -- Test predictions for Kaggle
  local sparse_test_x = sparsify(test_x):t()
  local z = W:t() * sparse_test_x
  local py_x = softmax(z)
  local max, pred = py_x:max(1)
  writeToFile(pred:int():resize(pred:size(2)))

  print('Logistic regression model training complete!')
end

function linear_svm()
  print('Training linear svm...')
end

function main()
  -- Parse input params
  opt = cmd:parse(arg)
  alpha = opt.alpha
  lr = opt.lr
  lambda = opt.lambda
  n_epochs = opt.n_epochs
  batch_size = opt.m

  local f = hdf5.open(opt.datafile, 'r')
  nclasses = f:read('nclasses'):all():long()[1]
  nfeatures = f:read('nfeatures'):all():long()[1]

  -- Split training and validation data
  train_x = f:read('train_input'):all()
  train_y = f:read('train_output'):all()
  valid_x = f:read('valid_input'):all()
  valid_y = f:read('valid_output'):all()
  test_x = f:read('test_input'):all()
  -- 1 = padding
  -- Each number thereafter corresponds to an appearance of a vocabulary word
  -- indicated by that index

  -- Train
  if opt.classifier == 'nb' then
    naive_bayes()
  elseif opt.classifier == 'lr' then
    logistic_regression()
  elseif opt.classifier == 'svm' then
    linear_svm()
  elseif opt.classifier == 'nn' then
    multilayer_logistic_regression()
  end

  -- Test
end

-- Writing to file
function writeToFile(predictions)
  local f = torch.DiskFile('predictions.txt', 'w')
  f:writeString('ID,Category\n')
  local id = 1

  for i = 1, predictions:size(1) do
    f:writeString(id .. ',' .. predictions[i] .. '\n')
    id = id + 1
  end
  f:close()
end

main()

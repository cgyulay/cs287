-- Only requirement allowed
require("hdf5")
require("gnuplot")

cmd = torch.CmdLine()

-- Cmd Args / Hyperparameters
cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use')
cmd:option('-alpha', 1.0, 'Laplace smoothing coefficient')
cmd:option('-lr', 0.01, 'learning rate')
cmd:option('-lambda', 0.05, 'l2 regularization coefficient')
cmd:option('-n_epochs', 3, 'number of training epochs')
cmd:option('-m', 20, 'minibatch size')
cmd:option('-kfold', 10, 'number of k-folds')
-- Flag, default, description

function nb_predict(set, word_occurrences, p_y)

  -- initialize tensor for predictions
  local pred = torch.IntTensor(set:size(1))
  for e = 1, set:size(1) do
    local example = set[e]
    local p_y_hat = torch.zeros(nclasses)
    for i = 1, nclasses do
      -- set prior probability
      p_y_hat[i] = p_y[i]

      -- multiply by conditional if not space (1)
      example:apply(function(w)
        if w ~= 1 then
          p_y_hat[i] = p_y_hat[i] * word_occurrences[i][w]
        end
      end)
    end
    -- normalize
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
  -- intialize sparse matrix
  sparse = torch.zeros(dense:size(1), nfeatures)

  -- Loops are prohibitively slow...unusuable for whole dataset
  for i = 1, dense:size(1) do
    for j = 1, dense:size(2) do
      idx = dense[i][j]

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

-- return one-hot vector of scalar classes
function onehot(y, nclasses)
  local y_onehot = torch.zeros(nclasses, y:size(2))
  return y_onehot:scatter(1, y:long(), 1)
end

function softmax(x)
  s1 = x:size(1)
  s2 = x:size(2)

  -- M
  max = torch.max(x, 1):expand(s1, s2)

  e_x = torch.exp(torch.csub(x, max)) -- + max
  log_exp = torch.expand(torch.log(torch.sum(e_x, 1)), s1, s2) + max

  -- LogSoftMax
  soft = torch.csub(x, log_exp) 

  -- Enforce normalization
  -- soft = torch.cdiv(soft, torch.sum(soft, 1):expand(s1, s2))

  return soft
end

function cross_entropy_loss(py_x, y_onehot, L2)
  -- calculate gradient using the predictions
  local dL_dz = torch.csub(torch.DoubleTensor(py_x:size()):copy(py_x), y_onehot)

  -- Negative log probability of each correct class
  local y = torch.cmul(py_x+L2, y_onehot):sum(1)
  local loss = -torch.log(y)

  return dL_dz, loss
end

function hinge_loss(z, y_onehot, L2)

  local y_true = torch.Tensor(y_onehot:size()):copy(y_onehot)

  -- find true value and its index 
  true_vals, y_true_index = torch.max(y_true, 1)

  -- find next highest value and its index
  -- subtract 1000 from correct class so it doesn't show up as max
  y_true:mul(1000)
  y_pred, y_pred_index = torch.max(torch.Tensor(z:size()):copy(z):csub(y_true), 1)


  local dL_dz = torch.zeros(nclasses, batch_size)

  for i = 1, dL_dz:size(2) do
    -- find index spot of true val and predicted val
    local curr_yp = y_pred_index[1][i]
    local curr_yt = y_true_index[1][i]

    -- if (yhat_c - yhat_c') > 1 then do nothing
    if true_vals[1][i] - y_pred[1][i] > 1 then
      -- print("Minimum Difference Reached: ".. true_vals[1][i] - y_pred[1][i])
      n = 0
    -- else yhat_c = -1 and yhat_c' = 1
    else
      dL_dz[curr_yp][i] = 1
      dL_dz[curr_yt][i] = -1
    end
  end

  local loss = torch.cmax(y_pred:csub(true_vals):add(1), 0)

  return dL_dz, loss
end

-- calculate accuracy of Wx + b using y
function accuracy(x, y, W, b)
  local z = W:t() * x
  local py_x = softmax(z)
  local max, pred = py_x:max(1)
  return pred:int():resize(pred:size(2)):eq(y):double():mean()
end

function kfolds_logistic(loss_fn, k)
  print("Beginning K-fold Cross-Validation")

  -- calculate size of each k-fold group
  local subset_size = train_x:size(1)/k

  -- randomly sample k groups from training data
  order = torch.randperm(train_x:size(1))
  subsets = torch.range(1,order:size(1)-subset_size, subset_size)


  n_train_batches = 1000

  -- intialize aggregate weights
  W_agg = torch.DoubleTensor(nfeatures,nclasses):fill(0)
  b_agg = torch.DoubleTensor(nclasses):fill(0)

  for i=0, subsets:size(1)-1 do
    print("Current k-fold iteration: " .. i+1 .."/"..k)

    -- pick one validation set
    local valid_set = i

    -- initialize weights for current iteration
    local W = torch.DoubleTensor(nfeatures, nclasses):fill(0)
    local b = torch.DoubleTensor(nclasses)
    b = torch.expand(b:resize(nclasses,1), nclasses, batch_size):fill(0) -- For minibatch

    for j = 0, subsets:size(1) do
      if j == valid_set then
        n=0
      else
        print("  Current Fold: " .. j+1 .. "/" ..k)
        for k = 0, n_train_batches do

          -- randomly select mini-batch from current group k
          local batch_start = torch.random(i*subset_size+1, (i+1)*subset_size)
          local batch_end = math.min((batch_start + batch_size - 1), order:size(1))
          -- print(batch_start .." - " .. batch_end)
          batch = torch.Tensor(order[{{batch_start,batch_end}}])

          local x_rand = sparsify(train_x:index(1, batch:long())):t()
          local y_rand = train_y:index(1, batch:long()):resize(1, batch_size)
          local y_onehot = onehot(y_rand, nclasses)
          local z = W:t()*x_rand + b 

          -- l2 regularization
          local l2 = torch.cmul(W, W):sum() * lambda / 2.

          -- calculate partial derivatives and loss
          if loss_fn == 'cross_entropy' then
            local py_x = torch.exp(softmax(z))
            dL_dz, loss = cross_entropy_loss(py_x, y_onehot, l2)
          elseif loss_fn == 'hinge' then
            dL_dz, loss = hinge_loss(z, y_onehot, l2)
          end

          -- if k % 100 == 0 then
          --   print('    Loss after ' .. k .. ' minibatches: ' .. loss:sum())
          -- end

          -- calculate gradients
          local W_grad = x_rand * dL_dz:t()
          local b_grad = torch.expand(torch.mean(dL_dz, 2), nclasses, batch_size)

          -- update weights
          local decay = (1 - (lr * lambda) / 10.0)
          W = (W * decay) - (W_grad * lr)
          b = (b * decay) - (b_grad * lr)
        end
      end      
    end


    print("K-fold Iteration " .. i+1 .. " Complete")

    -- calculate accuracy on validation set
    local valid_batch = torch.Tensor(order[{{(1+valid_set*subset_size),((valid_set+1)*subset_size)}}])
    local valid_x = sparsify(train_x:index(1, valid_batch:long())):t()
    local valid_y = train_y:index(1, valid_batch:long())
    local acc = accuracy(valid_x, valid_y, W, b)
    print("Iteration " .. i+1 .. " Accuracy Score: " .. acc)

    -- -- add k-fold iteration weights to aggregate weights
    -- local fold_valid_x = sparsify(train_x:index(1, valid_batch:long())):t()
    -- local fold_valid_y = train_y:index(1, valid_batch:long())
    -- local acc = accuracy(fold_valid_x, fold_valid_y, W)
    -- print("Iteration " .. i+1 .. " accuracy: " .. acc)

    local sparse_valid_x = sparsify(valid_x):t()
    acc = accuracy(sparse_valid_x, valid_y, W)
    print('Held out validation accuracy: ' .. acc)

    W_agg = W_agg + W:mul(1/subsets:size(1))
    b_agg = b_agg + b[{{}, 2}]:mul(1/subsets:size(1))

  end

  -- test predictions on Kaggle
  local sparse_test_x = sparsify(test_x):t()
  local z = W_agg:t() * sparse_test_x + b_agg
  local py_x = softmax(z)
  local max, pred = py_x:max(1)
  writeToFile(pred:int():resize(pred:size(2)))
end

function logistic_regression(loss_fn)
  print('Building logistic regression model...')

  -- Create sparse representation of training/validation data
  -- For now, select only first n examples
  -- print('Converting training data to sparse format...')
  -- local n_examples = 75000 -- train_x:size(1)
  -- local sparse_train_x = sparsify(train_x:index(1, torch.range(1, n_examples):long()))

  -- For plotting progress over epoch
  local n_train_batches = math.floor(train_x:size(1) / batch_size) - 1
  local train_plot_data = torch.DoubleTensor(n_epochs + 1, 2)
  local valid_plot_data = torch.DoubleTensor(n_epochs + 1, 2)
  train_plot_data[{ {},1 }] = torch.range(0, n_epochs)
  valid_plot_data[{ {},1 }] = torch.range(0, n_epochs)

  print('Beginning training...')

  local W = torch.DoubleTensor(nfeatures, nclasses)
  local b = torch.DoubleTensor(nclasses)
  b = torch.expand(b:resize(nclasses,1), nclasses, batch_size):fill(0) -- For minibatch
  W:fill(0)


  -- Sample random train batch and return accuracy
  function train_accuracy(subset_size)
    local subset_start = torch.random(1, train_x:size(1) - subset_size)
    local subset_end = subset_start + subset_size - 1
    local sparse_train_subset_x = sparsify(train_x:index(1, torch.range(subset_start, subset_end):long())):t()
    local train_subset_y = train_y:index(1, torch.range(subset_start, subset_end):long())
    return accuracy(sparse_train_subset_x, train_subset_y)
  end

  -- Start with an initial benchmark for train and validation accuracy
  local sparse_valid_x = sparsify(valid_x):t()
  acc = accuracy(sparse_valid_x, valid_y)
  valid_plot_data[1][2] = acc
  print('Untrained validation set accuracy: ' .. acc)


  acc = train_accuracy(50000)
  train_plot_data[1][2] = acc
  print('Untrained training set accuracy: ' .. acc)

  for i = 1, n_epochs do

    -- Minibatch forward pass and gradient calculation
    print('Beginning epoch ' .. i .. ' training...')
    for j = 0, n_train_batches do


      -- Sample random batch and sparsify on the fly
      local batch_start = torch.random(1, train_x:size(1) - batch_size)
      local batch_end = batch_start + batch_size - 1
      local x = sparsify(train_x:index(1, torch.range(batch_start, batch_end):long())):t()
      local y = train_y:index(1, torch.range(batch_start, batch_end):long()):resize(1, batch_size)
      local y_onehot = onehot(y, nclasses)

      local z = W:t() * x + b -- + torch.expand(b:resize(nclasses, 1), nclasses, batch_size)

      -- l2 regularization
      local l2 = torch.cmul(W, W):sum() * lambda / 2.

      -- calculate partial derivatives and loss
      if loss_fn == 'cross_entropy' then
        local py_x = torch.exp(softmax(z))
        dL_dz, loss = cross_entropy_loss(py_x, y_onehot, l2)
      elseif loss_fn == 'hinge' then
        dL_dz, loss = hinge_loss(z, y_onehot, l2)
      end


      -- if j % 1000 == 0 then
      --   print('Loss after ' .. j .. ' minibatches: ' .. loss:sum())
      -- end

      -- calculate gradients
      local W_grad = x * dL_dz:t()
      local b_grad = torch.expand(torch.mean(dL_dz, 2), nclasses, batch_size)

      assert(W_grad:size(1) == W:size(1))
      assert(W_grad:size(2) == W:size(2))
      assert(b_grad:size(1) == b:size(1))
      assert(b_grad:size(2) == b:size(2))

      -- Update weights
      local decay = (1 - (lr * lambda) / 10.0)
      W = (W * decay) - (W_grad * lr)
      b = (b * decay) - (b_grad * lr)
    end -- End minibatch sgd

    print('Epoch ' .. i .. ' training complete!')
    print('Evaluating accuracy on training subset and validation set...')

    -- calculate accuracy on subset
    local subset_size = 50000
    local subset_start = torch.random(1, train_x:size(1) - subset_size)
    local subset_end = subset_start + subset_size - 1
    local sparse_train_subset_x = sparsify(train_x:index(1, torch.range(subset_start, subset_end):long())):t()
    local train_subset_y = train_y:index(1, torch.range(subset_start, subset_end):long())
    local acc = accuracy(sparse_train_subset_x, train_subset_y, W, b)
    print('Training subset accuracy: ' .. acc)

    local sparse_valid_x = sparsify(valid_x):t()
    acc = accuracy(sparse_valid_x, valid_y, W, b)
    
    local acc = train_accuracy(subset_size)
    train_plot_data[i + 1][2] = acc
    print('Training subset accuracy: ' .. acc)

    local sparse_valid_x = sparsify(valid_x):t()
    acc = accuracy(sparse_valid_x, valid_y)
    valid_plot_data[i + 1][2] = acc
    print('Validation accuracy: ' .. acc)
  end -- End epoch evaluation

  plot(train_plot_data, 'Linear SVM Training Accuracy', 'Epochs', 'Accuracy', 'train')
  plot(valid_plot_data, 'Linear SVM Validation Accuracy', 'Epochs', 'Accuracy', 'valid')

  -- Test predictions for Kaggle
  -- local sparse_test_x = sparsify(test_x):t()
  -- local z = W:t() * sparse_test_x
  -- local py_x = softmax(z)
  -- local max, pred = py_x:max(1)
  -- writeToFile(pred:int():resize(pred:size(2)))

  print('Logistic regression model training complete!')
end

function main()
  -- Parse input params
  opt = cmd:parse(arg)
  alpha = opt.alpha
  lr = opt.lr
  lambda = opt.lambda
  n_epochs = opt.n_epochs
  batch_size = opt.m
  kf = opt.kfold

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
  elseif opt.classifier == 'lr-cross' then
    logistic_regression('cross_entropy')
  elseif opt.classifier == 'lr-hinge' then
    logistic_regression('hinge')
<<<<<<< HEAD
  elseif opt.classifier == 'svm' then
    linear_svm()
  elseif opt.classifier == 'nn' then
    multilayer_logistic_regression()
  elseif opt.classifier =='kf-cross' then
    kfolds_logistic('cross_entropy', kf) 
  elseif opt.classifier =='kf-hinge' then
    kfolds_logistic('hinge', kf)
=======
  elseif opt.classifier =='kf' then
    kfolds_logistic('cross_entropy', 10)
>>>>>>> 7e04d6feaa4a4765ea2d8f5dad3b85e29ac9cd19
  end
  -- Test
end

-- Misc functions

-- Plotting
function plot(data, title, xlabel, ylabel, filename)
  -- NB: requires gnuplot
  -- gnuplot.raw('set xtics (0, 1, 2, 3, 4, 5)')
  gnuplot.pngfigure(filename .. '.png')
  gnuplot.plot(data)
  gnuplot.title(title)
  gnuplot.xlabel(xlabel)
  gnuplot.ylabel(ylabel)
  gnuplot.plotflush()
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

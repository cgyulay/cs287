-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-lm', 'nb', 'classifier to use')
cmd:option('-alpha', 0.01, 'Laplace smoothing coefficient')
cmd:option('-eta', 0.01, 'learning rate')
cmd:option('-lambda', 0.05, 'l2 regularization coefficient')
cmd:option('-nepochs', 3, 'number of training epochs')
cmd:option('-m', 20, 'minibatch size')

function tensor_to_key(t)
  --[[
  Helper function which converts a tensor to a key for table lookup.
  ]]

  local key = ''
  local table = torch.totable(t)
  for k, v in pairs(table) do
    key = key .. tostring(v) .. ','
  end
  return string.sub(key, 1, -2) -- Remove trailing comma
end

function count_based(smoothing)
  print('Building count based model (ngram = ' .. ngram .. ')...')

  -- Easier to handle smoothing if each context gram has own dim
  -- Actually this wouldn't work for full vocab size (size constraints)
  -- local ngrams = torch.Tensor() -- Hideous but can't think of a better way
  -- if ngram == 3 then ngrams = torch.Tensor(torch.LongStorage({nwords,nwords,nwords})) end
  -- if ngram == 2 then ngrams = torch.Tensor(torch.LongStorage({nwords,nwords})) end

  print('Building p(w_i|w_i-n+1,...,w_i-1)...')
  local ngrams = {}
  for i = 1, train_x:size(1) do
    local ctx = tensor_to_key(train_x[i])
    local wi = train_y[i]
    local val = ngrams[ctx]
    if val == nil then
      ngrams[ctx] = {}
      ngrams[ctx][wi] = 1
    else
      local innerval = ngrams[ctx][wi]
      if innerval == nil then
        ngrams[ctx][wi] = 1
      else
        ngrams[ctx][wi] = ngrams[ctx][wi] + 1
      end
    end

    if i % 100000 == 0 then
      print('Processed ' .. i .. ' training examples.')
    end
  end

  -- Add smoothing to account for unseen ngrams
  -- If a context has never been seen before, then we can assume uniform
  -- or perhaps unigram distribution (backoff)

  -- Renormalize probabilities for each ngram
  print('Renormalizing ngram probabilities...')
  for ctx, wis in pairs(ngrams) do
    local sum = 0
    local seen = 0 -- To calculate the number of unseen wi in given ctx
    for wi, tot in pairs(wis) do
      sum = sum + tot
      seen = seen + 1
    end
    for wi, tot in pairs(wis) do
      ngrams[ctx][wi] = ngrams[ctx][wi] / sum -- Normalize
    end
  end

  function perplexity(x, y)
    local sum = 0
    local unseen = 0
    for i = 1, x:size(1) do
      local ctx = tensor_to_key(x[i])
      local wi = y[i]
      local pctx = ngrams[ctx]
      local p = 0
      if pctx == nil then
        unseen = unseen + 1
        p = (1.0 / nwords) -- Uniform assumption
      else
        p = ngrams[ctx][wi]
        if p == nil then
          unseen = unseen + 1
          p = (1.0 / nwords)
        end
      end
      sum = sum + math.log(p)
    end
    local nll = (-1.0 / x:size(1)) * sum
    print('Unseen ngrams: ' .. unseen)
    return math.exp(nll)
  end

  print('Calculating perplexity on train...')
  local perp = perplexity(train_x, train_y)
  print('Training perplexity: ' .. perp)

  print('Calculating perplexity on valid...')
  perp = perplexity(valid_x, valid_y)
  print('Validation perplexity: ' .. perp)

end

function main()
  -- Parse input params
  opt = cmd:parse(arg)
  lm = opt.lm
  alpha = opt.alpha
  eta = opt.eta
  lambda = opt.lambda
  n_epochs = opt.nepochs
  batch_size = opt.m

  local f = hdf5.open(opt.datafile, 'r')
  nwords = f:read('nwords'):all():long()[1]
  nclasses = f:read('nclasses'):all():long()[1]
  ngram = f:read('ngram'):all():long()[1]

  -- Split training and validation data
  train_x = f:read('train_input'):all()
  train_y = f:read('train_output'):all()
  valid_x = f:read('valid_input'):all()
  valid_y = f:read('valid_output'):all()
  -- test_x = f:read('test_input'):all() TODO

  if lm == 'cb' or lm == 'lap' or lm == 'wb' or lm == 'kn' then
    count_based(lm)
  else
    print('Let\'s build some f*cking neural nets!')
    -- model(lm)
  end
end

main()

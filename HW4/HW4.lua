-- Only requirement allowed
require("hdf5")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-lm', 'nn', 'classifier to use')
cmd:option('-alpha', 0.01, 'Laplace smoothing coefficient')
cmd:option('-eta', 0.05, 'learning rate')
cmd:option('-nepochs', 3, 'number of training epochs')
cmd:option('-mb', 32, 'minibatch size')
cmd:option('-ngram', 3, 'ngram size to use for context')
cmd:option('-gpu', 0, 'whether to use gpu for training')

----------
-- Count based model
----------

-- Helper function which converts a tensor to a key for table lookup
function tensor_to_key(t)
  local key = ''
  local table = torch.totable(t)
  for k, v in pairs(table) do
    key = key .. tostring(v) .. ','
  end
  return string.sub(key, 1, -2) -- Remove trailing comma
end

-- Backs off a context
function back_off_context(ctx)
  local idx = ctx:match'^.*(),' - 1
  return string.sub(ctx, 1, idx)
end

function count_based(max_ngram)
  ngrams = {}

  -- Calculate the space/no-space distribution
  function build_unigram(y)
    local total = 0
    local spaces = 0
    for i = 1, y:size(1) do
      local s = y[i]
      if s == 1 then
        spaces = spaces + 1
      end
      total = total + 1
    end

    local unigram = {}
    unigram[0] = (total - spaces) / total
    unigram[1] = spaces / total
    return unigram
  end

  -- Create co-occurrence dictionary mapping contexts to following words
  function build_ngram(x, y, ngram)
    print('Building p(w_i|w_i-n+1,...,w_i-1) for i=' .. ngram .. '...')
    ngram = ngram - 1 -- Shorten by 1 for context only
    local grams = {}
    for i = ngram, x:size(1) do
      local ctx = tensor_to_key(x:narrow(1, i-ngram+1, ngram))
      local s = y[i]
      local val = grams[ctx]
      if val == nil then
        grams[ctx] = {}
        grams[ctx][s] = 1
      else
        local innerval = grams[ctx][s]
        if innerval == nil then
          grams[ctx][s] = 1
        else
          grams[ctx][s] = grams[ctx][s] + 1
        end
      end

      if i % 100000 == 0 then
        print('Processed ' .. i .. ' training examples.')
      end
    end
    return grams
  end

  -- Renormalize probabilities for each ngram
  function normalize_ngram(ngram)
    print('Renormalizing ngram probabilities...')
    for ctx, ys in pairs(ngram) do
      -- Total number of spaces/non-spaces we've seen in this context
      local tot0 = ngram[ctx][0] or 0
      local tot1 = ngram[ctx][1] or 0
      local sum = tot0 + tot1

      -- Convert to space/non-space probabilities for each context
      for s, tot in pairs(ys) do
        ngram[ctx][s] = ngram[ctx][s] / sum
      end
    end
  end

  function p_ngram(ctx, s)
    -- Select probability from longest valid context
    -- If no probability is established at this context, continue backing off
    -- until one is found, all the way till unigram if necessary
    local found = false
    local g = max_ngram
    local p = 0

    function back_off()
      g = g - 1
      if g == 1 then
        p = ngrams[g][s]
        found = true
      else
        ctx = back_off_context(ctx)
      end
    end

    while not found do
      ctxd = ngrams[g][ctx]
      if ctxd ~= nil then
        p = ngrams[g][ctx][s]
        if p ~= nil then
          found = true
        else
          back_off()
        end
      else
        back_off()
      end
    end

    return p
  end

  -- Perplexity = exponential(avg negative conditional-log-likelihood)
  function perplexity(x, y, ngram)
    local sum = 0
    for i = ngram, x:size(1) do
      local ctx = tensor_to_key(x:narrow(1, i-ngram+1, ngram))
      local s = y[i]
      local p = p_ngram(ctx, s)

      sum = sum + math.log(p)
      if p == 0 then
        print('0 prob for ctx')
      end
    end

    local nll = (-1.0 / x:size(1)) * sum
    return math.exp(nll)
  end

  print('Building count based model (ngram=' .. max_ngram .. ')...')
  -- Build all ngrams <= max ngram for backoff in unseen cases
  local unigram = build_unigram(train_y)
  ngrams[1] = unigram
  for i = max_ngram, 2, -1 do
    local ngram = build_ngram(train_x_cb, train_y_cb, i)
    normalize_ngram(ngram)
    ngrams[i] = ngram
  end

  print('Calculating perplexity on train...')
  local perp = perplexity(train_x_cb, train_y_cb, max_ngram - 1)
  print('Training perplexity: ' .. perp)

  print('Calculating perplexity on valid...')
  local perp = perplexity(valid_x_cb, valid_y_cb, max_ngram - 1)
  print('Validation perplexity: ' .. perp)
end

function main() 
  -- Parse input params
  opt = cmd:parse(arg)
  datafile = opt.datafile
  lm = opt.lm
  alpha = opt.alpha
  eta = opt.eta
  n_epochs = opt.nepochs
  batch_size = opt.mb
  ngram = opt.ngram
  gpu = opt.gpu

  local f = hdf5.open(opt.datafile, 'r')
  nclasses = f:read('nclasses'):all():long()[1]

  -- Split training, validation, test data
  train_x_cb = f:read('train_input_cb'):all()
  train_y_cb = f:read('train_output_cb'):all()
  valid_x_cb = f:read('valid_input_cb'):all()
  valid_y_cb = f:read('valid_output_cb'):all()

  train_x = f:read('train_input'):all()
  train_y = f:read('train_output'):all()
  valid_x = f:read('valid_input'):all()
  valid_y = f:read('valid_output'):all()
  valid_kaggle_x = f:read('valid_input'):all()
  valid_kaggle_y = f:read('valid_output'):all()
  test_x = f:read('test_input'):all()

  if lm == 'cb' then
    count_based(ngram)
  elseif lm == 'nn' then
    -- nnlm()
  else
    -- rnn()
  end
   
end

main()

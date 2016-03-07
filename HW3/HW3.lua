-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

cmd = torch.CmdLine()

UNSEEN = -1

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-lm', 'nb', 'classifier to use')
cmd:option('-alpha', 0.01, 'Laplace smoothing coefficient')
cmd:option('-eta', 0.01, 'learning rate')
cmd:option('-lambda', 0.05, 'l2 regularization coefficient')
cmd:option('-nepochs', 3, 'number of training epochs')
cmd:option('-m', 20, 'minibatch size')

-- Helper function which converts a tensor to a key for table lookup
function tensor_to_key(t)
  local key = ''
  local table = torch.totable(t)
  for k, v in pairs(table) do
    key = key .. tostring(v) .. ','
  end
  return string.sub(key, 1, -2) -- Remove trailing comma
end

function count_based(smoothing)
  -- Specializd function for building unigram probability distribution (no context)
  function build_unigram(x, y)
    local unigrams = {}
    for i = 1, train_y:size(1) do
      local wi = train_y[i]
      if unigrams[wi] == nil then
        unigrams[wi] = 1
      else
        unigrams[wi] = unigrams[wi] + 1
      end
    end
    return unigrams
  end

  print('Building unigram distribution...')
  local unigram = build_unigram()

  function p_unigram(wi)
    p = unigram[wi]
    if p == nil then -- Never before seen unigram, go with uniform dist
      p = (1.0 / nwords)
    end
    return p
  end

  -- Create co-occurrence dictionary mapping contexts to following words
  function build_ngram(x, y)
    print('Building p(w_i|w_i-n+1,...,w_i-1)...')
    local ngram = {}
    for i = 1, x:size(1) do
      local ctx = tensor_to_key(x[i])
      local wi = y[i]
      local val = ngram[ctx]
      if val == nil then
        ngram[ctx] = {}
        ngram[ctx][wi] = 1
      else
        local innerval = ngram[ctx][wi]
        if innerval == nil then
          ngram[ctx][wi] = 1
        else
          ngram[ctx][wi] = ngram[ctx][wi] + 1
        end
      end

      if i % 100000 == 0 then
        print('Processed ' .. i .. ' training examples.')
      end
    end
    return ngram
  end

  -- Renormalize probabilities for each ngram
  function normalize_ngram(ngram)
    print('Renormalizing ngram probabilities...')
    for ctx, wis in pairs(ngram) do
      local sum = 0
      local seen = 0 -- To calculate the number of unseen wi in given ctx
      for wi, tot in pairs(wis) do
        sum = sum + tot
        seen = seen + 1
      end

      local unseen = nwords - seen
      -- Smoothing + normalization
      if smoothing == 'mle' then
        -- How to predict unseen without smoothing for mle??
        sum = sum + (unseen * (1.0 / nwords)) -- Uniform assumption
        for wi, tot in pairs(wis) do
          ngram[ctx][wi] = ngram[ctx][wi] / sum
        end
        ngram[ctx][UNSEEN] = (1.0 / nwords) / sum
      elseif smoothing == 'lap' then
        sum = sum + (nwords * alpha) -- Add probability mass for every wi (including unseen)
        for wi, tot in pairs(wis) do
          ngram[ctx][wi] = (ngram[ctx][wi] + alpha) / sum
        end
        ngram[ctx][UNSEEN] = alpha / sum
      end
    end
  end

  function p_for_ngram(ngram, ctx, wi)
    local ctxd = ngram[ctx]
    local p = 0
    if ctxd == nil then -- Never before seen context, go with unigram
      p = p_unigram(wi)
    else
      p = ngram[ctx][wi]
      if p == nil then -- We've seen this context before, just not this wi
        p = ngram[ctx][UNSEEN]
      end
    end
    return p
  end

  -- Count total occurrences of context and unique types within context
  function lambda_for_ctx(ngram, ctx, word)
    local unique = 0
    local total = 0
    local ctx_wi_freq = 0
    if ngram[ctx] == nil then
      return 0, 0, 0, 0 -- By defn seems like should be 1, but intuitively should be 0
    else
      for wi, tot in pairs(ngram[ctx]) do
        unique = unique + 1
        total = total + tot
        if wi == word then ctx_wi_freq = tot end
      end

      local lambda = 1 - (unique / (unique + total))
      return lambda, unique, total, ctx_wi_freq
    end
  end

  -- Calculates interpolated probability based on occurrences of the specific
  -- context/word pair, unique words in context, total words in context, and
  -- the probability of the 1-order lower context
  function p_wb(ctx_wi_freq, unique, total, p_lower)
    if unique + total == 0 then -- Prevent / 0 in denom
      return 0
    else
      local num = ctx_wi_freq + unique * p_lower
      local denom = unique + total
      return num / denom
    end
  end

  -- Perplexity = exponential(avg negative log conditional probability)
  function perplexity(ngram, x, y)
    local sum = 0
    for i = 1, x:size(1) do
      local ctx = tensor_to_key(x[i])
      local wi = y[i]
      local p = p_for_ngram(ngram, ctx, wi)
      
      sum = sum + math.log(p)
    end

    local nll = (-1.0 / x:size(1)) * sum
    return math.exp(nll)
  end

  -- Perplexity = exponential(avg negative log conditional probability,
  -- interpolated across all lower order ngrams)
  function wb_perplexity(trigram, bigram, x, y)
    local sum = 0
    for i = 1, x:size(1) do
      if trigram ~= nil then -- Trigram, bigram, unigram interpolation
        local ctx3 = tensor_to_key(x[i])
        local ctx2 = tensor_to_key(torch.Tensor({x[i][1]}))
        local wi = y[i]

        unigram_freq = p_unigram(wi) * nwords -- Remultiply by nwords as it was previously normalized
        local l3, unq3, tot3, ctx_wi_freq3 = lambda_for_ctx(trigram, ctx3, wi)
        local l2, unq2, tot2, ctx_wi_freq2 = lambda_for_ctx(bigram, ctx2, wi)
        local l1 = nwords / (nwords + unigram_freq)
        local all = l3 + l2 + l1 -- Ensure lambdas form complex combination
        l3 = l3 / all
        l2 = l2 / all
        l1 = l1 / all

        p1 = p_unigram(wi)
        p2 = p_wb(ctx_wi_freq2, unq2, tot2, p1)
        p3 = p_wb(ctx_wi_freq3, unq3, tot3, p2)

        p_interp = (l3 * p3) + (l2 * p2) + (l1 * p1)
        sum = sum + math.log(p_interp)
      else -- Bigram, unigram interpolation
        local ctx = tensor_to_key(x[i])
        local wi = y[i]

        unigram_freq = p_unigram(wi) * nwords
        local l2, unq2, tot2, ctx_wi_freq2 = lambda_for_ctx(bigram, ctx, wi)
        local l1 = nwords / (nwords + unigram_freq)
        local all = l2 + l1 -- Ensure lambdas form complex combination
        l2 = l2 / all
        l1 = l1 / all

        p1 = p_unigram(wi)
        p2 = p_wb(ctx_wi_freq2, unq2, tot2, p1)

        p_interp = (l2 * p2) + (l1 * p1)
        sum = sum + math.log(p_interp)
      end
    end

    local nll = (-1.0 / x:size(1)) * sum
    return math.exp(nll)
  end

  if smoothing ~= 'wb' then
    print('Building count based model (ngram=' .. ngram .. ', smoothing=' .. smoothing .. ')...')

    local ngram = build_ngram(train_x, train_y)
    normalize_ngram(ngram)

    print('Calculating perplexity on train...')
    local perp = perplexity(ngram, train_x, train_y)
    print('Training perplexity: ' .. perp)

    print('Calculating perplexity on valid...')
    perp = perplexity(ngram, valid_x, valid_y)
    print('Validation perplexity: ' .. perp)
  else
    print('Building count based model with Witten-Bell smoothing (max ngram=' .. ngram .. ')...')

    local bigram = build_ngram(bigram_train_x, bigram_train_y)
    -- normalize_ngram(bigram)

    local trigram = nil
    if interp == 3 then
      trigram = build_ngram(trigram_train_x, trigram_train_y)
      -- normalize_ngram(trigram)
    end

    print('Calculating preplexity on train...')
    local perp = wb_perplexity(trigram, bigram, train_x, train_y)
    print('Training perplexity: ' .. perp)

    print('Calculating preplexity on valid...')
    perp = wb_perplexity(trigram, bigram, valid_x, valid_y)
    print('Validation perplexity: ' .. perp)
  end
end

function nn()
  print('Let\'s build some f*cking neural nets!')
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

  -- Format datasets for Witten-Bell
  if lm == 'wb' then
    if string.find(opt.datafile, '3') then
      interp = 3
      -- Need to load bigrams
      local f2 = hdf5.open('PTB_2gram.hdf5', 'r')
      bigram_train_x = f2:read('train_input'):all()
      bigram_train_y = f2:read('train_output'):all()
      bigram_valid_x = f2:read('valid_input'):all()
      bigram_valid_y = f2:read('valid_output'):all()

      trigram_train_x = train_x
      trigram_train_y = train_y
      trigram_valid_x = valid_x
      trigram_valid_y = valid_y
    else
      interp = 2
      bigram_train_x = train_x
      bigram_train_y = train_y
      bigram_valid_x = valid_x
      bigram_valid_y = valid_y
    end
  end

  if lm == 'mle' or lm == 'lap' or lm == 'wb' then
    count_based(lm)
  elseif lm == 'nn' then
    nn()
  else
    print('Unrecognized model, bailing out.')
  end
end

main()

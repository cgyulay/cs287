-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use')

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
   nwords = f:read('words'):all():long()[1]
   nclasses = f:read('nclasses'):all():long()[1]

   -- Training, valid, and test windows
   train_input_word_windows = f:read('train_input_word_windows'):all()
   train_input_cap_windows = f:read('train_input_cap_windows'):all()
   train_output = f:read('train_output'):all()

   valid_input_word_windows = f:read('valid_input_word_windows'):all()
   valid_input_cap_windows = f:read('valid_input_cap_windows'):all()
   valid_output = f:read('valid_output'):all()

   test_input_word_windows = f:read('test_input_word_windows'):all()
   test_input_cap_windows = f:read('test_input_cap_windows'):all()

   local W = torch.DoubleTensor(nclasses, nfeatures)
   local b = torch.DoubleTensor(nclasses)

   -- Train.

   -- Test.
end

main()

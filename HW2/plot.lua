require("gnuplot")
require("math")

function plot(one, two, title, xlabel, ylabel, filename)
  -- NB: requires gnuplot
  gnuplot.pngfigure('img/' .. filename .. '.png')
  gnuplot.plot({'Valid', one}, {'Train', two})
  -- gnuplot.plot(one)
  gnuplot.title(title)
  gnuplot.xlabel(xlabel)
  gnuplot.ylabel(ylabel)
  -- gnuplot.movelegend('right', 'bottom')
  gnuplot.plotflush()
end

-- Read network output file and plot
local name = 'training_output/mlptest_pretrainedembed_eta=0.04.txt'
-- local f = torch.DiskFile(name, 'r')
-- local lines = f:readString('*a')

-- http://lua-users.org/wiki/FileInputOutput
-- see if the file exists
function file_exists(file)
  local f = io.open(file, "rb")
  if f then f:close() end
  return f ~= nil
end

function lines_from(file)
  if not file_exists(file) then return {} end
  lines = {}
  for line in io.lines(file) do 
    lines[#lines + 1] = line
  end
  return lines
end

function split(str, pat)
  local t = {}  -- NOTE: use {n = 0} in Lua-5.0
  local fpat = "(.-)" .. pat
  local last_end = 1
  local s, e, cap = str:find(fpat, 1)
  while s do
    if s ~= 1 or cap ~= "" then
  table.insert(t,cap)
    end
    last_end = e+1
    s, e, cap = str:find(fpat, last_end)
  end
  if last_end <= #str then
    cap = str:sub(last_end)
    table.insert(t, cap)
  end
  return t
end

function tablelength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end

lines = lines_from('training_output/lrepoch=9.txt')
len = tablelength(lines) - 3

-- Store values for plotting
local vacc = torch.DoubleTensor(len, 2)
local tacc = torch.DoubleTensor(len, 2)
local vloss = torch.DoubleTensor(len, 2)
local tloss = torch.DoubleTensor(len, 2)
local etime = torch.DoubleTensor(len, 2)

vloss[{ {},1 }] = torch.range(1, len)
tloss[{ {},1 }] = torch.range(1, len)
vacc[{ {},1 }] = torch.range(1, len)
tacc[{ {},1 }] = torch.range(1, len)
etime[{ {},1 }] = torch.range(1, len)

skip = 3 -- Skip non-numeric lines
idx = 1
for k,v in pairs(lines) do
  if idx > skip then
    local sp = split(v, ',')

    vacc[idx - 3][2] = tonumber(sp[1])
    tacc[idx - 3][2] = tonumber(sp[2])
    vloss[idx - 3][2] = tonumber(sp[3])
    tloss[idx - 3][2] = tonumber(sp[4])
    etime[idx - 3][2] = tonumber(sp[5])
  end
  idx = idx + 1
end

function accuracyforfile(filename)
  local l = lines_from(filename)
  local length = tablelength(l) - 3

  local acc = torch.DoubleTensor(length, 2)
  acc[{ {},1 }] = torch.range(1, length)

  local skip = 3 -- Skip non-numeric lines
  local i = 1
  for k,v in pairs(l) do
    if i > skip then
      local sp = split(v, ',')
      acc[i - 3][2] = tonumber(sp[1])
    end
    i = i + 1
  end

  return acc
end

plot(vacc, tacc, 'Logistic Regression Training and Validation Accuracy', 'Epochs', 'Acc (%)', 'lracc')

-- o1 = accuracyforfile('training_output/mlptest_pretrainedembed_eta=0.01.txt')
-- o2 = accuracyforfile('training_output/mlptest_pretrainedembed_eta=0.02.txt')
-- o4 = accuracyforfile('training_output/mlptest_pretrainedembed_eta=0.04.txt')
-- o4 = accuracyforfile('training_output/mlptest_pretrainedembed_eta=0.05.txt')
-- o1 = accuracyforfile('training_output/mlp_vanilla_test_eta=0.01.txt')

-- plot(o1, 'Logistic Regression Training and Validation Accuracy', 'Epochs', 'Acc (%)', 'lracc')

require("gnuplot")
require("math")

function plot(one, two, title, xlabel, ylabel, filename)
  -- NB: requires gnuplot
  gnuplot.pngfigure('img/' .. filename .. '.png')
  -- gnuplot.raw('set logscale x 10')
  -- gnuplot.plot({'eta=0.1', one}, {'eta=1', two}, {'eta=2', three}, {'eta=3', four}, {'eta=5', five})
  gnuplot.plot({'m=0', one}, {'m=0.8', two})
  -- gnuplot.plot(one)
  gnuplot.title(title)
  gnuplot.xlabel(xlabel)
  gnuplot.ylabel(ylabel)
  gnuplot.movelegend('right', 'bottom')
  gnuplot.plotflush()
  print('finished plotting img: ' .. filename)
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

-- lines = lines_from('training_output/model=nn,dataset=PTB_6gram.hdf5,lr=0.04.txt')
-- len = tablelength(lines)

-- -- Store values for plotting
-- local vperp = torch.DoubleTensor(len, 2)
-- local etime = torch.DoubleTensor(len, 2)

-- vperp[{ {},1 }] = torch.range(1, len)
-- etime[{ {},1 }] = torch.range(1, len)

-- for k,v in pairs(lines) do
--   local sp = split(v, ',')

--   vperp[idx][2] = tonumber(sp[1])
--   etime[idx][2] = tonumber(sp[5])
-- end

function range(from, to, step)
  step = step or 1
  return function(_, lastvalue)
    local nextvalue = lastvalue + step
    if step > 0 and nextvalue <= to or step < 0 and nextvalue >= to or
       step == 0
    then
      return nextvalue
    end
  end, nil, from - step
end

local sel = 2
function accuracyforfile(filename)
  local l = lines_from(filename)
  local length = tablelength(l)

  local acc = torch.DoubleTensor(length, 2)
  acc[{ {},1 }] = torch.range(1, length)

  local i = 0
  for k,v in pairs(l) do
    i = i + 1
    local sp = split(v, ',')
    acc[i][2] = tonumber(sp[sel])
  end

  return acc
end

function f1forfile(filename)
  local l = lines_from(filename)
  local length = tablelength(l)

  local acc = torch.DoubleTensor(10, 2)
  acc[{ {},1 }] = torch.range(1, 10)

  local i = 0
  local idx = 1
  for k,v in pairs(l) do
    i = i + 1
    if i % 10 == 0 then
      local sp = split(v, ',')
      acc[idx][2] = tonumber(sp[3])
      idx = idx + 1
    end
  end

  acc:select(2,1):cmul(torch.Tensor(10):fill(10))
  return acc
end

-- o1 = f1forfile('training_output/model=memm,f1=0.30237541603156,mem=0,eta=0.1.txt')
-- o2 = f1forfile('training_output/model=memm,f1=0.40128617941548,mem=0,eta=1.txt')
-- o3 = f1forfile('training_output/model=memm,f1=0.41821051860721,mem=0,eta=2.txt')
-- o4 = f1forfile('training_output/model=memm,f1=0.45111164244042,mem=0,eta=3.txt')
-- o5 = f1forfile('training_output/model=memm,f1=0.44084614884707,mem=0,eta=5.txt')
o1 = accuracyforfile('training_output/model=sp,f1=0.16298988860521,mem=0,eta=1.txt')
o2 = accuracyforfile('training_output/model=sp,f1=0.56873507790478,mem=0.8,eta=1.txt')

plot(o1, o2, 'Comparing m on SP Validation Performance', 'Epoch', 'Accuracy', 'spcomp')

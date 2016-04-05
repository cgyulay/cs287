require("gnuplot")
require("math")

function plot(one, two, three, title, xlabel, ylabel, filename)
  -- NB: requires gnuplot
  gnuplot.pngfigure('img/' .. filename .. '.png')
  -- gnuplot.raw('set logscale x 10')
  gnuplot.plot({'dembed=15', one}, {'dembed=50', two}, {'dembed=100', three})
  -- gnuplot.plot({'NNLM', one}, {'NNLMHSM', two})
  -- gnuplot.plot(one)
  gnuplot.title(title)
  gnuplot.xlabel(xlabel)
  gnuplot.ylabel(ylabel)
  -- gnuplot.movelegend('right', 'bottom')
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

function accuracyforfile(filename)
  local l = lines_from(filename)
  local length = tablelength(l)

  local acc = torch.DoubleTensor(length, 2)
  acc[{ {},1 }] = torch.range(1, length)

  local i = 0
  for k,v in pairs(l) do
    i = i + 1
    local sp = split(v, ',')
    acc[i][2] = tonumber(sp[2])
  end

  return acc
end

-- local o1 = torch.DoubleTensor({{0.0001, 68.67}, {0.001, 73.77}, {0.01, 101.17}, {0.1, 216.82}, {1, 682.63}})
-- local o2 = torch.DoubleTensor({{0.0001, 506.65}, {0.001, 343.40}, {0.01, 294.94}, {0.1, 395.94}, {1, 846.90}})
-- plot(o1, o2, 'Tuning Alpha Smoothing Constant for CBLap', 'Alpha', 'Perp', 'laptune')

o1 = accuracyforfile('training_output/model=nn,dwin=9,dembed=15,mse=6.1554896142433.txt')
o2 = accuracyforfile('training_output/model=nn,dwin=9,dembed=50,mse=6.4412462908012.txt')
o3 = accuracyforfile('training_output/model=nn,dwin=9,dembed=100,mse=6.1810089020772.txt')
-- o4 = accuracyforfile('training_output/model=nn,dwin=11,dembed=50,mse=6.8.txt')

plot(o1, o2, o3, 'NNLM Comparison (dembed)', 'Epochs', 'Perp', 'nnlmcompdembed')

local log = {}

local function save_performance(name, vperp, tperp)
  print('saving ' .. name)

  local f = torch.DiskFile('training_output/' .. name .. '.txt', 'w')
  for j = 1, valid:size(1) do
    f:writeString(vperp[j] .. ','  .. tperp[j] .. '\n')
  end
  f:close()
end

log.save_performance = save_performance
return log
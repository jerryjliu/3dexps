require 'torch'
require 'nn'
require 'optim'
require 'paths'
assert(pcall(function () mat = require('fb.mattorch') end) or pcall(function() mat = require('matio') end), 'no mat IO interface available')

opt = {
  gpu=1,
  name = 'shapenet101',
  data_dir='/data/jjliu/checkpoints/',
  checkpointf='checkpoints_64chair100o_vaen2',
  epoch=1450,
  ext = 'net_G',
  genEpoch=1450,
}
for k,v in pairs(opt) do 
  opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] 
  print(k .. ': ' .. opt[k])
end 
if opt.gpu > 0 then
  require 'cunn'
  require 'cudnn'
  require 'cutorch'
  cutorch.setDevice(opt.gpu)
  nn.DataParallelTable.deserializeNGPUs = 1
end

if opt.ext == 'net_P' then
  fname = opt.name .. '_' .. opt.genEpoch .. '_' .. opt.epoch .. '_' .. opt.ext .. '.t7'
else
  fname = opt.name .. '_' .. opt.epoch .. '_' .. opt.ext .. '.t7'
end
local fpath = paths.concat(opt.data_dir .. opt.checkpointf, fname)

local net = torch.load(fpath)
print(net)
local name = torch.type(net)
if name:find('DataParallelTable') then
  net = net:get(1)
end
if opt.gpu > 0 then
  net = cudnn.convert(net, nn)
  net = net:float()
end

print(net)
torch.save(fpath, net)
--torch.save(opath, net)

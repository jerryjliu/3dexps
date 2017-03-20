require 'torch'
require 'nn'
require 'optim'
require 'paths'
assert(pcall(function () mat = require('fb.mattorch') end) or pcall(function() mat = require('matio') end), 'no mat IO interface available')

opt = {
  nz = 200,
  name = 'shapenet101',
  ext = 'net_C',
  checkpointd = '/data/jjliu/checkpoints/',
  checkpointf='checkpoints_64class100',
  epoch=25,
  gpu=1,
  splitIndex=7
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end 

if opt.gpu > 0 then
  require 'cunn'
  require 'cudnn'
  require 'cutorch'
  cutorch.setDevice(opt.gpu)
end

local net = torch.load(paths.concat(opt.checkpointd .. opt.checkpointf, opt.name .. '_' .. opt.epoch .. '_' .. opt.ext .. '.t7'))
net = net:clone()
--print(net)
assert(opt.splitIndex <= net:size())
num2remove = net:size() - opt.splitIndex
for i = 1, num2remove do
  net:remove()
end

print(net)
print(net:size())

torch.save(paths.concat(opt.checkpointd .. opt.checkpointf, opt.name .. '_' .. opt.epoch .. '_' .. opt.ext .. '_split' .. opt.splitIndex .. '.t7'), net)

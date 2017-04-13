require 'torch'
require 'nn'
require 'optim'
require 'paths'
assert(pcall(function () mat = require('fb.mattorch') end) or pcall(function() mat = require('matio') end), 'no mat IO interface available')

opt = {
  gpu=2,
  checkpointd = '/data/jjliu/checkpoints/',
  checkpointf='checkpoints_64class100_5/shapenet101_200_net_C_split9.t7',
}
for k,v in pairs(opt) do 
  opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] 
  --print(k .. ': ' .. opt[k])
end 

if opt.gpu > 0 then
  require 'cunn'
  require 'cudnn'
  require 'cutorch'
  cutorch.setDevice(opt.gpu)
end

local net = torch.load(paths.concat(opt.checkpointd .. opt.checkpointf))
print(net)


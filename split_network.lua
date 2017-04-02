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
  parallel=false,
  splitIndex=7,
  removeDropout=false,
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
end

local net = torch.load(paths.concat(opt.checkpointd .. opt.checkpointf, opt.name .. '_' .. opt.epoch .. '_' .. opt.ext .. '.t7'))
net = net:clone()
if opt.parallel then
  net = net:get(1)
end
print(net)
assert(opt.splitIndex <= net:size())
num2remove = net:size() - opt.splitIndex
for i = 1, num2remove do
  net:remove()
end

if opt.removeDropout then
  while true do
    done = true
    for i = 1, net:size() do
      local name = torch.type(net:get(i))
      print(name)
      if name:find('Dropout') then
        net:remove(i)
        done = false
      end
    end
    if done then
      break
    end
  end
end

print(net)
print(net:size())

out_path = paths.concat(opt.checkpointd .. opt.checkpointf, opt.name .. '_' .. opt.epoch .. '_' .. opt.ext .. '_split' .. opt.splitIndex .. '.t7')
torch.save(out_path, net)
print(out_path)

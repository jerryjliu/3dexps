require 'torch'
require 'nn'
require 'optim'
require 'paths'
assert(pcall(function () mat = require('fb.mattorch') end) or pcall(function() mat = require('matio') end), 'no mat IO interface available')

opt = {
  data_dir='/data/jjliu/models',
  data_name='proj_inputs_voxel/test_chair4854',
  droprate = 0.3,
  gpu=1,
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end 

if opt.gpu > 0 then
  require 'cunn'
  require 'cudnn'
  require 'cutorch'
  cutorch.setDevice(opt.gpu)
end
print(paths.concat(opt.data_dir, opt.data_name .. '.mat'))
local obj = mat.load(paths.concat(opt.data_dir, opt.data_name .. '.mat'), 'off_volume')
local dropNN = nn.Dropout()

if opt.gpu > 0 then
  obj = obj:cuda()
  dropNN = dropNN:cuda()
  dropNN = cudnn.convert(dropNN, cudnn)
end

obj = dropNN:forward(obj)

if opt.gpu > 0 then
  obj = obj:double()
end
out_path = paths.concat(opt.data_dir, opt.data_name .. '_drop.mat')
mat.save(out_path, {['off_volume'] = obj})

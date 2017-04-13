require 'torch'
require 'nn'
require 'optim'
require 'paths'
assert(pcall(function () mat = require('fb.mattorch') end) or pcall(function() mat = require('matio') end), 'no mat IO interface available')

opt = {
  data_dir='/data/jjliu/models',
  data_name='full_dataset_voxels_64/chair/2088',
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end 

local obj = mat.load(paths.concat(opt.data_dir, opt.data_name .. '.mat'), 'off_volume')
nz=200
tmplatent = torch.Tensor(nz, 1,1,1)
out_path = paths.concat('./output/', 'finput.mat')
print(out_path)
print(obj:size())
local outobj = torch.Tensor(1,64,64,64)
outobj:copy(obj)
print(outobj:size())
mat.save(out_path, {['inputs'] = latent, ['voxels'] = outobj})

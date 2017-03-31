require 'torch'
require 'paths'
assert(pcall(function () mat = require('fb.mattorch') end) or pcall(function() mat = require('matio') end), 'no mat IO interface available')

opt = {
  dir = '/data/jjliu/models/proj_inputs_voxel',
  src_file = 'testvoxels',
  dest_file = 'testvoxels',
  format = 'ascii',
}
for k,v in pairs(opt) do 
  opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] 
end 

full_src_file = paths.concat(opt.dir, opt.src_file .. '.t7')
obj = torch.load(full_src_file, opt.format)
rand_input = torch.rand(200, 1, 1, 1)

full_dest_file = paths.concat(opt.dir, opt.dest_file .. '.mat')
mat.save(full_dest_file, {['inputs'] = rand_input, ['voxels'] = obj})
print("Saved to " .. full_dest_file)

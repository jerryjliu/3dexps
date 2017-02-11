require 'nn'
require 'torch'
require 'xlua'
assert(pcall(function () mat = require('fb.mattorch') end) or pcall(function() mat = require('matio') end), 'no mat IO interface available')

opt = {
  npairs = 7,
  ninter = 5,
  genEpoch = 580,
  nz = 200,
  gpu = 1,
  gen_checkpointd = '/data/jjliu/checkpoints',
  gen_checkpointf = 'checkpoints_64chair_ref',
  interpolated  = '/data/jjliu/interpolate',
  interpolatef = 'chair_500_1',
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end 

if opt.gpu > 0 then
  require 'cunn'
  require 'cudnn'
  require 'cutorch'
  cutorch.setDevice(opt.gpu)
end
checkpoint_path = paths.concat(opt.gen_checkpointd,opt.gen_checkpointf)
print('Loading network..')
gen_path = paths.concat(checkpoint_path, 'shapenet101_' .. opt.genEpoch .. '_net_G.t7')
netG = torch.load(gen_path)
if opt.gpu == 0 then
  netG = netG:double()
end
netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)     -- convolution bias is removed during training
netG:evaluate() -- batch normalization behaves differently during evaluation

print('Setting inputs..')
input = torch.Tensor(2,opt.nz,1,1,1)
if opt.gpu > 0 then
  netG = netG:cuda()
  netG = cudnn.convert(netG, cudnn)
  input = input:cuda()
end
testinput = netG:forward(input)
dim = testinput:size()[3]
inputs = torch.Tensor(opt.ninter, opt.nz, 1, 1, 1)
results = torch.Tensor(opt.ninter, 1, dim, dim, dim)

if paths.dir(opt.interpolated) == nil then
  paths.mkdir(opt.interpolated)
end

print('Forward prop')
for i = 1, opt.npairs do
  point1 = torch.zeros(opt.nz, 1, 1, 1)
  point2 = torch.zeros(opt.nz, 1, 1, 1)
  point = torch.zeros(opt.nz, 1, 1, 1)
  point1:uniform(0,1)
  point2:uniform(0,1)
  inputs:zero()
  results:zero()
  for j = 1, opt.ninter do 
    lam = (j-1)/opt.ninter
    point:zero()
    point:add(lam, point1)
    point:add(1 - lam, point2)
    inputs[{j}]:copy(point)
  end
  if opt.gpu > 0 then
    outputs = netG:forward(inputs:cuda())
  else 
    outputs = netG:forward(inputs)
  end
  results:copy(outputs)
  mat.save(paths.concat(opt.interpolated, opt.interpolatef .. '_' .. i .. '.mat'), {['inputs']=inputs, ['voxels']=results})
end
print('saving done')

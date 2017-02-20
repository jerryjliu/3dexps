require 'nn'
require 'torch'
require 'xlua'
require 'KLDPenalty'
require 'Sampler'
assert(pcall(function () mat = require('fb.mattorch') end) or pcall(function() mat = require('matio') end), 'no mat IO interface available')

cmd = torch.CmdLine()
cmd:option('-gpu', 0, 'GPU id, starting from 1. Set it to 0 to run it in CPU mode. ')
cmd:option('-sample', false, 'whether to sample input latent vectors from an i.i.d. uniform distribution, or to generate shapes with demo vectors')
cmd:option('-normal', false, 'whether to sample from a uniform dist (false) or a normal (true)')
cmd:option('-bs', 100, 'batch size')
cmd:option('-ss', 100, 'number of generated shapes, only used in `-sample` mode')
cmd:option('-ck', 25, 'Checkpoint to start from (default is 25)')
cmd:option('-ckp','checkpoints_32table80', 'the name of the checkpoint folder')
cmd:option('-nc', 0, 'the number of categories (default is 0 for non conditional model)')
cmd:option('-cat', 0, 'the category to specify (not necessary if nc = 0)')

opt = cmd:parse(arg or {})
if opt.gpu > 0 then
  require 'cunn'
  require 'cudnn'
  require 'cutorch'
  cutorch.setDevice(opt.gpu)
  nn.DataParallelTable.deserializeNGPUs = 1
end

checkpoint_path = '/data/jjliu/checkpoints/' .. opt.ckp
print(checkpoint_path)

print('Loading network..')
gen_path = paths.concat(checkpoint_path, 'shapenet101_' .. opt.ck .. '_net_G.t7')
disc_path = paths.concat(checkpoint_path, 'shapenet101_' .. opt.ck .. '_net_D.t7') -- for retrieving meta info
netG = torch.load(gen_path)
netD = torch.load(disc_path)
print(netD)
-- only if originally saved as parallel model
--netG = netG:get(1)

if opt.gpu == 0 then
  netG = netG:double()
  netD = netD:double()
end

print(netD)

print(netG)
for i,module in ipairs(netG:listModules()) do
  if i == 2 then
    paramsG = module:getParameters()
    print(module)
    print(paramsG[{{1,100}}])
  end
end
nz = 200
--nz = netG:get(1).nInputPlane
netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)     -- convolution bias is removed during training
netG:evaluate() -- batch normalization behaves differently during evaluation

print('Setting inputs..')
inputs = torch.rand(opt.ss, nz + opt.nc, 1, 1, 1)
if opt.normal then
  inputs:normal(0,1)
end
input = torch.Tensor(opt.bs, nz + opt.nc, 1, 1, 1)
if opt.gpu > 0 then
  netG = netG:cuda()
  netG = cudnn.convert(netG, cudnn)
  input = input:cuda()
end
testinput = netG:forward(input)
dim = testinput:size()[3]
--results = torch.zeros(opt.ss, 1, 64, 64, 64):double()
results = torch.zeros(opt.ss, 1, dim, dim, dim):double()

if opt.nc > 0 then
  for i = 1, opt.ss do
    inputs[{i,{nz + 1, nz + opt.nc}}]:fill(0)
    inputs[{i, nz + opt.cat}]:fill(1)
  end
end

print('Forward prop')
for i = 1, math.ceil(opt.ss / opt.bs) do
  ind_low = (i-1)*opt.bs + 1
  ind_high = math.min(opt.ss, i * opt.bs)
  input:zero() 
  input[{{1,ind_high-ind_low+1},{},{},{},{}}] = inputs[{{ind_low,ind_high},{},{},{},{}}]
  res = netG:forward(input):double()
  results[{{ind_low,ind_high},{},{},{},{}}] = res[{{1,ind_high-ind_low+1},{},{},{},{}}]
  --print(res[1])
end

print('Saving result')
if paths.dir('./output/') == nil then
  paths.mkdir('output')
end
cur_time = os.time()
cur_times = '' .. cur_time
fname = 'sample_' ..cur_times .. '_' .. opt.ck .. '_' .. opt.ss
fullfname = paths.concat('./output', fname)
print(results:size())
mat.save(fullfname .. '.mat', {['inputs'] = inputs, ['voxels'] = results}) 
--for i = 1,opt.ss do
  --mat.save(fullfname .. '_' .. i .. '.mat', {['off_volume'] = results[{i,1}]})
--end
print('saving done')

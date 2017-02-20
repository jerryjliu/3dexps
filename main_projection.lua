require 'torch'
require 'nn'
require 'optim'
require 'paths'
assert(pcall(function () mat = require('fb.mattorch') end) or pcall(function() mat = require('matio') end), 'no mat IO interface available')

opt = {
  genEpoch = 580,
  leakyslope = 0.2,
  plr = 0.00005,
  beta1 = 0.5,
  batchSize = 100,
  --nout = 32,
  nz = 200,
  niter = 25,
  gpu = 2,
  name = 'shapenet101',
  cache_dir = '/data/jjliu/cache/',
  data_dir = '/data/jjliu/models/',
  data_name = 'full_dataset_voxels_32_chair',
  checkpointd = '/data/jjliu/checkpoints/',
  gen_checkpointf='checkpoints_64chair100o',
  out_ext = '',
  checkpointn = 0,
  is32 = 1,
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end 

if opt.is32 == 1 then
  opt.nout = 32
else
  opt.nout = 64
end

if opt.gpu > 0 then
  require 'cunn'
  require 'cudnn'
  require 'cutorch'
  cutorch.setDevice(opt.gpu)
end

-- Initialize data loader --
local DataLoader = paths.dofile('data.lua')
print('Loading all models into memory...')
local data = DataLoader.new(opt)
print('data size: ' .. data:size())
----------------------------

real_label = 1
fake_label = 0

local function weights_init(m)
  local name = torch.type(m)
  if name:find('Convolution') then
    --m.weight:normal(0.0, 0.02)
    --m.weight:normal(0.0, 0.4)
    
    fan_in = m.kW * m.kT * m.kH * m.nInputPlane
    fan_out = m.kW * m.kT * m.kH * m.nOutputPlane
    std = math.sqrt(4 / (fan_in + fan_out))
    m.weight:normal(0.0, std)
    print(m)
    print(std)
    if m.bias then 
      m.bias:fill(0) 
    end
  elseif name:find('BatchNormalization') then
    --if m.weight then m.weight:fill(0) end
    --if m.bias then m.bias:fill(0) end
  end
end

if opt.is32 == 0 then
  net = paths.dofile('net64.lua')
else
  net = paths.dofile('net32.lua')
end
opt.checkpointf = opt.gen_checkpointf .. '_p' .. opt.genEpoch
if opt.out_ext ~= '' then
  opt.checkpointf = opt.checkpointf .. '_' .. opt.out_ext
end
-- Generator (decoder) to use
print(opt.name .. '_' .. opt.genEpoch .. '_net_G.t7')
local netG = torch.load(paths.concat(opt.checkpointd .. opt.gen_checkpointf, opt.name .. '_' .. opt.genEpoch .. '_net_G.t7'))
--netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)     -- convolution bias is removed during training
netG:training()
--netG:evaluate() -- batch normalization behaves differently during evaluation

-- Projection network
local netP = net.netP
netP:apply(weights_init)
optimStateP = {
  learningRate = opt.plr,
  beta1 = opt.beta1,
}
local criterion = nn.BCECriterion()
local input = torch.Tensor(opt.batchSize, 1, opt.nout, opt.nout, opt.nout)
local label = torch.Tensor(opt.batchSize)
local errP
if opt.gpu > 0 then
  input = input:cuda()
  label = label:cuda()
  criterion = criterion:cuda()
  netP = netP:cuda()
  netP = cudnn.convert(netP, cudnn)
  netG = netG:cuda()
  netG = cudnn.convert(netG, cudnn)
end

local parametersP, gradParametersP = netP:getParameters()
-- update step Adam optim
local fPx = function(x)
  netP:zeroGradParameters()
  local sample, labels = data:getBatch(opt.batchSize)
  local actualBatchSize = sample:size(1)
  input[{{1,actualBatchSize}}]:copy(sample)
  local latent = netP:forward(input[{{1,actualBatchSize}}])

  local projout = netG:forward(latent)
  errP = criterion:forward(projout, input[{{1,actualBatchSize}}])
  local df_do = criterion:backward(projout, input[{{1,actualBatchSize}}])
  local df_dz = netG:updateGradInput(latent, df_do)
  netP:backward(input[{{1,actualBatchSize}}], df_dz)

  return errP, gradParametersP
end

begin_epoch = opt.checkpointn + 1
for epoch = begin_epoch, opt.niter do
  data:resetAndShuffle()
  for i = 1, data:size(), opt.batchSize do
    -- for each batch, first generate 50 generated samples and compute
    -- BCE loss on generator and discriminator
    print('Optimizing proj network')
    optim.adam(fPx, parametersP, optimStateP)
    -- logging
    print(('Epoch: [%d][%8d / %8d]\t Err_P: %.4f'):format(epoch, (i-1)/(opt.batchSize), math.floor(data:size()/(opt.batchSize)),errP))
  end
  if paths.dir(opt.checkpointd .. opt.checkpointf) == nil then
    paths.mkdir(opt.checkpointd .. opt.checkpointf)
  end
  parametersP, gradParametersP = nil,nil
  projCheckFile = opt.name .. '_' .. opt.genEpoch .. '_' .. epoch .. '_net_P.t7'
  torch.save(paths.concat(opt.checkpointd .. opt.checkpointf, projCheckFile), netP:clearState())
  parametersP, gradParametersP = netP:getParameters()
  print(('End of epoch %d / %d'):format(epoch, opt.niter))
end

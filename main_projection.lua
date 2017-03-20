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
  nc=101,
  niter = 25,
  gpu = 2,
  name = 'shapenet101',
  cache_dir = '/data/jjliu/cache/',
  data_dir = '/data/jjliu/models/',
  data_name = 'full_dataset_voxels_32_chair',
  checkpointd = '/data/jjliu/checkpoints/',
  gen_checkpointf='checkpoints_64chair100o',
  feat_checkpointf='checkpoints_64class100',
  feat_epochf = 'shapenet101_25_net_C_split7',
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

-- Classifier to use (for their feature space)
if opt.feat_checkpointf == '' then
  opt.feat_checkpointf = nil
end
local netF = nil
if opt.feat_checkpointf ~= nil then
  netF = torch.load(paths.concat(opt.checkpointd .. opt.feat_checkpointf, opt.feat_epochf .. '.t7'))
  netF:training()
  print(netF)
end

-- Projection network
local netP = net.netP
netP:apply(weights_init)
optimStateP = {
  learningRate = opt.plr,
  beta1 = opt.beta1,
}
if opt.checkpointn > 0 then
  netP = torch.load(paths.concat(opt.checkpointd .. opt.checkpointf, opt.name .. '_' .. opt.checkpointn .. '_net_P.t7'))
  optimStateC = torch.load(paths.concat(opt.checkpointd .. opt.checkpointf, opt.name .. '_' .. opt.checkpointn .. '_net_optimStateP.t7'))
end

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
  if netF ~= nil then
    netF = netF:cuda()
    netF = cudnn.convert(netF, cudnn)
  end
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
  local df_do
  if netF ~= nil then
    local featout = netF:forward(input[{{1,actualBatchSize}}])
    local projfeatout = netF:forward(projout)
    print(featout[{1,1,1,1,{20,30}}])
    print(projfeatout[{1,1,1,1,{20,30}}])
    errP = criterion:forward(projfeatout, featout)
    local df_dc = criterion:backward(projfeatout, featout)
    df_do = netF:updateGradInput(projout, df_dc)
  else
    errP = criterion:forward(projout, input[{{1,actualBatchSize}}])
    df_do = criterion:backward(projout, input[{{1,actualBatchSize}}])
  end
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
  optimStatePFile = opt.name .. '_' .. opt.genEpoch .. '_' .. epoch .. '_net_optimStateP.t7'
  torch.save(paths.concat(opt.checkpointd .. opt.checkpointf, projCheckFile), netP:clearState())
  torch.save(paths.concat(opt.checkpointd .. opt.checkpointf, optimStatePFile), optimStateP)
  parametersP, gradParametersP = netP:getParameters()
  print(('End of epoch %d / %d'):format(epoch, opt.niter))
end

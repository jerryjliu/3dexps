require 'torch'
require 'nn'
require 'optim'
require 'paths'
require 'evaluation/eval_utils'
assert(pcall(function () mat = require('fb.mattorch') end) or pcall(function() mat = require('matio') end), 'no mat IO interface available')

opt = {
  leakyslope = 0.2,
  clr = 0.00005,
  beta1 = 0.5,
  batchSize = 100,
  --nout = 32,
  nz = 200,
  nc = 101,
  niter = 25,
  gpu = 2,
  gpu2 = 3,
  name = 'shapenet101',
  cache_dir = '/data/jjliu/cache/',
  data_dir = '/data/jjliu/models/',
  data_name = 'full_dataset_voxels_32_chair',
  checkpointd = '/data/jjliu/checkpoints/',
  checkpointf = 'checkpoints_64class100',
  checkpointn = 0,
  is32 = 1,
  ctype = 'normal', -- cases: normal, voxception
  nmomentum = 0,
  rotated=0,
  orig_data_path='full_dataset_voxels_64_r8', -- only used if rotated > 0, provides data path of full rotated directory
  contains_split=1,
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
  elseif name:find('Linear') then
    m.weight:normal(0.0, 0.01)
  end
end

if opt.is32 == 0 then
  net = paths.dofile('net64.lua')
else
  net = paths.dofile('net32.lua')
end

-- Projection network
local netC = net.netC
if opt.ctype == 'voxception' then
  netC = net.netC_Vox
end
netC:apply(weights_init)
print(netC)
if opt.gpu2 > 0 then
  print(opt.gpu)
  print(opt.gpu2)
  tempnet = nn.DataParallelTable(1)
  tempnet:add(netC, {opt.gpu, opt.gpu2})
  netC = tempnet
end

optimStateC = {
  learningRate = opt.clr,
  beta1 = opt.beta1,
}
if opt.nmomentum > 0 then
  optimStateC.nesterov = true
  optimStateC.momentum = opt.nmomentum
  optimStateC.dampening = 0
end

if opt.checkpointn > 0 then
  netC = torch.load(paths.concat(opt.checkpointd .. opt.checkpointf, opt.name .. '_' .. opt.checkpointn .. '_net_C.t7'))
  optimStateC = torch.load(paths.concat(opt.checkpointd .. opt.checkpointf, opt.name .. '_' .. opt.checkpointn .. '_net_optimStateC.t7'))
end

local criterion = nn.CrossEntropyCriterion()
local input = torch.Tensor(opt.batchSize, 1, opt.nout, opt.nout, opt.nout)
local label = torch.Tensor(opt.batchSize)
local errC
local valError = 0
local valClassError = 0
if opt.gpu > 0 then
  input = input:cuda()
  label = label:cuda()
  criterion = criterion:cuda()
  netC = netC:cuda()
  netC = cudnn.convert(netC, cudnn)
end

local parametersC, gradParametersC = netC:getParameters()
-- update step Adam optim
local fCx = function(x)
  netC:zeroGradParameters()
  local sample, labels = data:getBatchUniformSample(opt.batchSize)
  local actualBatchSize = sample:size(1)
  input[{{1,actualBatchSize}}]:copy(sample)
  local output = netC:forward(input[{{1,actualBatchSize}}])

  -- compute training accuracy
  local train_accuracy = compute_accuracy(output,labels)
  print(('train batch accuracy: %.4f'):format(train_accuracy))
  local train_class_accuracy = compute_class_weighted_accuracy(output,labels,opt.nc)
  print(('train class accuracy: %.4f'):format(train_class_accuracy))

  errC = criterion:forward(output, labels)
  local df_do = criterion:backward(output, labels)
  netC:backward(input[{{1,actualBatchSize}}], df_do)
  return errC, gradParametersC
end

function measure_validation_error(data, opt)
  netC:evaluate()
  local valset_models, valset_labels = data:loadValidationSet()
  if valset_models == nil then
    return
  end
  local accuracy = 0
  local class_accuracy = 0
  for i = 1, math.ceil(valset_models:size(1) / opt.batchSize) do
    print(('processing %d/%d'):format(i, math.ceil(valset_models:size(1)/opt.batchSize)))
    local ind_low = (i-1)*opt.batchSize + 1
    local ind_high = math.min(valset_models:size(1), i * opt.batchSize)
    input:zero()
    input[{{1,ind_high-ind_low+1},{},{},{},{}}] = valset_models[{{ind_low,ind_high},{},{},{},{}}]
    local res = netC:forward(input):double()
    local curAccuracy = compute_accuracy(res, valset_labels[{{ind_low, ind_high}}])
    local curClassAccuracy = compute_class_weighted_accuracy(res, valset_labels[{{ind_low, ind_high}}], opt.nc)
    --print(curAccuracy)
    accuracy = accuracy + (curAccuracy * (ind_high - ind_low + 1)/(valset_models:size(1)))
    class_accuracy = class_accuracy + (curClassAccuracy * (ind_high - ind_low + 1)/(valset_models:size(1)))
    print(accuracy)
    print(class_accuracy)
  end
  netC:training()
  print('ACCURACY: ' .. accuracy)
  print('CLASS ACCURACY: ' .. class_accuracy)
  assert(math.abs(accuracy) <= 1)
  return accuracy, class_accuracy
end

begin_epoch = opt.checkpointn + 1
for epoch = begin_epoch, opt.niter do
  data:resetAndShuffle()
  tmpAcc, tmpClassAcc = measure_validation_error(data, opt)
  valError = 1 - tmpAcc
  valClassError = 1 - tmpClassAcc
  for i = 1, data:size(), opt.batchSize do
    -- for each batch, first generate 50 generated samples and compute
    -- BCE loss on generator and discriminator
    print('Optimizing proj network')
    if opt.nmomentum > 0 then
      optim.sgd(fCx, parametersC, optimStateC)
    else
      optim.adam(fCx, parametersC, optimStateC)
    end
    -- logging
    print(('Validation Error: %.4f'):format(valError))
    print(('Validation Class Error: %.4f'):format(valClassError))
    print(('Epoch: [%d][%8d / %8d]\t Err_C: %.4f'):format(epoch, (i-1)/(opt.batchSize), math.floor(data:size()/(opt.batchSize)),errC))
  end
  if paths.dir(opt.checkpointd .. opt.checkpointf) == nil then
    paths.mkdir(opt.checkpointd .. opt.checkpointf)
  end
  parametersC, gradParametersC = nil,nil
  netCheckFile = opt.name .. '_' .. epoch .. '_net_C.t7'
  optimStateCFile = opt.name .. '_' .. epoch .. '_net_optimStateC.t7'
  torch.save(paths.concat(opt.checkpointd .. opt.checkpointf, netCheckFile), netC:clearState())
  torch.save(paths.concat(opt.checkpointd .. opt.checkpointf, optimStateCFile), netC:clearState())
  parametersC, gradParametersC = netC:getParameters()

  print(('End of epoch %d / %d'):format(epoch, opt.niter))
end

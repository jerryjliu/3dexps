require 'torch'
require 'nn'
require 'optim'
require 'paths'
assert(pcall(function () mat = require('fb.mattorch') end) or pcall(function() mat = require('matio') end), 'no mat IO interface available')

opt = {
  leakyslope = 0.2,
  glr = 0.001,
  dlr = 0.00008,
  beta1 = 0.5,
  batchSize = 40,
  nz = 200,
  nc = 7,
  niter = 25,
  gpu = 2,
  gpu2 = 0,
  name = 'shapenet101',
  cache_dir = '/data/jjliu/cache/',
  data_dir = '/data/jjliu/models/',
  data_name = 'full_dataset_voxels_32_chair',
  checkpointd = '/data/jjliu/checkpoints/',
  checkpointf = 'checkpoints_32chair40ld',
  checkpointn = 0,
  is32 = 1,
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end 

print(opt.checkpointf)

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

netBuilder = require 'netbuilder'
net = netBuilder.buildnet(opt)
-- Generator
local netG = net.netG
netG:apply(weights_init)
-- Discriminator (same as Generator but uses LeakyReLU)
local netD = net.netD
netD:apply(weights_init)

if opt.gpu2 > 0 then
  tempnet = nn.DataParallelTable(1)
  tempnet:add(netG, {opt.gpu, opt.gpu2})
  netG = tempnet

  tempnet = nn.DataParallelTable(1)
  tempnet:add(netD, {opt.gpu, opt.gpu2})
  netD = tempnet
end

if opt.checkpointn > 0 then
  netG = torch.load(paths.concat(opt.checkpointd .. opt.checkpointf, opt.name .. '_' .. opt.checkpointn .. '_net_G.t7'))
  netD = torch.load(paths.concat(opt.checkpointd .. opt.checkpointf, opt.name .. '_' .. opt.checkpointn .. '_net_D.t7'))
end

optimStateG = {
  learningRate = opt.glr,
  beta1 = opt.beta1,
}
optimStateD = {
  learningRate = opt.dlr,
  beta1 = opt.beta1,
}

-------------------------------------------------
-- put all cudnn-enabled variables here --
-- ex: input, noise, label, errG, errD, (epoch_tm, tm, data_tm - purely for timing purposes)
-- criterion
--local criterion = nn.BCECriterion()
--local ceCriterion = nn.CrossEntropyCriterion()
local nllCriterion = nn.ClassNLLCriterion()
local bceCriterion = nn.BCECriterion()
-- input to discriminator
local input = torch.Tensor(opt.batchSize, 1, opt.nout, opt.nout, opt.nout)
-- input to generator
local noise = torch.Tensor(opt.batchSize, opt.nz, 1, 1, 1)
-- label tensor (used in training)
local label = torch.Tensor(opt.batchSize)
local fake_label = opt.nc + 1

local errG, errD
-------------------------------------------------
if opt.gpu > 0 then
  input = input:cuda()
  noise = noise:cuda()
  label = label:cuda()
  netG = netG:cuda()
  netG = cudnn.convert(netG, cudnn)
  netD = netD:cuda()
  netD = cudnn.convert(netD, cudnn)
  --criterion = criterion:cuda()
  --ceCriterion = ceCriterion:cuda()
  nllCriterion = nllCriterion:cuda()
  bceCriterion = bceCriterion:cuda()
end
local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

function compute_generator_loss(output_logits)
  local binTensor = torch.Tensor(output_logits:size(1))
  local binLabel = torch.Tensor(output_logits:size(1))
  softMax = nn.SoftMax()
  if opt.gpu > 0 then
    softMax = softMax:cuda()
  end
  output_logits = softMax:forward(output_logits)
  --print(output_logits[1])
  for i = 1, output_logits:size(1) do
    fake_prob = output_logits[{i,output_logits:size(2)}]
    real_prob = 1 - fake_prob
    --print(real_prob)
    --print(torch.sum(output_logits[{i,{1,output_logits:size(2)-1}}]))
    assert(math.abs(torch.sum(output_logits[{i,{1,output_logits:size(2)-1}}]) - real_prob) <= 0.0001)
    binTensor[{i}] = real_prob
  end
  binLabel:fill(1)
  local errG = bceCriterion:forward(binTensor, binLabel)
  return errG
end


function compute_accuracy(output_logits, is_real)
  softMax = nn.SoftMax()
  if opt.gpu > 0 then
    softMax = softMax:cuda()
  end
  output_logits = softMax:forward(output_logits)
  numCorrect = 0
  for i = 1, output_logits:size(1) do
    fake_prob = output_logits[{i,output_logits:size(2)}]
    if fake_prob <= 0.5 and is_real then
      numCorrect = numCorrect + 1
    elseif fake_prob > 0.5 and not is_real then
      numCorrect = numCorrect + 1
    end
  end
  return (numCorrect / output_logits:size(1))
end

function nn.Module:getSerialModel()
  if opt.gpu2 > 0 then
    return self:get(1)
  end
  return self
end

-- evaluate f(X), df/dX, discriminator
local fDx = function(x)

  netD:zeroGradParameters()

  print('getting real batch')
  local numCorrect = 0
  local real, rclasslabels = data:getBatch(opt.batchSize)
  local actualBatchSize = real:size(1)
  input[{{1,actualBatchSize}}]:copy(real)
  --label:fill(real_label)
  label[{{1,actualBatchSize}}]:copy(rclasslabels)
  local rout = netD:forward(input[{{1,actualBatchSize}}])
  local errD_real = nllCriterion:forward(rout, label[{{1,actualBatchSize}}])
  local df_do = nllCriterion:backward(rout, label[{{1,actualBatchSize}}])
  netD:backward(input[{{1,actualBatchSize}}], df_do)

  real_accuracy = compute_accuracy(netD:getSerialModel():get(14).output, true)
  --for i = 1,rout:size(1) do
    --if rout[{i,1}] >= 0.5 then
      --numCorrect = numCorrect + 1
    --end
  --end

  print('getting fake batch')
  noise:uniform(0, 1)
  local fake = netG:forward(noise)
  input:copy(fake)
  label:fill(fake_label)
  local fout = netD:forward(input[{{1,actualBatchSize}}])
  local errD_fake = nllCriterion:forward(fout, label[{{1,actualBatchSize}}])
  local df_do = nllCriterion:backward(fout, label[{{1,actualBatchSize}}])
  netD:backward(input[{{1,actualBatchSize}}], df_do)

  fake_accuracy = compute_accuracy(netD:getSerialModel():get(14).output, false)

  --for i = 1,fout:size(1) do
    --if fout[{i,1}] < 0.5 then
      --numCorrect = numCorrect + 1
    --end
  --end

  local accuracy = (real_accuracy + fake_accuracy) / 2
  --local accuracy = (numCorrect/(2*opt.batchSize))
  print(('disc accuracy: %.4f'):format(accuracy))
  if accuracy > 0.8 then
    print('ZEROED')
    netD:zeroGradParameters()
  end

  print(rout:size())
  print(fout:size())
  print(errD_real)
  print(errD_fake)

  errD = errD + errD_real + errD_fake
  return errD, gradParametersD
end

-- evaluate f(X), df/dX, generator
local fGx = function(x)
  netG:zeroGradParameters()
  label:fill(fake_label)
  print('filled real label')
  local output = netD.output
  local outputSize = output:size(1)
  local tempoutput = netD:getSerialModel():get(14).output
  errG = compute_generator_loss(tempoutput)

  print('forwarding output')
  nllCriterion:forward(output, label[{{1,outputSize}}])
  --errG = errG + nllCriterion:forward(output, label)
  print('errG: ' .. errG)
  print('..forwarded')
  local df_do = nllCriterion:backward(output, label[{{1,outputSize}}])
  local df_dg = netD:updateGradInput(input[{{1,outputSize}}], df_do)
  print('updated discriminator gradient input')

  netG:backward(noise[{{1,outputSize}}], df_dg) -- negate gradient because in this case maximizing loss
  print('accumulated G')

  --print(-gradParametersG[{{1,10}}])

  return errG, -gradParametersG
end

begin_epoch = opt.checkpointn + 1

for epoch = begin_epoch, opt.niter do
  data:resetAndShuffle()
  for i = 1, data:size(), opt.batchSize do
    -- for each batch, first generate 50 generated samples and compute
    -- BCE loss on generator and discriminator
    errG = 0
    errD = 0
    print('Optimizing disc')
    optim.adam(fDx, parametersD, optimStateD)
    --netD:zeroGradParameters()
    if opt.gpu2 > 0 then
      netD:syncParameters()
    end
    print('Optimizing gen')
    optim.adam(fGx, parametersG, optimStateG)
    --netG:zeroGradParameters()
    if opt.gpu2 > 0 then
      netG:syncParameters()
    end
    -- logging
    print(('Epoch: [%d][%8d / %8d]\t Err_G: %.4f Err_D: %.4f'):format(epoch, (i-1)/(opt.batchSize), math.floor(data:size()/(opt.batchSize)),errG, errD))
  end
  if paths.dir(opt.checkpointd .. opt.checkpointf) == nil then
    paths.mkdir(opt.checkpointd .. opt.checkpointf)
  end
  parametersD, gradParametersD = nil,nil
  parametersG, gradParametersG = nil,nil
  genCheckFile = opt.name .. '_' .. epoch .. '_net_G.t7'
  disCheckFile = opt.name .. '_' .. epoch .. '_net_D.t7'
  torch.save(paths.concat(opt.checkpointd .. opt.checkpointf, genCheckFile), netG:clearState())
  torch.save(paths.concat(opt.checkpointd .. opt.checkpointf, disCheckFile), netD:clearState())
  
  parametersD, gradParametersD = netD:getParameters()
  parametersG, gradParametersG = netG:getParameters()
  print(('End of epoch %d / %d'):format(epoch, opt.niter))
end

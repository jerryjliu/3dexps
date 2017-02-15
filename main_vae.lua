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
  nskip = 5,
  is32 = 1,
  nc = 0,
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
  end
end

if opt.is32 == 0 then
  net = paths.dofile('net64.lua')
else
  net = paths.dofile('net32.lua')
end
-- Generator
local netG = net.netG
netG:apply(weights_init)
-- Discriminator (same as Generator but uses LeakyReLU)
local netD = net.netD
netD:apply(weights_init)
-- Encoder (same structure as Discriminator)
local netE = net.netE
netE:apply(weights_init)

if opt.gpu2 > 0 then
  tempnet = nn.DataParallelTable(1)
  tempnet:add(netG, {opt.gpu, opt.gpu2})
  netG = tempnet

  tempnet = nn.DataParallelTable(1)
  tempnet:add(netD, {opt.gpu, opt.gpu2})
  netD = tempnet

  tempnet = nn.DataParallelTable(1)
  tempnet:add(netE, {opt.gpu, opt.gpu2})
  netE = tempnet
end

optimStateG = {
  learningRate = opt.glr,
  beta1 = opt.beta1,
}
optimStateD = {
  learningRate = opt.dlr,
  beta1 = opt.beta1,
}
optimStateE = {
  learningRate = opt.elr,
  beta1 = opt.beta1,
}

if opt.checkpointn > 0 then
  netG = torch.load(paths.concat(opt.checkpointd .. opt.checkpointf, opt.name .. '_' .. opt.checkpointn .. '_net_G.t7'))
  netD = torch.load(paths.concat(opt.checkpointd .. opt.checkpointf, opt.name .. '_' .. opt.checkpointn .. '_net_D.t7'))
  netE = torch.load(paths.concat(opt.checkpointd .. opt.checkpointf, opt.name .. '_' .. opt.checkpointn .. '_net_E.t7'))
  optimStateG = torch.load(paths.concat(opt.checkpointd .. opt.checkpointf, opt.name .. '_' .. opt.checkpointn .. '_net_optimStateG.t7'))
  optimStateD = torch.load(paths.concat(opt.checkpointd .. opt.checkpointf, opt.name .. '_' .. opt.checkpointn .. '_net_optimStateD.t7'))
  optimStateE = torch.load(paths.concat(opt.checkpointd .. opt.checkpointf, opt.name .. '_' .. opt.checkpointn .. '_net_optimStateE.t7'))
end


-------------------------------------------------
-- put all cudnn-enabled variables here --
-- ex: input, noise, label, errG, errD, (epoch_tm, tm, data_tm - purely for timing purposes)
-- criterion
local criterion = nn.BCECriterion()
-- real batch - input to encoder
local real = torch.Tensor(opt.batchSize, 1, opt.nout, opt.nout, opt.nout)
local actualBatchSize = 0
-- projected noise from encoder - different from noise in that noise is sampled from p(z) whereas projnoise is sampled from q(z | x)
local projnoise = torch.Tensor(opt.batchSize, opt.nz, 1, 1, 1)
-- input to discriminator
local input = torch.Tensor(opt.batchSize, 1, opt.nout, opt.nout, opt.nout)
-- input to generator
local noise = torch.Tensor(opt.batchSize, opt.nz, 1, 1, 1)
-- label tensor (used in training)
local label = torch.Tensor(opt.batchSize)

local errE, errG, errD
-------------------------------------------------
if opt.gpu > 0 then
  input = input:cuda()
  noise = noise:cuda()
  label = label:cuda()
  netG = netG:cuda()
  netG = cudnn.convert(netG, cudnn)
  netD = netD:cuda()
  netD = cudnn.convert(netD, cudnn)
  netE = netE:cuda()
  netE = cudnn.convert(netE, cudnn)
  criterion = criterion:cuda()
end
local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()
local parametersE, gradParametersE = netE:getParameters()

local fEx = function(x)
  netE:zeroGradParameters()
  netG:zeroGradParameters()

  local realBatch, rclasslabels = data:getBatch(opt.batchSize)
  actualBatchSize = realBatch:size(1)
  realBatch[{{1,actualBatchSize}}]:copy(real)
  projnoise[{{1,actualBatchSize}}]:copy(netE:forward(realBatch[{{1,actualBatchSize}}]))
  local tempgen = netG:forward(projnoise[{{1,actualBatchSize}}])
  errE = criterion:forward(tempgen)
  local df_do = criterion:backward(tempgen, recLoss)
  -- TODO: fix the magnitude of the contribution of the reconstruction loss to netG
  local df_de = netG:backward(projnoise[{{1,actualBatchSize}}], df_do)
  -- TODO: fix relative magnitudes of rec loss / KL divergence for netE
  netE:backward(realBatch[{{1,actualBatchSize}}], df_de)

  return errE, gradParametersE 
end

-- evaluate f(X), df/dX, discriminator
local fDx = function(x)
  netD:zeroGradParameters()

  print('getting real batch')
  local numCorrect = 0

  input[{{1,actualBatchSize}}]:copy(real)
  label:fill(real_label)
  local rout = netD:forward(input[{{1,actualBatchSize}}])
  local errD_real = criterion:forward(rout, label[{{1,actualBatchSize}}])
  local df_do = criterion:backward(rout, label[{{1,actualBatchSize}}])
  netD:backward(input[{{1,actualBatchSize}}], df_do)

  for i = 1,rout:size(1) do
    if rout[{i,1}] >= 0.5 then
      numCorrect = numCorrect + 1
    end
  end

  print('copying half of projnoise into noise, so samples from both p(z) and q(z | x)')
  noise:uniform(0,1)
  noise[{{1,actualBatchSize/2}}]:copy(projnoise[{{1,projBatchSize/2}}])
  local fake = netG:forward(noise[{{1,actualBatchSize}}])
  input[{{1, actualBatchSize}}]:copy(fake)
  label:fill(fake_label)
  local fout = netD:forward(input[{{1,actualBatchSize}}])
  local errD_fake = criterion:forward(fout, label[{{1,actualBatchSize}}])
  local df_do = criterion:backward(fout, label[{{1,actualBatchSize}}])
  netD:backward(input[{{1,actualBatchSize}}], df_do)
  for i = 1,fout:size(1) do
    if fout[{i,1}] < 0.5 then
      numCorrect = numCorrect + 1
    end
  end

  local accuracy = (numCorrect/(2*actualBatchSize))
  print(('disc accuracy: %.4f'):format(accuracy))
  if accuracy > 0.8 then
    print('ZEROED')
    netD:zeroGradParameters()
  end

  print(rout:size())
  print(fout:size())
  print(errD_real)
  print(errD_fake_proj .. ' ' .. errD_fake_rand)

  errD = errD_real + errD_fake
  return errD, gradParametersD
end

-- evaluate f(X), df/dX, generator
local fGx = function(x)
  --netG:zeroGradParameters() -- because already accumulated netG params in fEx
  label:fill(real_label)
  print('filled real label')
  local output = netD.output
  local outputSize = output:size(1)
  print('forwarding output')
  errG = criterion:forward(output, label[{{1,outputSize}}])
  print('errG: ' .. errG)
  print('..forwarded')
  local df_do = criterion:backward(output, label[{{1,outputSize}}])
  local df_dg = netD:updateGradInput(input[{{1,outputSize}}], df_do)
  print('updated discriminator gradient input')

  print(outputSize)

  netG:backward(noise[{{1,outputSize}}], df_dg)
  print('accumulated G')
  return errG, gradParametersG
end

begin_epoch = opt.checkpointn + 1

for epoch = begin_epoch, opt.niter do
  data:resetAndShuffle()
  for i = 1, data:size(), opt.batchSize do
    print('Optimizing encoder')
    optim.adam(fEx, parametersE, optimStateE)
    if opt.gpu2 > 0 then
      netE:syncParameters()
    end
    print('Optimizing disc')
    optim.adam(fDx, parametersD, optimStateD)
    if opt.gpu2 > 0 then
      netD:syncParameters()
    end
    print('Optimizing gen')
    optim.adam(fGx, parametersG, optimStateG)
    if opt.gpu2 > 0 then
      netG:syncParameters()
    end
    -- logging
    print(('Epoch: [%d][%8d / %8d]\t Err_G: %.4f Err_D: %.4f Err_E: %.4f'):format(epoch, (i-1)/(opt.batchSize), math.floor(data:size()/(opt.batchSize)),errG, errD, errE))
  end
  if paths.dir(opt.checkpointd .. opt.checkpointf) == nil then
    paths.mkdir(opt.checkpointd .. opt.checkpointf)
  end
  parametersD, gradParametersD = nil,nil
  parametersG, gradParametersG = nil,nil
  parametersE, gradParametersE = nil,nil
  genCheckFile = opt.name .. '_' .. epoch .. '_net_G.t7'
  disCheckFile = opt.name .. '_' .. epoch .. '_net_D.t7'
  encCheckFile = opt.name .. '_' .. epoch .. '_net_E.t7'
  optimStateGFile = opt.name .. '_' .. epoch .. '_net_optimStateG.t7'
  optimStateDFile = opt.name .. '_' .. epoch .. '_net_optimStateD.t7'
  optimStateEFile = opt.name .. '_' .. epoch .. '_net_optimStateE.t7'
  if epoch % opt.nskip == 0 then
    torch.save(paths.concat(opt.checkpointd .. opt.checkpointf, genCheckFile), netG:clearState())
    torch.save(paths.concat(opt.checkpointd .. opt.checkpointf, disCheckFile), netD:clearState())
    torch.save(paths.concat(opt.checkpointd .. opt.checkpointf, disCheckFile), netE:clearState())
    torch.save(paths.concat(opt.checkpointd .. opt.checkpointf, optimStateGFile), optimStateG)
    torch.save(paths.concat(opt.checkpointd .. opt.checkpointf, optimStateDFile), optimStateD)
    torch.save(paths.concat(opt.checkpointd .. opt.checkpointf, optimStateEFile), optimStateE)
  end
  parametersD, gradParametersD = netD:getParameters()
  parametersG, gradParametersG = netG:getParameters()
  parametersE, gradParametersE = netE:getParameters()
  print(('End of epoch %d / %d'):format(epoch, opt.niter))
end

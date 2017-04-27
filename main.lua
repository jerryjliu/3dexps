require 'torch'
require 'nn'
require 'optim'
require 'paths'
assert(pcall(function () mat = require('fb.mattorch') end) or pcall(function() mat = require('matio') end), 'no mat IO interface available')

opt = {
  leastsquares = false,
  leakyslope = 0.2,
  --glr = 0.002,
  --dlr = 0.00003,
  glr = 0.001, -- G=0.0025
  dlr = 0.00008, -- D=0.00001
  zsample = 'uniform1', -- 'normal', 'uniform1', 'uniform2'
  --glr = 0.00125,
  --dlr = 0.000005,
  --glr = 0.0021, -- scaling by 0.7 from 100
  --dlr = 8.3666e-6,
  --glr = 0.0025,
  --dlr = 0.00001,
  --glr = 0.00025,  these weights work when initializing with pretrained weights
  --dlr = 0.0001,  these weights work when initializing with pretrained weights
  --glr = 0.0025 * 0.6,
  --dlr = 0.00001 * 0.6,
  --glr = 0.0001,
  --dlr = 0.00007,
  beta1 = 0.5,
  --batchSize = 75,
  batchSize = 40, --batchSize = 100
  --nout = 64,
  --nout = 32,
  nz = 200,
  niter = 25,
  gpu = 2,
  gpu2 = 0,
  name = 'shapenet101',
  --checkpointf = 'checkpoints_table_cheat2'
  --checkpointf = 'checkpoints_chair_cheat'
  --checkpointf = '/data/jjliu/checkpoints/checkpoints_seven2',
  --checkpointf = '/data/jjliu/checkpoints/checkpoints_chair80_parallel',
  --checkpointf = '/data/jjliu/checkpoints/checkpoints_chair4',
  cache_dir = '/data/jjliu/cache/',
  data_dir = '/data/jjliu/models/',
  data_name = 'full_dataset_voxels_32_chair',
  checkpointd = '/data/jjliu/checkpoints/',
  checkpointf = 'checkpoints_32chair40ld',
  checkpointn = 0,
  nskip = 5,
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
      --m.bias:fill(0) 
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

optimStateG = {
  learningRate = opt.glr,
  beta1 = opt.beta1,
}
optimStateD = {
  learningRate = opt.dlr,
  beta1 = opt.beta1,
}

------------------------------
--If least squares GAN, then remove Sigmoid layer in discriminator
------------------------------
if opt.leastsquares then
  print("REMOVING SIGMOID")
  numLayers = netD:size()
  for i = 1, numLayers do 
    m = netD:get(i)
    local name = torch.type(m)
    if name:find('Sigmoid') then
      netD:remove(i)
      break
    end
  end
end

if opt.checkpointn > 0 then
  netG = torch.load(paths.concat(opt.checkpointd .. opt.checkpointf, opt.name .. '_' .. opt.checkpointn .. '_net_G.t7'))
  netD = torch.load(paths.concat(opt.checkpointd .. opt.checkpointf, opt.name .. '_' .. opt.checkpointn .. '_net_D.t7'))
  optimStateG = torch.load(paths.concat(opt.checkpointd .. opt.checkpointf, opt.name .. '_' .. opt.checkpointn .. '_net_optimStateG.t7'))
  optimStateD = torch.load(paths.concat(opt.checkpointd .. opt.checkpointf, opt.name .. '_' .. opt.checkpointn .. '_net_optimStateD.t7'))
end

if opt.gpu2 > 0 then
  tempnet = nn.DataParallelTable(1)
  tempnet:add(netG, {opt.gpu, opt.gpu2})
  netG = tempnet

  tempnet = nn.DataParallelTable(1)
  tempnet:add(netD, {opt.gpu, opt.gpu2})
  netD = tempnet
end



-------------------------------------------------
-- put all cudnn-enabled variables here --
-- ex: input, noise, label, errG, errD, (epoch_tm, tm, data_tm - purely for timing purposes)
-- criterion
local criterion = nn.BCECriterion()
if opt.leastsquares then
  criterion = nn.MSECriterion()
end
-- input to discriminator
local input = torch.Tensor(opt.batchSize, 1, opt.nout, opt.nout, opt.nout)
-- input to generator
local noise = torch.Tensor(opt.batchSize, opt.nz, 1, 1, 1)
-- label tensor (used in training)
local label = torch.Tensor(opt.batchSize)

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
  criterion = criterion:cuda()
end
local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

-- evaluate f(X), df/dX, discriminator
local fDx = function(x)

  netD:zeroGradParameters()

  print('getting real batch')
  local numCorrect = 0
  local real, rclasslabels = data:getBatch(opt.batchSize)

  local actualBatchSize = real:size(1)
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

  print('getting fake batch')
  --noise:uniform(0, 1)
  if opt.zsample == 'normal' then
    print('sampling normal dist')
    noise:normal(0,1)
  elseif opt.zsample == 'uniform1' then
    print('SHOULD NOT HIT THIS')
    noise:uniform(0,1)
  elseif opt.zsample == 'uniform2' then
    print('SHOULD NOT HIT THIS')
    noise:uniform(-1,1)
  end
  local fake = netG:forward(noise)
  input:copy(fake)
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
  print(errD_fake)

  errD = errD_real + errD_fake
  return errD, gradParametersD
end

-- evaluate f(X), df/dX, generator
local fGx = function(x)
  netG:zeroGradParameters()
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

  --print(gradParametersG[{{1,10}}])

  return errG, gradParametersG
  --netG:zeroGradParameters()
  --label:fill(fake_label)

  --templabel = torch.Tensor(opt.batchSize)
  --if opt.gpu > 0 then
    --templabel = templabel:cuda()
  --end
  --templabel:fill(real_label)

  --print('filled real label')
  --local output = netD.output
  --local outputSize = output:size(1)
  --print('forwarding output')
  --errG = criterion:forward(output, templabel[{{1,outputSize}}])
  --criterion:forward(output, label[{{1,outputSize}}])
  --print('errG: ' .. errG)
  --print('..forwarded')
  --local df_do = criterion:backward(output, label[{{1,outputSize}}])
  --local df_dg = netD:updateGradInput(input[{{1,outputSize}}], df_do)
  --print('updated discriminator gradient input')

  --netG:backward(noise[{{1,outputSize}}], df_dg)
  --print('accumulated G')

  ----print(gradParametersG[{{1,10}}])

  --return errG, -gradParametersG
end

begin_epoch = opt.checkpointn + 1

for epoch = begin_epoch, opt.niter do
  data:resetAndShuffle()
  for i = 1, data:size(), opt.batchSize do
    -- for each batch, first generate 50 generated samples and compute
    -- BCE loss on generator and discriminator
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
  optimStateGFile = opt.name .. '_' .. epoch .. '_net_optimStateG.t7'
  optimStateDFile = opt.name .. '_' .. epoch .. '_net_optimStateD.t7'
  if epoch % opt.nskip == 0 then
    torch.save(paths.concat(opt.checkpointd .. opt.checkpointf, genCheckFile), netG:clearState())
    torch.save(paths.concat(opt.checkpointd .. opt.checkpointf, disCheckFile), netD:clearState())
    torch.save(paths.concat(opt.checkpointd .. opt.checkpointf, optimStateGFile), optimStateG)
    torch.save(paths.concat(opt.checkpointd .. opt.checkpointf, optimStateDFile), optimStateD)
  end
  parametersD, gradParametersD = netD:getParameters()
  parametersG, gradParametersG = netG:getParameters()
  print(('End of epoch %d / %d'):format(epoch, opt.niter))
end

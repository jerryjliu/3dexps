require 'torch'
require 'nn'
require 'optim'
require 'paths'
assert(pcall(function () mat = require('fb.mattorch') end) or pcall(function() mat = require('matio') end), 'no mat IO interface available')

opt = {
  leakyslope = 0.2,
  glr = 0.0025,
  dlr = 0.00001,
  --glr = 0.0025,
  --dlr = 0.00001,
  --glr = 0.00025,  these weights work when initializing with pretrained weights
  --dlr = 0.0001,  these weights work when initializing with pretrained weights
  --glr = 0.0025 * 0.6,
  --dlr = 0.00001 * 0.6,
  --glr = 0.0001,
  --dlr = 0.00007,
  beta1 = 0.5,
  batchSize = 90,
  nout = 64,
  nz = 200,
  niter = 25,
  gpu = 1,
  name = 'shapenet101',
  --checkpointf = 'checkpoints_table_cheat2'
  --checkpointf = 'checkpoints_chair_cheat'
  checkpointf = 'checkpoints_chair5',
  checkpointn = 4
}

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

-- Generator
local netG = nn.Sequential() 
-- 200x1x1x1 -> 512x4x4x4
netG:add(nn.VolumetricFullConvolution(200,512,4,4,4))
netG:add(nn.VolumetricBatchNormalization(512))
netG:add(nn.ReLU())
-- 512x4x4x4 -> 256x8x8x8
netG:add(nn.VolumetricFullConvolution(512,256,4,4,4,2,2,2,1,1,1))
netG:add(nn.VolumetricBatchNormalization(256))
netG:add(nn.ReLU())
-- 256x8x8x8 -> 128x16x16x16
netG:add(nn.VolumetricFullConvolution(256,128,4,4,4,2,2,2,1,1,1))
netG:add(nn.VolumetricBatchNormalization(128))
netG:add(nn.ReLU())
-- 128x16x16x16 -> 64x32x32x32
netG:add(nn.VolumetricFullConvolution(128,64,4,4,4,2,2,2,1,1,1))
netG:add(nn.VolumetricBatchNormalization(64))
netG:add(nn.ReLU())
-- 64x32x32x32 -> 1x64x64x64
netG:add(nn.VolumetricFullConvolution(64,1,4,4,4,2,2,2,1,1,1))
netG:add(nn.Sigmoid())
netG:apply(weights_init)

-- Discriminator (same as Generator but uses LeakyReLU)
local netD = nn.Sequential()
-- 1x64x64x64 -> 64x32x32x32
netD:add(nn.VolumetricConvolution(1,64,4,4,4,2,2,2,1,1,1))
netD:add(nn.VolumetricBatchNormalization(64))
netD:add(nn.LeakyReLU(opt.leakyslope, true))
-- 64x32x32x32 -> 128x16x16x16
netD:add(nn.VolumetricConvolution(64,128,4,4,4,2,2,2,1,1,1))
netD:add(nn.VolumetricBatchNormalization(128))
netD:add(nn.LeakyReLU(opt.leakyslope, true))
-- 128x16x16x16 -> 256x8x8x8
netD:add(nn.VolumetricConvolution(128,256,4,4,4,2,2,2,1,1,1))
netD:add(nn.VolumetricBatchNormalization(256))
netD:add(nn.LeakyReLU(opt.leakyslope, true))
-- 256x8x8x8 -> 512x4x4x4
netD:add(nn.VolumetricConvolution(256,512,4,4,4,2,2,2,1,1,1))
netD:add(nn.VolumetricBatchNormalization(512))
netD:add(nn.LeakyReLU(opt.leakyslope, true))
-- 512x4x4x4 -> 1x1x1x1
netD:add(nn.VolumetricConvolution(512,1,4,4,4))
netD:add(nn.Sigmoid())
netD:add(nn.View(1):setNumInputDims(4))
netD:apply(weights_init)

if opt.checkpointn > 0 then
  netG = torch.load(paths.concat(opt.checkpointf, opt.name .. '_' .. opt.checkpointn .. '_net_G.t7'))
  netD = torch.load(paths.concat(opt.checkpointf, opt.name .. '_' .. opt.checkpointn .. '_net_D.t7'))
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
local criterion = nn.BCECriterion()
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
  gradParametersD:zero()

  print('getting real batch')
  local numCorrect = 0
  local real, rclasslabels = data:getBatch(opt.batchSize)
  input:copy(real)
  label:fill(real_label)
  local rout = netD:forward(input)
  local errD_real = criterion:forward(rout, label)
  local df_do = criterion:backward(rout, label)
  netD:backward(input, df_do)

  --local temptensor = torch.Tensor(2,1,opt.nout,opt.nout,opt.nout)
  --temptensor:copy(input[{{1,2},{},{},{},{}}])
  --local temptensor2 = torch.Tensor(2,opt.nz,1,1,1)
  --temptensor2:copy(noise[{{1,2},{},{},{},{}}])
  --mat.save('hello_' .. rclasslabels[1] .. '_' .. rclasslabels[2] .. '.mat', {['inputs'] = temptensor2, ['voxels'] = temptensor}) 

  for i = 1,rout:size(1) do
    if rout[{i,1}] >= 0.5 then
      numCorrect = numCorrect + 1
    end
  end

  print('getting fake batch')
  noise:uniform(0, 1)
  local fake = netG:forward(noise)
  input:copy(fake)
  label:fill(fake_label)
  local fout = netD:forward(input)
  local errD_fake = criterion:forward(fout, label)
  local df_do = criterion:backward(fout, label)
  netD:backward(input, df_do)

  for i = 1,fout:size(1) do
    if fout[{i,1}] < 0.5 then
      numCorrect = numCorrect + 1
    end
  end


  local accuracy = (numCorrect/(2*opt.batchSize))
  print(('disc accuracy: %.4f'):format(accuracy))
  --if accuracy > 0.8 then
    --print('ZEROED')
    --gradParametersD:zero()
  --end

  print(rout:size())
  print(fout:size())
  print(errD_real)
  print(errD_fake)

  errD = errD_real + errD_fake
  return errD, gradParametersD
end

-- evaluate f(X), df/dX, generator
local fGx = function(x)
  gradParametersG:zero()
  label:fill(real_label)

  print('filled real label')
  local output = netD.output
  print('forwarding output')
  errG = criterion:forward(output, label)
  print('errG: ' .. errG)
  print('..forwarded')
  local df_do = criterion:backward(output, label)
  local df_dg = netD:updateGradInput(input, df_do)
  print('updated discriminator gradient input')

  netG:backward(noise, df_dg)
  print('accumulated G')

  return errG, gradParametersG
end


begin_epoch = opt.checkpointn + 1

for epoch = begin_epoch, opt.niter do
  for i = 1, data:size(), opt.batchSize do
    -- for each batch, first generate 50 generated samples and compute
    -- BCE loss on generator and discriminator
    print('Optimizing disc')
    optim.adam(fDx, parametersD, optimStateD)
    print('Optimizing gen')
    optim.adam(fGx, parametersG, optimStateG)

    -- logging
    print(('Epoch: [%d][%8d / %8d]\t Err_G: %.4f Err_D: %.4f'):format(epoch, (i-1)/opt.batchSize, math.floor(data:size()/opt.batchSize),errG, errD))
  end
  if paths.dir(opt.checkpointf) == nil then
    paths.mkdir(opt.checkpointf)
  end
  parametersD, gradParametersD = nil,nil
  parametersG, gradParametersG = nil,nil
  genCheckFile = opt.name .. '_' .. epoch .. '_net_G.t7'
  disCheckFile = opt.name .. '_' .. epoch .. '_net_D.t7'
  torch.save(paths.concat(opt.checkpointf, genCheckFile), netG:clearState())
  torch.save(paths.concat(opt.checkpointf, disCheckFile), netD:clearState())
  
  parametersD, gradParametersD = netD:getParameters()
  parametersG, gradParametersG = netG:getParameters()
  print(('End of epoch %d / %d'):format(epoch, opt.niter))
end

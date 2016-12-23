require 'torch'
require 'nn'
require 'optim'
assert(pcall(function () mat = require('fb.mattorch') end) or pcall(function() mat = require('matio') end), 'no mat IO interface available')

model_path = '/data/models/full_dataset_voxels/';

opt = {
  leakyslope = 0.2,
  glr = 0.0025,
  dlr = 1e-5,
  beta1 = 0.5,
  batchSize = 100,
  nout = 64,
  nz = 200,
  niter = 25,
  gpu = 1,
  name = 'shapenet101'
}

if opt.gpu > 0 then
  require 'cunn'
  require 'cudnn'
  require 'cutorch'
  cutorch.setDevice(opt.gpu)
end

-- Initialize data loader --
local DataLoader = paths.dofile('data.lua')
local data = DataLoader.new(opt)
----------------------------

real_label = 1
fake_label = 0

local function weights_init(m)
  local name = torch.type(m)
  if name:find('Convolution') then
    m.weight:normal(0.0, 0.02)
    m.bias:fill(0)
  elseif name:find('BatchNormalization') then
    if m.weight then m.weight:normal(1.0, 0.02) end
    if m.bias then m.bias:fill(0) end
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

optimStateG = {
  learningRate = opt.glr,
  beta1 = opt.beta1,
}
optimStateD = {
  learningRate = opt.dlr,
  beta1 = opt.beta1
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
  netG = cudnn(netG, cudnn)
  netD = netD:cuda()
  netD = cudnn(netD, cudnn)
  criterion = criterion:cuda()
end
local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

-- evaluate f(X), df/dX, discriminator
local fDx = function(x)
  gradParametersD:zero()

  local real, rclasslabels = data:getBatch()
  input:copy(real)
  label:fill(real_label)
  local rout = netD:forward(input)
  local errD_real = criterion:forward(rout, label)
  local df_do = criterion:backward(rout, label)
  netD:backward(input, df_do)

  noise:uniform(-1, 1)
  local fake = netG:forward(noise)
  input:copy(fake)
  label:fill(fake_label)
  local fout = netD:forward(input)
  local errD_fake = criterion:forward(fout, label)
  local df_do = criterion:backward(fout, label)
  netD:backward(input, df_do)

  errD = errD_real + errD_fake
  return errD, gradParametersD
end

-- evaluate f(X), df/dX, generator
local fGx = function(x)
  gradParametersG:zero()
  label:fill(real_label)

  local output = netD.output
  errG = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  local df_dg = netD:updateGradInput(input, df_do)

  netG:backward(input, df_dg)

  return errG, gradParametersG
end


for epoch = 1, opt.niter do
  for i = 1, data:size(), opt.batchSize do
    -- for each batch, first generate 50 generated samples and compute
    -- BCE loss on generator and discriminator
    optim.adam(fDx, parametersD, optimStateD)
    optim.adam(fGx, parametersG, optimStateG)

    -- logging
    print(('Epoch: [%d][%8d / %8d]\t Err_G: %.4f Err_D: %.4f'):format(epoch, (i-1)/opt.batchSize, errG, errD))
  end
  paths.mkdir('checkpoints')
  parametersD, gradParametersD = nil,nil
  parametersG, gradParametersG = nil,nil
  torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG:clearState())
  torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD:clearState())
  
  parametersD, gradParametersD = netD:getParameters()
  parametersG, gradParametersG = netG:getParameters()
  print(('End of epoch %d / %d'):format(epoch, opt.niter))
end

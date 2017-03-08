-- function for generating custom layers for voxel-based deep networks

require 'nn'

vl = {}

function vl.Voxception(nDim, nInputPlanes, num_filters, opt)
  local leakyslope = opt.leakyslope or 0.1
  local cnn = nn.Concat(1)
  if nDim == 5 then
    cnn = nn.Concat(2)
  end
  local nn1 = nn.Sequential()
  nn1:add(nn.VolumetricConvolution(nInputPlanes,num_filters,3,3,3,1,1,1,1,1,1))
  nn1:add(nn.LeakyReLU(opt.leakyslope, true))
  local nn2 = nn.Sequential()
  nn2:add(nn.VolumetricConvolution(nInputPlanes,num_filters,1,1,1,1,1,1))
  nn2:add(nn.LeakyReLU(opt.leakyslope, true))
  cnn:add(nn1)
  cnn:add(nn2)
  return cnn
end

function vl.VoxceptionDown(nDim, nInputPlanes, num_filters, opt)
  local leakyslope = opt.leakyslope or 0.1
  local cnn = nn.Concat(1)
  if nDim == 5 then
    cnn = nn.Concat(2)
  end
  local nn1 = nn.Sequential()
  nn1:add(nn.VolumetricConvolution(nInputPlanes,num_filters,3,3,3,1,1,1,1,1,1))
  nn1:add(nn.VolumetricMaxPooling(2,2,2))
  nn1:add(nn.LeakyReLU(opt.leakyslope, true))
  local nn2 = nn.Sequential()
  nn2:add(nn.VolumetricConvolution(nInputPlanes,num_filters,1,1,1,1,1,1))
  nn2:add(nn.VolumetricMaxPooling(2,2,2))
  nn2:add(nn.LeakyReLU(opt.leakyslope, true))
  local nn3 = nn.Sequential()
  nn3:add(nn.VolumetricConvolution(nInputPlanes,num_filters,3,3,3,2,2,2,1,1,1))
  nn3:add(nn.LeakyReLU(opt.leakyslope, true))
  local nn4 = nn.Sequential()
  nn4:add(nn.VolumetricConvolution(nInputPlanes,num_filters,1,1,1,2,2,2))
  nn4:add(nn.LeakyReLU(opt.leakyslope, true))
  cnn:add(nn1)
  cnn:add(nn2)
  cnn:add(nn3)
  cnn:add(nn4)
  return cnn
end

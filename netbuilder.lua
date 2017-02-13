local netbuilder = {}
function netbuilder.buildnet(opt)
  if opt.is32 == 1 then
    return netbuilder.net32(opt)
  else
    return netbuilder.net64(opt)
  end
end

function netbuilder.net32(opt)
  -- Encoder (VAE) 
  local netE = nn.Sequential()
  -- 3x256x256 -> 64x64x64
  netE:add(nn.SpatialConvolution(3,64,11,11,4,4,4,4))
  netE:add(nn.SpatialBatchNormalization(64))
  netE:add(nn.ReLU())
  -- 64x64x64 -> 128x32x32
  netE:add(nn.SpatialConvolution(64,128,5,5,2,2,2,2))
  netE:add(nn.SpatialBatchNormalization(128))
  netE:add(nn.ReLU())
  -- 128x32x32 -> 256x16x16
  netE:add(nn.SpatialConvolution(128,256,5,5,2,2,2,2))
  netE:add(nn.SpatialBatchNormalization(256))
  netE:add(nn.ReLU())
  -- 256x16x16 -> 512x8x8
  netE:add(nn.SpatialConvolution(256,512,5,5,2,2,2,2))
  netE:add(nn.SpatialBatchNormalization(512))
  netE:add(nn.ReLU())
  -- 512x8x8 -> 400x1x1
  netE:add(nn.SpatialConvolution(512,400,8,8,1,1))


  -- Generator
  local netG = nn.Sequential() 
  -- 200x1x1x1 -> 256x4x4x4
  netG:add(nn.VolumetricFullConvolution(200,256,4,4,4))
  netG:add(nn.VolumetricBatchNormalization(256))
  netG:add(nn.ReLU())
  -- 256x4x4x4 -> 128x8x8x8
  netG:add(nn.VolumetricFullConvolution(256,128,4,4,4,2,2,2,1,1,1))
  netG:add(nn.VolumetricBatchNormalization(128))
  netG:add(nn.ReLU())
  -- 128x8x8x8 -> 64x16x16x16
  netG:add(nn.VolumetricFullConvolution(128,64,4,4,4,2,2,2,1,1,1))
  netG:add(nn.VolumetricBatchNormalization(64))
  netG:add(nn.ReLU())
  -- 64x16x16x16 -> 1x32x32x32
  netG:add(nn.VolumetricFullConvolution(64,1,4,4,4,2,2,2,1,1,1))
  netG:add(nn.Sigmoid())

  -- Discriminator (same as Generator but uses LeakyReLU)
  local netD = nn.Sequential()
  -- 1x32x32x32 -> 64x16x16x16
  --netD:add(nn.VolumetricDropout(0.2))
  netD:add(nn.VolumetricConvolution(1,64,4,4,4,2,2,2,1,1,1))
  netD:add(nn.VolumetricBatchNormalization(64))
  netD:add(nn.LeakyReLU(opt.leakyslope, true))
  -- 64x16x16x16 -> 128x8x8x8
  --netD:add(nn.VolumetricDropout(0.2))
  netD:add(nn.VolumetricConvolution(64,128,4,4,4,2,2,2,1,1,1))
  netD:add(nn.VolumetricBatchNormalization(128))
  netD:add(nn.LeakyReLU(opt.leakyslope, true))
  -- 128x8x8x8 -> 256x4x4x4
  --netD:add(nn.VolumetricDropout(0.2))
  netD:add(nn.VolumetricConvolution(128,256,4,4,4,2,2,2,1,1,1))
  netD:add(nn.VolumetricBatchNormalization(256))
  netD:add(nn.LeakyReLU(opt.leakyslope, true))
  -- 256x4x4x4 -> (opt.nc + 1)x1x1x1
  --netD:add(nn.VolumetricDropout(0.2))
  netD:add(nn.VolumetricConvolution(256,1 + opt.nc,4,4,4))
  netD:add(nn.View(1 + opt.nc):setNumInputDims(4))
  if opt.nc > 0 then
    netD:add(nn.LogSoftMax())
  else
    netD:add(nn.Sigmoid())
  end

  local net32 = {}
  net32.netG = netG
  net32.netD = netD

  return net32
end

function netbuilder.net64(opt)
  -- Generator
  local netG = nn.Sequential() 
  -- 200x1x1x1 -> 512x4x4x4
  netG:add(nn.VolumetricDropout(0.2))
  netG:add(nn.VolumetricFullConvolution(200 + opt.nc,512,4,4,4))
  netG:add(nn.VolumetricBatchNormalization(512))
  netG:add(nn.ReLU())
  -- 512x4x4x4 -> 256x8x8x8
  netG:add(nn.VolumetricDropout(0.5))
  netG:add(nn.VolumetricFullConvolution(512,256,4,4,4,2,2,2,1,1,1))
  netG:add(nn.VolumetricBatchNormalization(256))
  netG:add(nn.ReLU())
  -- 256x8x8x8 -> 128x16x16x16
  netG:add(nn.VolumetricDropout(0.5))
  netG:add(nn.VolumetricFullConvolution(256,128,4,4,4,2,2,2,1,1,1))
  netG:add(nn.VolumetricBatchNormalization(128))
  netG:add(nn.ReLU())
  -- 128x16x16x16 -> 64x32x32x32
  netG:add(nn.VolumetricDropout(0.5))
  netG:add(nn.VolumetricFullConvolution(128,64,4,4,4,2,2,2,1,1,1))
  netG:add(nn.VolumetricBatchNormalization(64))
  netG:add(nn.ReLU())
  -- 64x32x32x32 -> 1x64x64x64
  netG:add(nn.VolumetricFullConvolution(64,1,4,4,4,2,2,2,1,1,1))
  netG:add(nn.Sigmoid())

  -- Discriminator (same as Generator but uses LeakyReLU)
  local netD = nn.Sequential()
  -- 1x64x64x64 -> 64x32x32x32
  netD:add(nn.VolumetricConvolution(1,64,4,4,4,2,2,2,1,1,1))
  netD:add(nn.VolumetricBatchNormalization(64))
  netD:add(nn.LeakyReLU(opt.leakyslope, true))
  -- 64x32x32x32 -> 128x16x16x16
  netD:add(nn.VolumetricDropout(0.5))
  netD:add(nn.VolumetricConvolution(64,128,4,4,4,2,2,2,1,1,1))
  netD:add(nn.VolumetricBatchNormalization(128))
  netD:add(nn.LeakyReLU(opt.leakyslope, true))
  -- 128x16x16x16 -> 256x8x8x8
  netD:add(nn.VolumetricDropout(0.5))
  netD:add(nn.VolumetricConvolution(128,256,4,4,4,2,2,2,1,1,1))
  netD:add(nn.VolumetricBatchNormalization(256))
  netD:add(nn.LeakyReLU(opt.leakyslope, true))
  -- 256x8x8x8 -> 512x4x4x4
  netD:add(nn.VolumetricDropout(0.5))
  netD:add(nn.VolumetricConvolution(256,512,4,4,4,2,2,2,1,1,1))
  netD:add(nn.VolumetricBatchNormalization(512))
  netD:add(nn.LeakyReLU(opt.leakyslope, true))
  -- 512x4x4x4 -> 1x1x1x1
  netD:add(nn.VolumetricDropout(0.5))
  netD:add(nn.VolumetricConvolution(512,1+opt.nc,4,4,4))
  netD:add(nn.View(1 + opt.nc):setNumInputDims(4))
  if opt.nc > 0 then
    netD:add(nn.LogSoftMax())
  else
    netD:add(nn.Sigmoid())
  end

  net64 = {}
  net64.netG = netG
  net64.netD = netD
  return net64
end

return netbuilder

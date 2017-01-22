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
-- 256x4x4x4 -> 1x1x1x1
--netD:add(nn.VolumetricDropout(0.2))
netD:add(nn.VolumetricConvolution(256,1,4,4,4))
netD:add(nn.Sigmoid())
netD:add(nn.View(1):setNumInputDims(4))

net32 = {}
net32.netG = netG
net32.netD = netD

return net32

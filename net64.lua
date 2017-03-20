require 'voxlayers'

-- Generator
local netG = nn.Sequential() 
-- 200x1x1x1 -> 512x4x4x4
--netG:add(nn.VolumetricDropout(0.2))
netG:add(nn.VolumetricFullConvolution(200,512,4,4,4))
netG:add(nn.VolumetricBatchNormalization(512))
netG:add(nn.ReLU())
-- 512x4x4x4 -> 256x8x8x8
--netG:add(nn.VolumetricDropout(0.5))
netG:add(nn.VolumetricFullConvolution(512,256,4,4,4,2,2,2,1,1,1))
netG:add(nn.VolumetricBatchNormalization(256))
netG:add(nn.ReLU())
-- 256x8x8x8 -> 128x16x16x16
--netG:add(nn.VolumetricDropout(0.5))
netG:add(nn.VolumetricFullConvolution(256,128,4,4,4,2,2,2,1,1,1))
netG:add(nn.VolumetricBatchNormalization(128))
netG:add(nn.ReLU())
-- 128x16x16x16 -> 64x32x32x32
--netG:add(nn.VolumetricDropout(0.5))
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
--netD:add(nn.VolumetricDropout(0.5))
netD:add(nn.VolumetricConvolution(64,128,4,4,4,2,2,2,1,1,1))
netD:add(nn.VolumetricBatchNormalization(128))
netD:add(nn.LeakyReLU(opt.leakyslope, true))
-- 128x16x16x16 -> 256x8x8x8
--netD:add(nn.VolumetricDropout(0.5))
netD:add(nn.VolumetricConvolution(128,256,4,4,4,2,2,2,1,1,1))
netD:add(nn.VolumetricBatchNormalization(256))
netD:add(nn.LeakyReLU(opt.leakyslope, true))
-- 256x8x8x8 -> 512x4x4x4
--netD:add(nn.VolumetricDropout(0.5))
netD:add(nn.VolumetricConvolution(256,512,4,4,4,2,2,2,1,1,1))
netD:add(nn.VolumetricBatchNormalization(512))
netD:add(nn.LeakyReLU(opt.leakyslope, true))
-- 512x4x4x4 -> 1x1x1x1
--netD:add(nn.VolumetricDropout(0.5))
netD:add(nn.VolumetricConvolution(512,1,4,4,4))
netD:add(nn.Sigmoid())
netD:add(nn.View(1):setNumInputDims(4))

-- Projector (project output into latent space)
local netP = nn.Sequential()
-- 1x64x64x64 -> 64x32x32x32
netP:add(nn.VolumetricConvolution(1,64,4,4,4,2,2,2,1,1,1))
netP:add(nn.VolumetricBatchNormalization(64))
netP:add(nn.LeakyReLU(opt.leakyslope, true))
-- 64x32x32x32 -> 128x16x16x16
netP:add(nn.VolumetricConvolution(64,128,4,4,4,2,2,2,1,1,1))
netP:add(nn.VolumetricBatchNormalization(128))
netP:add(nn.LeakyReLU(opt.leakyslope, true))
-- 128x16x16x16 -> 256x8x8x8
netP:add(nn.VolumetricConvolution(128,256,4,4,4,2,2,2,1,1,1))
netP:add(nn.VolumetricBatchNormalization(256))
netP:add(nn.LeakyReLU(opt.leakyslope, true))
-- 256x8x8x8 -> 512x4x4x4
netP:add(nn.VolumetricConvolution(256,512,4,4,4,2,2,2,1,1,1))
netP:add(nn.VolumetricBatchNormalization(512))
netP:add(nn.LeakyReLU(opt.leakyslope, true))
-- 512x4x4x4 -> 200x1x1x1
netP:add(nn.VolumetricConvolution(512,200,4,4,4))
netP:add(nn.Sigmoid())


-- Volumetric Classifier (learn better features representations)
-- could be own adaptation of VoxNet or something different
local netC = nn.Sequential()
-- 1x64x64x64 -> 32x32x32x32
netC:add(nn.VolumetricConvolution(1,32,4,4,4,2,2,2,1,1,1))
netC:add(nn.VolumetricBatchNormalization(32))
netC:add(nn.LeakyReLU(opt.leakyslope, true))
netC:add(nn.VolumetricDropout(0.2))
-- 32x32x32x32 -> 32x15x15x15
netC:add(nn.VolumetricConvolution(32,32,5,5,5,2,2,2,1,1,1))
netC:add(nn.VolumetricBatchNormalization(32))
netC:add(nn.LeakyReLU(opt.leakyslope,true))
netC:add(nn.VolumetricDropout(0.3))
-- 32x15x15x15 -> 32x12x12x12
netC:add(nn.VolumetricConvolution(32,32,3,3,3,1,1,1))
netC:add(nn.VolumetricBatchNormalization(32))
netC:add(nn.LeakyReLU(opt.leakyslope,true))
-- 32x12x12x12 -> 32x6x6x6
netC:add(nn.VolumetricMaxPooling(2,2,2))
netC:add(nn.VolumetricBatchNormalization(32))
netC:add(nn.VolumetricDropout(0.4))
-- 32x6x6x6 -> 128
netC:add(nn.View(6912))
netC:add(nn.Linear(6912,128))
netC:add(nn.ReLU(true))
netC:add(nn.Dropout(0.5))
-- 128 -> 101
netC:add(nn.Linear(128,opt.nc or 101))

-- Voxception
local netC_Vox = nn.Sequential()
netC_Vox:add(vl.Voxception(5,1,16,opt))
netC_Vox:add(vl.VoxceptionDown(5,32,8,opt))
netC_Vox:add(nn.VolumetricDropout(0.2))
netC_Vox:add(vl.Voxception(5,32,16,opt))
netC_Vox:add(vl.VoxceptionDown(5,32,8,opt))
netC_Vox:add(nn.VolumetricDropout(0.4))
netC_Vox:add(vl.Voxception(5,32,32,opt))
netC_Vox:add(vl.VoxceptionDown(5,64,16,opt))
netC_Vox:add(nn.VolumetricDropout(0.5))
netC_Vox:add(vl.Voxception(5,64,64,opt))
netC_Vox:add(vl.VoxceptionDown(5,128,32,opt))
netC_Vox:add(nn.VolumetricDropout(0.5))
netC_Vox:add(nn.View(8192))
netC_Vox:add(nn.Linear(8192,128))
netC_Vox:add(nn.Dropout(0.5))
netC_Vox:add(nn.LeakyReLU(opt.leakyslope, true))
netC_Vox:add(nn.Linear(128,opt.nc or 101))

net64 = {}
net64.netG = netG
net64.netD = netD
net64.netP = netP
net64.netC = netC
net64.netC_Vox = netC_Vox

return net64

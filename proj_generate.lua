require 'nn'
require 'torch'
require 'xlua'
assert(pcall(function () mat = require('fb.mattorch') end) or pcall(function() mat = require('matio') end), 'no mat IO interface available')

cmd = torch.CmdLine()
cmd:option('-gpu', 0, 'GPU id, starting from 1. Set it to 0 to run it in CPU mode. ')
cmd:option('-input','test_chair1', 'the name of the input file')
cmd:option('-ckp', 'checkpoints_64chair_ref', 'checkpoint folder of gen model')
cmd:option('-ckgen', '888', 'checkpoint of the gen model')
cmd:option('-ckproj', '40', 'checkpoint of the projection model')

opt = cmd:parse(arg or {})
if opt.gpu > 0 then
  require 'cunn'
  require 'cudnn'
  require 'cutorch'
  cutorch.setDevice(opt.gpu)
end

data_dir = '/data/jjliu/models/proj_inputs_voxel/'
checkpoint_gen_path = '/data/jjliu/checkpoints/' .. opt.ckp
checkpoint_proj_path = '/data/jjliu/checkpoints/' .. opt.ckp .. '_p' .. opt.ckgen
print('Loading network..')
gen_path = paths.concat(checkpoint_gen_path, 'shapenet101_' .. opt.ckgen .. '_net_G.t7')
proj_path = paths.concat(checkpoint_proj_path, 'shapenet101_' .. opt.ckgen .. '_' .. opt.ckproj .. '_net_P.t7')
netG = torch.load(gen_path)
netP = torch.load(proj_path)
-- only if originally saved as parallel model
--netG = netG:get(1)

if opt.gpu == 0 then
  netG = netG:double()
  netP = netP:double()
end

nz = 200
netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)     -- convolution bias is removed during training
netG:evaluate() -- batch normalization behaves differently during evaluation
netP:evaluate()

print('Setting inputs..')
tmpinput = mat.load(paths.concat(data_dir, opt.input .. '.mat'), 'off_volume')
input = torch.Tensor(1,64,64,64)
testlatent = torch.Tensor(1,nz,1,1,1)
if opt.gpu > 0 then
  netG = netG:cuda()
  netG = cudnn.convert(netG, cudnn)
  netP = netP:cuda()
  netP = cudnn.convert(netP, cudnn)
  input = input:cuda()
  testlatent = testlatent:cuda()
end
testoutput = netG:forward(testlatent)
dim = testoutput:size()[3]
input:copy(tmpinput)
--input:uniform(0,1)

print('Forward prop')
latent = netP:forward(input)
--latent:uniform(0,1)
print(latent)
output = netG:forward(latent)
print('Saving result')
if paths.dir('./output/') == nil then
  paths.mkdir('output')
end
cur_time = os.time()
cur_times = '' .. cur_time
fname = 'proj_' ..cur_times .. '_' .. opt.ckgen .. '_' .. opt.ckproj
fullfname = paths.concat('./output', fname)
if opt.gpu > 0 then
  latent = latent:double()
  output = output:double()
end
mat.save(fullfname .. '.mat', {['inputs'] = latent, ['voxels'] = output}) 
print('saving done')

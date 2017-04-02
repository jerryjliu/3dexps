require 'nn'
require 'torch'
require 'xlua'
assert(pcall(function () mat = require('fb.mattorch') end) or pcall(function() mat = require('matio') end), 'no mat IO interface available')

cmd = torch.CmdLine()
cmd:option('-gpu', 0, 'GPU id, starting from 1. Set it to 0 to run it in CPU mode. ')
cmd:option('-input','test_chair1', 'the name of the input file')
cmd:option('-informat', 'mat', 'format of input object (mat, t7)')
cmd:option('-ckp', 'checkpoints_64chair_ref', 'checkpoint folder of gen model')
cmd:option('-ckgen', '888', 'checkpoint of the gen model')
cmd:option('-ckproj', '40', 'checkpoint of the projection model')
cmd:option('-ckext', '', 'extension to ckp to specify name of projection folder ( default is none )')
cmd:option('-out', '', 'specify full output file path  (if none put in local output/ folder)')
cmd:option('-outformat', 'mat', 'specify format of output (mat, t7)')

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
if opt.ckext ~= '' then
  checkpoint_proj_path = '/data/jjliu/checkpoints/' .. opt.ckp .. '_p' .. opt.ckgen .. '_' .. opt.ckext
end
print('Loading network..')
gen_path = paths.concat(checkpoint_gen_path, 'shapenet101_' .. opt.ckgen .. '_net_G.t7')
proj_path = paths.concat(checkpoint_proj_path, 'shapenet101_' .. opt.ckgen .. '_' .. opt.ckproj .. '_net_P.t7')
netG = torch.load(gen_path)
netP = torch.load(proj_path)
print(netG)
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
if opt.informat ~= 'mat' and opt.informat ~= 't7' then
  opt.informat = 'mat'
end
if opt.informat == 'mat' then
  tmpinput = mat.load(paths.concat(data_dir, opt.input .. '.mat'), 'off_volume')
else
  tmpinput = torch.load(paths.concat(data_dir, opt.input .. '.t7'), 'ascii')
end
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
print('Output dimensions: ')
print(output:size())
if paths.dir('./output/') == nil then
  paths.mkdir('output')
end
cur_time = os.time()
cur_times = '' .. cur_time
fname = 'proj_' ..cur_times .. '_' .. opt.ckgen .. '_' .. opt.ckproj
local fullfname
if opt.out == '' or opt.out == nil then
  fullfname = paths.concat('./output', fname)
else
  fullfname = opt.out
end
if opt.gpu > 0 then
  latent = latent:double()
  output = output:double()
end

if opt.outformat ~= 'mat' and opt.outformat ~= 't7' then
  opt.outformat = 'mat'
end

mat.save(fullfname .. '.mat', {['inputs'] = latent, ['voxels'] = output}) 
if opt.outformat == "t7" then
  torch.save(fullfname .. '.t7', output, 'ascii')
end

--if opt.outformat == 'mat' then
  --mat.save(fullfname .. '.mat', {['inputs'] = latent, ['voxels'] = output}) 
--else
  --torch.save(fullfname .. '.t7', output, 'ascii')
--end
print('saving done')

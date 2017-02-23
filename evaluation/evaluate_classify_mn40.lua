--- Evaluate the classifier on the test data and report accuracy / other metrics
require 'nn'
require 'torch'
require 'xlua'
require 'KLDPenalty'
require 'Sampler'
require 'paths'
assert(pcall(function () mat = require('fb.mattorch') end) or pcall(function() mat = require('matio') end), 'no mat IO interface available')

cmd = torch.CmdLine()
cmd:option('-gpu', 0, 'GPU id, starting from 1. Set it to 0 to run it in CPU mode. ')
cmd:option('-ck', 25, 'Checkpoint of classifier')
cmd:option('-ckp','checkpoints_64class100', 'name of checkpoint folder for classifer')
cmd:option('-data', 'ModelNet40_fix', 'name of ModelNet40 folder')
cmd:option('-dim',64, 'dimensions of input (default 64)')
cmd:option('-nc',101, 'number of categories in original model')
cmd:option('-cachedir', '/data/jjliu/cache', 'path of cache directory (for loading mappings from cat to number)')
cmd:option('-catarr', 'all_models_catarr_data', 'name of catarr file from cache directory (for loading above)')

opt = cmd:parse(arg or {})
if opt.gpu > 0 then
  require 'cunn'
  require 'cudnn'
  require 'cutorch'
  cutorch.setDevice(opt.gpu)
  nn.DataParallelTable.deserializeNGPUs = 1
end

checkpoint_path = paths.concat('/data/jjliu/checkpoints', opt.ckp)
data_dir = paths.concat('/data/jjliu/models', opt.data)
catarr_dir = paths.concat(opt.cachedir, opt.catarr .. '.t7')
testdata_f = paths.concat(opt.cachedir, 'all_models_testmodelnet.t7')

print('Loading network..')
net_path = paths.concat(checkpoint_path, 'shapenet101_' .. opt.ck .. '_net_C.t7')
netC = torch.load(net_path)
print(netC)
if opt.gpu == 0 then
  netC = netC:double()
end
netC:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)     -- convolution bias is removed during training
netC:evaluate() -- batch normalization behaves differently during evaluation

-- load catmap
catarr = torch.load(catarr_dir)
catmap = {}
for i = 1, #catarr do
  catmap[catarr[i]] = i
end
-- load models and label tensor
num_test = 0
for cat in paths.iterdirs(data_dir) do
  print(cat)
  cat_dir = paths.concat(paths.concat(data_dir, cat), 'test')
  if catmap[cat] ~= nil then
    for f in paths.iterfiles(cat_dir) do
      num_test = num_test + 1
    end
  end
end
all_test_models = torch.ByteTensor(num_test, 1, opt.dim, opt.dim, opt.dim)
truthTensor = torch.zeros(num_test)
cached = false
if paths.filep(testdata_f) then
  all_test_models = torch.load(testdata_f)
  cached = true
end
cur_index = 1
for cat in paths.iterdirs(data_dir) do
  cat_dir = paths.concat(paths.concat(data_dir, cat), 'test')
  catindex = catmap[cat]
  if catmap[cat] ~= nil then
    for f in paths.iterfiles(cat_dir) do
      print(f)
      if not cached then
        full_file = paths.concat(cat_dir, f)
        off_tensor = mat.load(full_file, 'off_volume')
        all_test_models[cur_index] = off_tensor
      end
      print(cur_index)
      print(cat)
      print(catmap[cat])
      truthTensor[cur_index] = catindex
      cur_index = cur_index + 1
    end
  end
end
if not cached then
  torch.save(testdata_f, all_test_models)
end

print(num_test)
-- run through all test data
bs = 10
input = torch.zeros(bs, 1, opt.dim, opt.dim, opt.dim)
results = torch.zeros(num_test, opt.nc)
if opt.gpu > 0 then
  netC = netC:cuda()
  netC = cudnn.convert(netC, cudnn)
  input = input:cuda()
end
for i = 1, math.ceil(num_test / bs) do
  print(('processing %d/%d'):format(i, math.ceil(num_test/bs)))
  ind_low = (i-1)*bs + 1
  ind_high = math.min(num_test, i * bs)
  input:zero()
  input[{{1,ind_high-ind_low+1},{},{},{},{}}] = all_test_models[{{ind_low,ind_high},{},{},{},{}}]
  res = netC:forward(input):double()
  results[{{ind_low,ind_high},{}}] = res[{{1,ind_high-ind_low+1},{}}]
end

-- perform zero-one loss on results
max, maxindices = torch.max(results, 2)
accuracy = 0
--print(maxindices:size())
for i = 1, maxindices:size(1) do
  --print(results[{i}])
  --print(truthTensor[i])
  --print(maxindices[{i, 1}] .. ' ' .. truthTensor[i])
  if (maxindices[{i, 1}] == truthTensor[i]) then
    accuracy = accuracy + 1
  end
end
accuracy = accuracy/(maxindices:size(1))
print('ACCURACY: ' .. accuracy)

--- Evaluate the classifier on the test data and report accuracy / other metrics
require 'nn'
require 'torch'
require 'xlua'
require 'KLDPenalty'
require 'Sampler'
require 'paths'
require 'util'
assert(pcall(function () mat = require('fb.mattorch') end) or pcall(function() mat = require('matio') end), 'no mat IO interface available')

cmd = torch.CmdLine()
cmd:option('-gpu', 0, 'GPU id, starting from 1. Set it to 0 to run it in CPU mode. ')
cmd:option('-ck', 25, 'Checkpoint of classifier')
cmd:option('-ckp','checkpoints_64class100', 'name of checkpoint folder for classifer')
cmd:option('-data', 'full_dataset_voxels_64_split2', 'name of data models folder (needs a test directory inside)')
cmd:option('-origdata', '', 'name of original data models folder (no splits). Only used if rotated > 0')
cmd:option('-dim',64, 'dimensions of input (default 64)')
cmd:option('-nc',101, 'number of categories to classify')
cmd:option('-cachedir', '/data/jjliu/cache', 'path of cache directory (for loading mappings from cat to number)')
cmd:option('-catarr', 'all_models_catarr_data', 'name of catarr file from cache directory (for loading above)')
cmd:option('-catpr', 'chair', 'measure precision recall of a category')
cmd:option('-rotated',0, 'whether to evaluate each test obj on an ensemble of rotated views or not')
-- TODO: incomplete
cmd:option('-avg', false, 'whether to average predictions or use a voting approach (not necessary if opt.rotated=0')

opt = cmd:parse(arg or {})
if opt.gpu > 0 then
  require 'cunn'
  require 'cudnn'
  require 'cutorch'
  cutorch.setDevice(opt.gpu)
  nn.DataParallelTable.deserializeNGPUs = 1
end

checkpoint_path = paths.concat('/data/jjliu/checkpoints', opt.ckp)
test_dir = paths.concat(paths.concat('/data/jjliu/models', opt.data), 'test')
labels_f = paths.concat(paths.concat('/data/jjliu/models',opt.data), 'testlabels.txt')
orig_data_dir = paths.concat('/data/jjliu/models',opt.origdata)
catarr_dir = paths.concat(opt.cachedir, opt.catarr .. '.t7')
cache_dir = '/data/jjliu/cache'
testdata_f = paths.concat(cache_dir, 'all_models_testmodels.t7')
testdata_rotated_f = paths.concat(cache_dir, 'all_models_testmodels_rotated.t7')


print('Loading network..')
net_path = paths.concat(checkpoint_path, 'shapenet101_' .. opt.ck .. '_net_C.t7')
netC = torch.load(net_path)

-- TODO: remove if not parallel
netC = netC:get(1)

print(netC)
--if opt.gpu == 0 then
  netC = netC:double()
--end
netC:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)     -- convolution bias is removed during training
netC:evaluate() -- batch normalization behaves differently during evaluation

-- load labels
catarr = torch.load(catarr_dir)
catmap = {}
for i = 1, #catarr do
  catmap[catarr[i]] = i
end
truthlabels = {}
origfilemap = {}
num_test = 0
for line in io.lines(labels_f) do
  local tokens = line:split(',')
  local fname = tokens[1]
  local cat = tokens[2]
  local origFile = tokens[3]
  truthlabels[fname] = catmap[cat]
  origfilemap[fname] = origFile
  num_test = num_test + 1
end

--TODO: remove
for k, v in pairs(catmap) do
  if v == 32 then
    print(k)
    print(v)
    print(' ')
  end
end

-- load all test models
all_test_models = torch.ByteTensor(num_test, 1, opt.dim, opt.dim, opt.dim)
truthTensor = torch.zeros(num_test)
cached = false
if opt.rotated > 0 then
  if paths.filep(testdata_rotated_f) then
    cached=true
    all_test_models = torch.load(testdata_rotated_f)
  end
else
  if paths.filep(testdata_f) then
    cached=true
    all_test_models = torch.load(testdata_f)
  end
end

if not cached and opt.rotated > 0 then
  all_test_models = torch.ByteTensor(num_test * opt.rotated, 1, opt.dim, opt.dim, opt.dim)
end
cur_index = 1
for test_file in paths.iterfiles(test_dir) do 
  print(('loading %d, %s, %s'):format(cur_index, test_file, catarr[truthlabels[test_file]]))
  if not cached then
    if opt.rotated > 0 then
      local origFile = origfilemap[test_file]
      local origCat = catarr[truthlabels[test_file]]
      local cat_dir = paths.concat(orig_data_dir, origCat)
      for i = 1, opt.rotated do
        local orig_file_parts = origFile:split('.')
        local orig_file_name = orig_file_parts[1]
        local orig_file_rot = orig_file_name .. '_' .. opt.rotated .. '.mat'
        local full_orig_file_rot = paths.concat(cat_dir, orig_file_rot)
        local off_rot_tensor = mat.load(full_orig_file_rot, 'off_volume')
        assert(off_rot_tensor ~= nil)
        all_test_models[(cur_index - 1)*opt.rotated + i] = off_rot_tensor
      end
    else
      full_test_file = paths.concat(test_dir, test_file)
      off_tensor = mat.load(full_test_file, 'off_volume')
      all_test_models[cur_index] = off_tensor
    end
  end
  truthTensor[cur_index] = truthlabels[test_file]
  cur_index = cur_index + 1
end
if not cached then
  if opt.rotated > 0 then
    torch.save(testdata_rotated_f, all_test_models)
  else
    torch.save(testdata_f, all_test_models)
  end
end

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
  if opt.rotated > 0 then
    for j = ind_low, ind_high do
      input[{{1,opt.rotated}}] = all_test_models[{{(j-1)*opt.rotated+1,j*opt.rotated}}]
      res = netC:forward(input[{{1,opt.rotated}}])
      mean_res = torch.mean(res, 1)
      results[{j}]:copy(mean_res)
    end
  else
    input[{{1,ind_high-ind_low+1},{},{},{},{}}] = all_test_models[{{ind_low,ind_high},{},{},{},{}}]
    res = netC:forward(input):double()
    results[{{ind_low,ind_high},{}}] = res[{{1,ind_high-ind_low+1},{}}]
  end
end

-- perform zero-one loss on results
max, maxindices = torch.max(results, 2)
accuracy = 0
num_pred = 0
num_pred_corr = 0
num_corr = 0
--print(maxindices:size())
for i = 1, maxindices:size(1) do
  --print(results[{i}])
  --print(truthTensor[i])
  --print(maxindices[{i, 1}] .. ' ' .. truthTensor[i])
  if (maxindices[{i, 1}] == truthTensor[i]) then
    accuracy = accuracy + 1
  else
    --print('ACTUAL CATEGORY: ' .. catarr[truthTensor[i]])
    --print('PREDICTED CATEGORY: ' .. catarr[maxindices[{i,1}]])
    -- calculate precision/recall for chairs
  end

  if catarr[truthTensor[i]] == opt.catpr then
    num_corr = num_corr + 1
  end
  if catarr[maxindices[{i,1}]] == opt.catpr then
    num_pred = num_pred + 1
  end
  if catarr[truthTensor[i]] == opt.catpr and catarr[truthTensor[i]] == catarr[maxindices[{i,1}]] then
    num_pred_corr = num_pred_corr + 1
  end
end
accuracy = accuracy/(maxindices:size(1))
print('ACCURACY: ' .. accuracy)
print('CHAIR precision: ' .. (num_pred_corr/num_pred))
print('CHAIR recall: ' .. (num_pred_corr/num_corr))

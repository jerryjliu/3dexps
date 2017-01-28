require 'paths'
threads = require 'threads'
threads.Threads.serialization('threads.sharedserialize')
local matio = require 'matio'

--data_name = '32_chair'
--data_path = '/data/jjliu/models/full_dataset_voxels_' .. data_name

--cache_path = '/data/jjliu/cache/'

--all_models_tensor_p = cache_path .. 'all_models_tensor_' .. data_name .. '.t7'
--all_models_catarr_p = cache_path .. 'all_models_catarr_' .. data_name .. '.t7'
--all_models_catdict_p = cache_path .. 'all_models_catdict' .. data_name .. '.t7'

data = {}
data.__index = data

function data.new(opt)
  local self = {}
  setmetatable(self, data)
  self.opt = opt

  -- initialize variables
  self.data_path = self.opt.data_dir .. self.opt.data_name
  self.all_models_tensor_p = self.opt.cache_dir .. 'all_models_tensor_' .. self.opt.data_name .. '.t7'
  self.all_models_catarr_p = self.opt.cache_dir .. 'all_models_catarr_' .. self.opt.data_name .. '.t7'
  self.all_models_catdict_p = self.opt.cache_dir .. 'all_models_catdict_' .. self.opt.data_name .. '.t7'
  self.all_models_catdictr_p = self.opt.cache_dir .. 'all_models_catdictr_' .. self.opt.data_name .. '.t7'

  print(self.data_path)

  local catarr = {}
  local catdict = {}
  local catdictr = {}
  cats = paths.dir(self.data_path)
  self._size = 0
  for i, v in ipairs(cats) do
    if v ~= '.' and v ~= '..' then
      cat = cats[i]
      cat_path = paths.concat(self.data_path, cat)
      cat_models = paths.dir(cat_path)
      for j, v2 in ipairs(cat_models) do
        if v2 ~= '.' and v2 ~= '..' then
          self._size = self._size + 1
        end
      end
    end
  end
  print('TOTAL SIZE ' .. self._size)
  -- NOTE: this is going to take up > 20GB RAM. Will not work on most machines. 
  -- if your RAM isn't sufficient you will have to read directly from disk
  if paths.filep(self.all_models_tensor_p) then
    all_models = torch.load(self.all_models_tensor_p)
    catarr = torch.load(self.all_models_catarr_p)
    catdict = torch.load(self.all_models_catdict_p) 
    catdictr = torch.load(self.all_models_catdictr_p)
  else
    all_models = torch.ByteTensor(self._size, self.opt.nout, self.opt.nout, self.opt.nout)
    local curindex = 1
    for i, v in ipairs(cats) do
      if v ~= '.' and v ~= '..' then
        cat = cats[i]
        print('CATEGORY: ' .. cat)
        cur_dict = {}
        cat_path = paths.concat(self.data_path, cat)
        cat_models = paths.dir(cat_path)
        table.sort(cat_models)
        for j, v2 in ipairs(cat_models) do
          if v2 ~= '.' and v2 ~= '..' then
            cat_model = cat_models[j]
            cat_model_path = paths.concat(cat_path, cat_model)
            print(i .. ', ' .. j .. ' loading ' .. cat_model_path)
            off_tensor = matio.load(cat_model_path, 'off_volume')
            all_models[{curindex}] = off_tensor
            cur_dict[#cur_dict + 1] = curindex
            catdictr[curindex] = #catarr + 1
            print(cat_model)
            print(#cur_dict)
            curindex = curindex + 1
          end
        end
        catdict[cat] = cur_dict
        catarr[#catarr + 1] = cat
      end
    end
    --local catdict = {}
    --cats = paths.dir(data_path)
    --self._size = 0
    --for i, v in ipairs(cats) do
      --if v ~= '.' and v ~= '..' then
        --cat = cats[i]
        --print('CATEGORY: ' .. cat)
        --cur_dict = {}
        --cat_path = paths.concat(data_path, cat)
        --cat_models = paths.dir(cat_path)
        --for j, v2 in ipairs(cat_models) do
          --if v2 ~= '.' and v2 ~= '..' then
            --cat_model = cat_models[j]
            --cat_model_path = paths.concat(cat_path, cat_model)
            --print(i .. ', ' .. j .. ' loading ' .. cat_model_path)
            ----cur_dict[#cur_dict + 1] = cat_model_path
            --off_tensor = matio.load(cat_model_path, 'off_volume')
            --cur_dict[#cur_dict + 1] = off_tensor
            --self._size = self._size + 1
          --end
        --end
        --catdict[cat] = cur_dict
        --catarr[#catarr + 1] = cat
      --end
    --end
    torch.save(self.all_models_tensor_p, all_models)
    torch.save(self.all_models_catarr_p, catarr)
    torch.save(self.all_models_catdict_p, catdict)
    torch.save(self.all_models_catdictr_p, catdictr)
  end
  self.catarr = catarr
  self.catdict = catdict
  self.catdictr = catdictr
  self.all_models = all_models
  self.dindices = torch.LongTensor(self.all_models:size(1))
  self.dcur = 1
  self:resetAndShuffle()
  --print(self.catdict)
  return self
end

-- run knuth shuffle of indices (from http://rosettacode.org/wiki/Knuth_shuffle#Lua)
function shuffle_indices(t)
  for n = t:size(1), 1, -1 do
      local k = math.random(n)
      t[n], t[k] = t[k], t[n]
  end
       
  return t
end

function data:resetAndShuffle()
  num_models = self.all_models:size(1)
  assert(num_models == self._size)
  for i = 1,num_models do
    self.dindices[i] = i
  end
  shuffle_indices(self.dindices)
  self.dcur = 1
end

--function data:getBatch(quantity)
  ----local data = torch.Tensor(quantity, 1, self.opt.nout, self.opt.nout, self.opt.nout)
  ----data:cuda()
  ----local data = torch.Tensor(quantity, 1, 50,50,50)
  --local label = torch.Tensor(quantity)
  --visited = {}
  --i = 1
  --off_tindices = torch.LongTensor(quantity):zero()
  --while i <= quantity do
    --local catindex = torch.random(1, #self.catarr)
    --local cat = self.catarr[catindex]
    --local offindex = torch.random(1, #self.catdict[cat])
    --if visited[cat .. offindex] == nil then
      --visited[cat .. offindex] = 1
      --off_tindex = self.catdict[cat][offindex]
      --off_tindices[{i}] = off_tindex
      --label[i] = catindex
      ----label[i] = offindex
      --i = i + 1
    --end
  --end
  ----off_tindices = torch.sort(off_tindices)
  ----print(off_tindices)
  --local data = self.all_models:index(1,off_tindices)
  --if self.opt.gpu > 0 then
    --data:cuda()
  --end
  --collectgarbage()
  --collectgarbage()
  --return data, label
--end

function data:getBatch(quantity)
  local label = torch.Tensor(quantity)
  local minindex = self.dcur
  local maxindex = math.min(self.all_models:size(1), minindex + quantity - 1)

  local data = self.all_models:index(1, self.dindices[{{minindex,maxindex}}])
  --print(self.dindices[{{minindex,maxindex}}])
  --print(self.dindices[{{minindex,maxindex}}])
  for j = minindex, maxindex do 
    --print(j)
    label[{j - minindex + 1}] = self.catdictr[self.dindices[j]]
  end
  if self.opt.gpu > 0 then
    data:cuda()
  end
  collectgarbage()
  collectgarbage()
  self.dcur = self.dcur + (maxindex - minindex + 1)
  return data, label[{{1,maxindex-minindex+1}}]
end

function data:size()
  return self._size
end

function data:numClasses()
  return #self.catarr
end

--testopt={nout=64}
--test = data.new(testopt)
--test:getBatch(60)

return data


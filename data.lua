require 'paths'
threads = require 'threads'
threads.Threads.serialization('threads.sharedserialize')
local matio = require 'matio'
data_path = '/data/models/full_dataset_voxels_64/'

data = {}
data.__index = data

function data.new(opt)
  local self = {}
  setmetatable(self, data)
  self.opt = opt
  local catarr = {}
  local catdict = {}
  cats = paths.dir(data_path)
  self._size = 0
  for i, v in ipairs(cats) do
    if v ~= '.' and v ~= '..' then
      cat = cats[i]
      cat_path = paths.concat(data_path, cat)
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
  all_models = torch.ByteTensor(self._size, self.opt.nout, self.opt.nout, self.opt.nout)
  local curindex = 1
  for i, v in ipairs(cats) do
    if v ~= '.' and v ~= '..' then
      cat = cats[i]
      print('CATEGORY: ' .. cat)
      cur_dict = {}
      cat_path = paths.concat(data_path, cat)
      cat_models = paths.dir(cat_path)
      for j, v2 in ipairs(cat_models) do
        if v2 ~= '.' and v2 ~= '..' then
          cat_model = cat_models[j]
          cat_model_path = paths.concat(cat_path, cat_model)
          print(i .. ', ' .. j .. ' loading ' .. cat_model_path)
          off_tensor = matio.load(cat_model_path, 'off_volume')
          all_models[{curindex}] = off_tensor
          cur_dict[#cur_dict + 1] = curindex
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
  self.catarr = catarr
  self.catdict = catdict
  self.all_models = all_models
  --print(self.catdict)
  return self
end

function data:getBatch(quantity)
  --local data = torch.Tensor(quantity, 1, self.opt.nout, self.opt.nout, self.opt.nout)
  --data:cuda()
  --local data = torch.Tensor(quantity, 1, 50,50,50)
  local label = torch.Tensor(quantity)
  visited = {}
  i = 1
  off_tindices = torch.LongTensor(quantity)
  while i <= quantity do
    local catindex = torch.random(1, #self.catarr)
    local cat = self.catarr[catindex]
    local offindex = torch.random(1, #self.catdict[cat])
    if type(visited[cat .. offindex]) ~= nil then
      visited[cat .. offindex] = 1
      off_tindex = self.catdict[cat][offindex]
      off_tindices[{i}] = off_tindex
      label[i] = catindex
      i = i + 1
    end
  end
  local data = self.all_models:index(1,off_tindices)
  data:cuda()

  --while i <= quantity do
    --local catindex = torch.random(1, #self.catarr)
    --local cat = self.catarr[catindex]
    --local offindex = torch.random(1, #self.catdict[cat])
    --if type(visited[cat .. offindex]) ~= nil then
      --visited[cat .. offindex] = 1
      ----print(cat .. ': ' .. offindex .. ' ' .. self.catdict[cat][offindex])
      ----off_file = self.catdict[cat][offindex]
      ----off_tensor = matio.load(off_file, 'off_volume')
      --off_tensor = self.catdict[cat][offindex]
      ----data[{i,1}]:copy(off_tensor)
      --data[{i,1}] = off_tensor
      --label[i] = catindex
      --i = i + 1
    --end
  --end
  collectgarbage()
  collectgarbage()
  return data, label
end

function data:size()
  return self._size
end

--testopt={nout=64}
--test = data.new(testopt)
--test:getBatch(60)

return data


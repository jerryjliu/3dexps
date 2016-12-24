require 'paths'
threads = require 'threads'
threads.Threads.serialization('threads.sharedserialize')
local matio = require 'matio'
data_path = '/data/models/full_dataset_voxels_64_air/'

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
      print('CATEGORY: ' .. cat)
      cur_dict = {}
      cat_path = paths.concat(data_path, cat)
      cat_models = paths.dir(cat_path)
      for j, v2 in ipairs(cat_models) do
        if v2 ~= '.' and v2 ~= '..' then
          cat_model = cat_models[j]
          cat_model_path = paths.concat(cat_path, cat_model)
          print(i .. ', ' .. j .. ' loading ' .. cat_model_path)
          --cur_dict[#cur_dict + 1] = cat_model_path
          off_tensor = matio.load(cat_model_path, 'off_volume')
          cur_dict[#cur_dict + 1] = off_tensor
          self._size = self._size + 1
        end
      end
      catdict[cat] = cur_dict
      catarr[#catarr + 1] = cat
    end
  end
  self.catarr = catarr
  self.catdict = catdict
  --print(self.catdict)
  return self
end

function data:getBatch(quantity)
  local data = torch.Tensor(quantity, 1, self.opt.nout, self.opt.nout, self.opt.nout)
  data:cuda()
  --local data = torch.Tensor(quantity, 1, 50,50,50)
  local label = torch.Tensor(quantity)
  visited = {}
  i = 1

  --self.threads = threads.Threads(
    --quantity)
  --for i = 1, quantity do
    --self.threads:addjob(
      --function()
        --print('inside thread ' .. i)
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
          ----i = i + 1
        --end
        --collectgarbage(); collectgarbage();
      --end
    --)
  --end
  --print('done adding jobs')
  --self.threads.synchronize()
  --print('done synchronizing')
  --self.threads.terminate()
  --self.threads = threads.Threads(
    --quantity,
    --function(i)
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
        ----i = i + 1
      --end
    --end
  --)
  --self.threads:synchronize()
  while i <= quantity do
    local catindex = torch.random(1, #self.catarr)
    local cat = self.catarr[catindex]
    local offindex = torch.random(1, #self.catdict[cat])
    if type(visited[cat .. offindex]) ~= nil then
      visited[cat .. offindex] = 1
      --print(cat .. ': ' .. offindex .. ' ' .. self.catdict[cat][offindex])
      --off_file = self.catdict[cat][offindex]
      --off_tensor = matio.load(off_file, 'off_volume')
      off_tensor = self.catdict[cat][offindex]
      --data[{i,1}]:copy(off_tensor)
      data[{i,1}] = off_tensor
      label[i] = catindex
      i = i + 1
    end
  end
  collectgarbage()
  collectgarbage()
  return data, label
end

function data:size()
  return self._size
end

--testopt={nout=64}
--test = data.new(testopt)
--test:getBatch(10)

return data


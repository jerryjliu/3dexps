data_path = '/data/models/full_dataset_voxels_64/'

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
    print(i .. ' CATEGORY: ' .. cat)
    cur_dict = {}
    cat_path = paths.concat(data_path, cat)
    cat_models = paths.dir(cat_path)
    --for j, v2 in ipairs(cat_models) do
      --if v2 ~= '.' and v2 ~= '..' then
        --cat_model = cat_models[j]
        --cat_model_path = paths.concat(cat_path, cat_model)
        --print('loading ' .. cat_model_path)
        ----cur_dict[#cur_dict + 1] = cat_model_path
        ----off_tensor = matio.load(cat_model_path, 'off_volume')
        --cur_dict[#cur_dict + 1] = off_tensor
        --self._size = self._size + 1
      --end
    --end
    catdict[cat] = cur_dict
    catarr[#catarr + 1] = cat
  end
end
self.catarr = catarr
self.catdict = catdict

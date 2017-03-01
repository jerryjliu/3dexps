
-- compute 0-1 loss given some n x m dimension output, where m is dimension of output space per input, 
-- and a n-dimension truth vector containing the correct label
function compute_accuracy(outputs, truth)
  local accuracy = 0
  local max, maxindices = torch.max(outputs, 2)
  for i = 1, truth:size(1) do
    if (maxindices[{i, 1}] == truth[i]) then
      accuracy = accuracy + 1
    end
  end
  accuracy = accuracy/(maxindices:size(1))
  return accuracy
end

function compute_class_weighted_accuracy(outputs, truth, nc)
  local catTruth = torch.zeros(nc)
  local catCorrect = torch.zeros(nc)
  local max, maxindices = torch.max(outputs, 2)
  for i = 1, truth:size(1) do
    truthVal = truth[i]
    catTruth[truthVal] = catTruth[truthVal] + 1
    if (maxindices[{i,1}] == truthVal) then
      catCorrect[truthVal] = catCorrect[truthVal] + 1
    end
  end
  local accuracy = 0
  local numCount = 0
  for i = 1, nc do
    if catTruth[i] ~= 0 then
      accuracy = accuracy + (catCorrect[i]/catTruth[i])
      numCount = numCount + 1
    end
  end
  accuracy = accuracy/numCount
  return accuracy
end

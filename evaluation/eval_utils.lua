
-- compute 0-1 loss given some n x m dimension output, where m is dimension of output space per input, 
-- and a n-dimension truth vector containing the correct label
function compute_accuracy(outputs, truth)
  accuracy = 0
  max, maxindices = torch.max(outputs, 2)
  for i = 1, truth:size(1) do
    if (maxindices[{i, 1}] == truth[i]) then
      accuracy = accuracy + 1
    end
  end
  accuracy = accuracy/(maxindices:size(1))
  return accuracy
end

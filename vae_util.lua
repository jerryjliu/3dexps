-- Utils for the VAE components

function get_mean_logvar(input)
  if input:dim() == 5 then
    ndim = input:size(2)
    mid = ndim/2 -- only makes sense if ndim is even
    mean = input[{{},{1, mid}}]
    log_var = input[{{},{mid+1,ndim}}]
  elseif input:dim() == 4 then
    ndim = input:size(1)
    mid = ndim/2 -- only makes sense if ndim is even
    mean = input[{{1, mid}}]
    log_var = input[{{mid+1,ndim}}]
  end

  return mean, log_var, mid
end

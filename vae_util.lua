-- Utils for the VAE components

function get_mean_logvar(input)
  ndim = input:size(1)
  mid = ndim/2 -- only makes sense if ndim is even
  mean = input[{{1, mid}}]
  log_var = input[{{mid+1,ndim}}]

  return mean, log_var
end

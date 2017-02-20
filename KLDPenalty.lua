require 'vae_util'

local KLDPenalty, parent = torch.class('nn.KLDPenalty', 'nn.Module')

alpha_KLD = 1

function KLDPenalty:updateOutput(input)
    --local mean, log_var = table.unpack(input)
    mean, log_var, mid = get_mean_logvar(input)
    self.output = input

    -- Appendix B from VAE paper: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    local mean_sq = torch.pow(mean, 2)
    local KLDelements = log_var:clone()
    KLDelements:exp():mul(-1)
    KLDelements:add(-1, torch.pow(mean, 2))
    KLDelements:add(1)
    KLDelements:add(log_var)
    self.loss = -0.5 * torch.sum(KLDelements)

    return self.output
end

function KLDPenalty:updateGradInput(input, gradOutput)
    --assert(#gradOutput == 2)
    --local mean, log_var = table.unpack(input)
    mean, log_var, mid = get_mean_logvar(input)

    --self.gradInput = {}
    self.gradInput = input:clone()
    self.gradInput:resizeAs(input)
    if self.gradInput:dim() == 4 then
      self.gradInput[{{1, mid}}] = (alpha_KLD * mean:clone()) + gradOutput[{{1, mid}}]
      self.gradInput[{{mid+1, self.gradInput:size(1)}}] = (alpha_KLD * torch.exp(log_var):mul(-1):add(1):mul(-0.5)) + gradOutput[{{mid+1, self.gradInput:size(1)}}]
    elseif self.gradInput:dim() == 5 then
      self.gradInput[{{},{1, mid}}] = (alpha_KLD * mean:clone()) + gradOutput[{{},{1,mid}}]
      self.gradInput[{{},{mid+1, self.gradInput:size(2)}}] = (alpha_KLD * torch.exp(log_var):mul(-1):add(1):mul(-0.5)) + gradOutput[{{}, {mid+1, self.gradInput:size(2)}}]
    end
    return self.gradInput
end

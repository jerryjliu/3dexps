-- Based on JoinTable module

require 'nn'
require 'vae_util'

local Sampler, parent = torch.class('nn.Sampler', 'nn.Module')

function Sampler:__init()
    parent.__init(self)
    self.gradInput = {}
end 

function Sampler:updateOutput(input)
    mean, log_var = get_mean_logvar(input)
    self.eps = self.eps or mean.new()
    self.eps:resizeAs(mean):copy(torch.randn(mean:size()))

    self.output = self.output or self.output.new()
    self.output:resizeAs(log_var):copy(log_var)
    self.output:mul(0.5):exp():cmul(self.eps)

    self.output:add(mean)

    return self.output
end

function Sampler:updateGradInput(input, gradOutput)
    mean, log_var = get_mean_logvar(input)
    self.gradInput[1] = self.gradInput[1] or mean.new()
    self.gradInput[1]:resizeAs(gradOutput):copy(gradOutput)
    
    self.gradInput[2] = self.gradInput[2] or log_var.new()
    self.gradInput[2]:resizeAs(gradOutput):copy(log_var)
    
    self.gradInput[2]:mul(0.5):exp():mul(0.5):cmul(self.eps)
    self.gradInput[2]:cmul(gradOutput)

    return self.gradInput
end

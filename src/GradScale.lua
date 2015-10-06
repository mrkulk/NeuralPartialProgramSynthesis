local GradScale, parent = torch.class('nn.GradScale', 'nn.Module')

function GradScale:__init(scale)
   self.scale = scale
   parent.__init(self)
end
 
function GradScale:updateOutput(input)
   self.output = input
   return self.output
end

function GradScale:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput*self.scale
   return self.gradInput
end

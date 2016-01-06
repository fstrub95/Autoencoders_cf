local SDAECriterionGPU, parent = torch.class('nnsparse.SDAECriterionGPU', 'nn.Criterion')

function SDAECriterionGPU:__init(criterion, inputSize, SDAEconf)
   parent.__init(self)

   self.criterion = criterion
   
   self.alpha = SDAEconf.alpha or 1   
   self.beta  = SDAEconf.beta  or 0

   self.inputDim = inputSize

   self.hideRatio = SDAEconf.hideRatio or 0

   self.criterion.sizeAverage = false
   self.sizeAverage = true
end


function SDAECriterionGPU:prepareInput(inputs)

   assert(torch.type(inputs) == "table")
   
   self.output     = self.inputs or inputs[1].new()
   self.mask       = self.inputs or inputs[1].new()
   
   self.output:resize(#inputs, self.inputDim):zero()
   self.mask:resize(#inputs, self.inputDim):zero()

   self.bufRand   = self.bufRand  or inputs[1].new()
   
   for k, oneInput in pairs(inputs) do

      self.bufRand:resize(oneInput:size(1)):uniform()
     
      local alphaMask = self.bufRand:lt(self.hideRatio)
      local betaMask  = alphaMask:eq(0)
      
      local index = oneInput[{{},1}]
      local data  = oneInput[{{},2}]

      local alphaIndex = index[alphaMask]
      local betaIndex  = index[betaMask]

      if betaIndex:nDimension() > 0 then
         self.output[k]:indexCopy(1, betaIndex,  data[betaMask])
         self.mask[k]:indexFill(1, betaIndex , self.beta)
      end

      if alphaIndex:nDimension() > 0 then
        self.mask[k]:indexFill(1, alphaIndex, self.alpha)
      end
      
   end   
   
   return self.output

end


function SDAECriterionGPU:updateOutput(estimate, target)
      --self.criterion:forward(estimate, target)
      return 0
end



function SDAECriterionGPU:updateGradInput(estimate, target)

   local dloss = self.criterion:updateGradInput(estimate , target)
   dloss:cmul(self.mask)

   if self.sizeAverage == true then
       dloss:div(estimate:nElement())
   end

   return dloss

end


function SDAECriterionGPU:__tostring__()
   return "Sparse " .. self.criterion:__tostring__()
end

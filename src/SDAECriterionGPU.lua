local SDAECriterionGPU, parent = torch.class('nnsparse.SDAECriterionGPU', 'nn.Criterion')

function SDAECriterionGPU:__init(criterion, inputSize, SDAEconf)
   parent.__init(self)

   self.criterion = criterion
   
   self.alpha = SDAEconf.alpha or 1   
   self.beta  = SDAEconf.beta  or 0

   self.inputDim = inputSize
   
   self.noiseRatio = SDAEconf.noiseRatio or 0
   self.noiseMean = SDAEconf.noiseMean or 0
   self.noiseStd = SDAEconf.noiseStd or 0.2
   
   self.hideRatio = SDAEconf.hideRatio or 0

   self.criterion.sizeAverage = false
   self.sizeAverage = true
   
   self.output = nil
end


function SDAECriterionGPU:prepareInput(inputs)


   if torch.type(inputs) == "table" then
   
      self.output = self.output or {}
      if #self.output ~=  #inputs then self.output = {} end

      self.mask       = self.inputs or inputs[1].new()   
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

         if torch.type(index) ~= "torch.CudaTensor" then
            alphaIndex = alphaIndex:long()
            betaIndex  = betaIndex:long()
         end

         --if there is no input : reverse alphaMask/betaMask
         if betaIndex:nDimension() == 0 then
            local swapBuf = alphaIndex
            alphaIndex = betaIndex
            betaIndex  = swapBuf

            swapBuf   = alphaMask
            alphaMask = betaMask
            betaMask  = swapBuf
         end

         self.output[k] = self.output[k] or oneInput.new()
         self.output[k]:resizeAs(oneInput):copy(oneInput)

         self.mask[k]:indexFill(1, betaIndex , self.beta)

         if alphaIndex:nDimension() > 0 then
            self.output[k][{{},2}][alphaMask] = 0 
            self.mask[k]:indexFill(1, alphaIndex, self.alpha)
         end

      end
   else
   
      self.shuffle = self.shuffle or torch.Tensor(inputs:size(2))
      
      if torch.type(inputs) ~= "torch.CudaTensor" then
         self.shuffle = self.shuffle:long()
      end

      self.mask   = self.inputs or inputs.new()   
      self.mask:resizeAs(inputs):fill(self.beta)

      self.output = self.output or inputs.new()   
      self.output:resizeAs(inputs)

      local noiseSize =  inputs:size(2) * (self.noiseRatio )
      local hideSize  =  inputs:size(2) * (self.noiseRatio + self.hideRatio )

      self.noise = self.noise or inputs.new()
      self.noise:resize(noiseSize)

      for i = 1, inputs:size(1) do
         self.shuffle:randperm(inputs:size(2))
       
         local shuflle = self.shuffle 
         if torch.type(inputs) == "torch.CudaTensor" then
            self.shuffleGPU = self.shuffleGPU or inputs.new()
            self.shuffleGPU:resize(self.shuffle:size()):copy(self.shuffle)
            shuffle = self.shuffleGPU
         end
 
         if noiseSize > 0 then
            self.output[i]:indexAdd(1 , shuffle[{{            1, noiseSize}}], self.noise:normal(self.noiseMean, self.noiseStd))
         end
         
         if hideSize > noiseSize then
            self.output[i]:indexFill(1, shuffle[{{noiseSize + 1, hideSize }}], 0)
         end
         
         self.mask[i]:indexFill(1, shuffle[{{1, hideSize}}], self.alpha)
      end         
               
   end

   return self.output

end


function SDAECriterionGPU:updateOutput(estimate, target)
      --self.criterion:forward(estimate, target)
      return 0
end



function SDAECriterionGPU:updateGradInput(estimate, target)

   --if sparse
   if torch.type(target) == "table" then
      self.densifier = self.densifier or nnsparse.Densify(self.inputDim)
      target = self.densifier:forward(target)
   end
   
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


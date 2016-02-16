cfn = cfn or {}
local SDAECriterionGPU, parent = torch.class('cfn.SDAECriterionGPU', 'nn.Criterion')

function SDAECriterionGPU:__init(criterion, SDAEconf, inputSize)
   parent.__init(self)

   self.criterion = criterion
   
   self.alpha = SDAEconf.alpha or 1   
   self.beta  = SDAEconf.beta  or 0

   self.inputDim = inputSize or 0
   
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

      --reset output if mini-bacth size is modified
      if #self.output ~=  #inputs then 
         self.output = {} 
      end

      self.mask       = self.mask or inputs[1].new()   
      self.mask:resize(#inputs, self.inputDim):zero()

      self.alphaMask = self.alphaMask or inputs[1].new()
      self.betaMask  = self.betaMask  or inputs[1].new()

      self.alphaIndex = self.alphaIndex or inputs[1].new()
      self.betaIndex  = self.betaIndex  or inputs[1].new()

      for k, oneInput in pairs(inputs) do

         local index = oneInput[{{},1}]

         --compte mask (lt et eq does not have inplace equivalent)
         self.alphaMask:resize(index:size()):bernoulli(self.hideRatio)
         self.betaMask:resize(index:size()):fill(1):add(-1,self.alphaMask)

        if torch.type(index) ~= "torch.CudaTensor" then
            index = index:long() 
            self.alphaMask  = self.alphaMask:byte()
            self.betaMask   = self.betaMask:byte()
            self.alphaIndex = self.alphaIndex:long()
            self.betaIndex  = self.betaIndex:long()
         end

         self.alphaIndex:maskedSelect(index, self.alphaMask)
         self.betaIndex:maskedSelect(index, self.betaMask)

         --if there is no input : reverse alphaMask/betaMask
         if self.betaIndex:nDimension() == 0 then
            local swapBuf   = self.alphaIndex
            self.alphaIndex = self.betaIndex
            self.betaIndex  = swapBuf

            swapBuf        = self.alphaMask
            self.alphaMask = self.betaMask
            self.betaMask  = swapBuf
         end

         self.output[k] = self.output[k] or oneInput.new()
         self.output[k]:resizeAs(oneInput):copy(oneInput)

         self.mask[k]:indexFill(1, self.betaIndex , self.beta)

         if self.alphaIndex:nDimension() > 0 then
             self.output[k][{{},2}][self.alphaMask] = 0
             self.mask[k]:indexFill(1, self.alphaIndex, self.alpha)
         end

      end   
   else
   
      self.shuffle = self.shuffle or inputs.new() 
      
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
      
         if torch.type(inputs) ~= "torch.CudaTensor" then
            self.shuffle:randperm(inputs:size(2))
         else
            self.shuffle:resize(inputs:size(2)):uniform(1,inputs:size(2)):floor()
         end
 
         if noiseSize > 0 then
            self.output[i]:indexAdd(1 , self.shuffle[{{            1, noiseSize}}], self.noise:normal(self.noiseMean, self.noiseStd))
         end
         
         if hideSize > noiseSize then
            self.output[i]:indexFill(1, self.shuffle[{{noiseSize + 1, hideSize }}], 0)
         end
         
         self.mask[i]:indexFill(1, self.shuffle[{{1, hideSize}}], self.alpha)
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
      self.densifier = self.densifier or nnsparse.Densify(estimate:size(2))
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


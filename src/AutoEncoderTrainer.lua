
AutoEncoderTrainer = {}


function AutoEncoderTrainer:new(network, conf, train, test, info, maxIndex)

   newObj = 
      {
         network  = network,
         loss     = conf.criterion,
         train    = train,
         test     = test,
         isSparse = network.isSparse or false,
         maxIndex  = maxIndex,
         rmse     = {},
         mae      = {},
      }

   -- this pre-compute the postprocessing offset
   if network.isSparse then
      local mean = torch.zeros(maxIndex,1)
      for k, oneInfo in pairs(info) do
         oneInfo.mean = oneInfo.mean or 0
         mean[k] = oneInfo.mean
   end
      newObj.mean = mean
   end

   self.__index = self

   return setmetatable(newObj, self)   

end



function AutoEncoderTrainer:Train(sgdOpt, epoch)

   -- pick alias
   local network = self.network
   local lossFct = self.loss  
   local inputs  = self.train

   -- Retrieve parameters and gradients
   local w, dw = network:getParameters()
   
   -- remove Torch size average, aka 1/n, for loss)
   lossFct.sizeAverage = false

   -- bufferize minibatch
   local input
   if network.isSparse then input = {}
   else                     input = torch.Tensor(sgdOpt.miniBatchSize, inputs[1]:size(1)) 
   end


   -- shuffle index for data
   local noSample = GetSize(inputs)
   local shuffle  = torch.randperm(noSample)

 
   -- Start training
   network:training()
   
   
   local cursor   = 1
   while cursor < noSample-1 do

      -- prepare minibatch)
      local noPicked = 1 
      while noPicked <= sgdOpt.miniBatchSize and cursor < noSample-1 do

         local shuffledIndex = shuffle[cursor]
         if inputs[shuffledIndex] then -- warning tables are not always continuous
            input[noPicked] = inputs[shuffledIndex]
            noPicked = noPicked + 1
         end

         cursor = cursor + 1
      end



      -- Define the closure to evalute w and dw
      local function feval(x)

         -- get new parameters
         if x ~= w then w:copy(x) end

         -- Reset gradients and losses
         network:zeroGradParameters()

         -- set the auto-encoder target
         local target = input

         -- prepare SDAE mask 
         input = lossFct:prepareInput(input)


          --- FORWARD
         local output = network:forward(input)
         local loss   = lossFct:forward(output, target)


         --- BACKWARD
         local dloss = lossFct:backward(output, target)
         local _     = network:backward(input, dloss)

         -- Return loss and gradients
         return loss/sgdOpt.miniBatchSize, dw:div(sgdOpt.miniBatchSize)
      end

      -- Optimize current iteration
      sgdOpt.evalCounter = epoch
      optim.sgd (feval, w, sgdOpt )

   end
   
   
end



function  AutoEncoderTrainer:Test()

   local network = self.network

   local train   = self.train
   local test    = self.test

   local loss, rmse, mae = NaN,NaN,NaN

  
   -- start evaluating
   network:evaluate()
   
   
   if self.isSparse then
      -- compute prediction error
      local rmseFct = nn.SparseCriterion(nn.MSECriterion())
      local maeFct  = nn.SparseCriterion(nn.AbsCriterion())
      
      -- compute the prediction
      local output = network:forward(train)  --WARNING indexing was lost while forwarding
      
      -- re-index the data
      local outputFull = torch.Tensor(self.maxIndex, output:size(2)):fill(NaN)
      local ouputIndex = 0 
      for realIndex, _ in pairs(train) do
          ouputIndex = ouputIndex + 1
          outputFull[realIndex] = output[ouputIndex]
      end
      
      outputFull:add(self.mean:expandAs(outputFull))   --recenter data
      outputFull[outputFull:eq(NaN)] = 0               --estimates with no training example are replaced with 0
      
      rmse = rmseFct:forward(outputFull, test)
      mae  = maeFct:forward(outputFull, test)
      
   else
      -- compute reconstruction loss 
      local lossFct = nn.MSECriterion() 
      local outputLoss = network:forward(test)
      loss = lossFct:forward(outputLoss, test)
   end

   return math.sqrt(loss), math.sqrt(rmse), mae

end




function AutoEncoderTrainer:Execute(sgdOpt)


   local noEpoch = sgdOpt.noEpoch

   for t = 1, noEpoch do

      xlua.progress(t, noEpoch)

      --train one epoch
      self:Train(sgdOpt, t)
      
      local newReconstructionRMSE, newPredictionRMSE, newMAE = self:Test()

      --resclase RMSE/MAE
      newPredictionRMSE     = newPredictionRMSE     * 2
      newMAE                = newMAE                * 2

      self.rmse[#self.rmse+1] =  newPredictionRMSE
      self.mae [#self.mae +1] =  newMAE


      if newPredictionRMSE ~= NaN and newMAE ~= NaN then
         print(t .. "/" .. noEpoch .. "\t RMSE : "  .. newPredictionRMSE .. "\t MAE : "  .. newMAE )
      end
      
      if newReconstructionRMSE ~= NaN  then
         print(t .. "/" .. noEpoch .. "\t RMSE : "  .. newReconstructionRMSE )
      end

   end
   
   return self.rmse, self.mae

end







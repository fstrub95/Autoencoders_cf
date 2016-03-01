local AutoEncoderTrainer = torch.class('AutoEncoderTrainer')

function AutoEncoderTrainer:__init(network, conf, train, test, info)


   self.network  = network
   self.loss     = conf.criterion
   self.train    = train
   self.test     = test
   self.info     = info
   self.isSparse = network.isSparse or false
   self.rmse     = {}
   self.mae      = {}

--   if network.isSparse then
--      newObj.tikhonov = nnsparse.Tikhonov(conf.weightDecay, newObj.network)
--      conf.weightDecay = 0
--      conf.weightDecays = newObj.tikhonov.lambdas
--   end

end



function AutoEncoderTrainer:Train(sgdOpt, epoch)

   -- pick alias
   local network = self.network
   local lossFct = self.loss
   local train  = self.train

   -- Retrieve parameters and gradients
   local w, dw = network:getParameters()

   -- remove Torch size average, aka 1/n, for loss)
   lossFct.sizeAverage = false

   -- bufferize minibatch
   local input
   if network.isSparse then input = {}
   else                     input = train.new(sgdOpt.miniBatchSize, train[1]:size(1))
   end
   
   -- prepare meta-data
   local appenderIn = sgdOpt.appenderIn
   local denseMetadata  = train[1].new(sgdOpt.miniBatchSize, self.info.metaDim or 0)
   local sparseMetadata = {}
   

   -- shuffle index for data
   local noSample = self.info.size
   local shuffle  = torch.randperm(noSample)


   -- Start training
   network:training()

   local cursor = 1
   while cursor < noSample-1 do

      -- prepare minibatch
      local noPicked = 1
      while noPicked <= sgdOpt.miniBatchSize and cursor < noSample-1 do

         local shuffledIndex = shuffle[cursor]
         
         if train[shuffledIndex] then -- handle table that are not contiguous
            input[noPicked] = train[shuffledIndex]
            
            if appenderIn then
               denseMetadata[noPicked]  = self.info[shuffledIndex].full
               sparseMetadata[noPicked] = self.info[shuffledIndex].fullSparse
            end
            
            noPicked = noPicked + 1
         end

         cursor = cursor + 1
      end


      -- Define the closure to evalute w and dw
      local function feval(x)

         -- Reset gradients and losses
         network:zeroGradParameters()

         --Prepare metadata
         if appenderIn then
            appenderIn:prepareInput(denseMetadata, sparseMetadata)
         end
         
         -- AutoEncoder targets
         local target = input
      
         -- Compute noisy input for Denoising AutoEnc 
         local noisyInput = lossFct:prepareInput(input) 
         

         --- FORWARD
         local output = network:forward(noisyInput)
         local loss   = lossFct:forward(output, target)
         
         --- BACKWARD
         local dloss = lossFct:backward(output, target)
         local _     = network:backward(noisyInput, dloss)

         -- Return loss and gradients
         return loss/sgdOpt.miniBatchSize, dw:div(sgdOpt.miniBatchSize)
      end


      -- Optimize current iteration
      sgdOpt.evalCounter = epoch

      -- compute new regularization according input/output 
      if self.tikhonov then
         self.tikhonov:computeLambda(input)
      end

      optim.sgd (feval, w, sgdOpt )

   end


end


inf = 1/0
function  AutoEncoderTrainer:Test(sgdOpt)

   local network = self.network

   local train   = self.train
   local test    = self.test

   local loss, rmse, mae = inf,inf,inf


   -- start evaluating
   network:evaluate()
      
   local appenderIn = sgdOpt.appenderIn

   if self.isSparse then

      -- Configure prediction error
      local rmseFct = nnsparse.SparseCriterion(nn.MSECriterion())
      local maeFct  = nnsparse.SparseCriterion(nn.AbsCriterion()) 

      -- remove criterion normalization, it will be done by hand to handle sparsity constraints
      rmseFct.sizeAverage = false
      maeFct.sizeAverage  = false

      rmse, mae = 0, 0

      -- Create minibatch
      local noRatings = 0
      local input, target, denseMeta, sparseMeta = {}, {}, {}, {}
      local minibatch = {}
      
      for k, _ in pairs(train) do
      
        if test[k] ~= nil then --ignore when there is no target 
          input     [#input  +1]  = train[k] 
          target    [#target +1]  = test[k]
          
          if appenderIn then
            denseMeta [#denseMeta + 1]  = self.info[k].full
            sparseMeta[#sparseMeta+ 1] = self.info[k].fullSparse
          end
          
          noRatings = noRatings + test[k]:size(1)
        
          if #input == sgdOpt.miniBatchSize then
            minibatch[#minibatch+1] = 
            {
               input = input, 
               target = target, 
               sparseMeta = sparseMeta, 
               denseMeta = denseMeta
            }
            
            input, target, denseMeta, sparseMeta = {}, {}, {}, {}  
          end
        end
      end
      
      if #input > 0 then 
         minibatch[#minibatch+1] = {
            input      = input, 
            target     = target, 
            sparseMeta = sparseMeta, 
            denseMeta  = denseMeta
         }
      end
   

      -- Compute the RMSE by predicting the testing dataset thanks to the training dataset
      local err = 0
      for _, oneBatch in pairs(minibatch) do
      
        --Prepare metadata
        if appenderIn then
           appenderIn:prepareInput(oneBatch.denseMeta, oneBatch.sparseMeta)
        end
      
        local output = network:forward(oneBatch.input)
        
        rmse = rmse + rmseFct:forward(output, oneBatch.target)
        mae  = mae  + maeFct:forward(output, oneBatch.target)
      end
      

      rmse = rmse/noRatings
      mae  = mae/noRatings
      
   else
   
      local lossFct    = nn.MSECriterion()
      local batchifier = cfn.Batchifier(network, nil, appenderIn, self.info)
      local output = batchifier:forward(test)
     
      loss = lossFct:forward(output, test)
   end

   return math.sqrt(loss), math.sqrt(rmse), mae

end




function AutoEncoderTrainer:Execute(sgdOpt)

   local noEpoch = sgdOpt.noEpoch

   for t = 1, noEpoch do

      xlua.progress(t, noEpoch)
      print("")

      --train one epoch
      self:Train(sgdOpt, t)

      local newReconstructionRMSE, newPredictionRMSE, newMAE = self:Test(sgdOpt)

      --rescale RMSE/MAE
      newPredictionRMSE     = newPredictionRMSE     * 2
      newMAE                = newMAE                * 2

      self.rmse[#self.rmse+1] =  newPredictionRMSE
      self.mae [#self.mae +1] =  newMAE

      if newPredictionRMSE ~= inf then --and newMAE ~= NaN then
         print(t .. "/" .. noEpoch .. "\t RMSE : "  .. newPredictionRMSE .. "\t MAE : "  .. newMAE )
      else
         print(t .. "/" .. noEpoch .. "\t RMSE : "  .. newReconstructionRMSE )
      end

   end

   return self.rmse, self.mae

end







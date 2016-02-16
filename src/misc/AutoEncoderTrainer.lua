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

      --Prepare minibatch
      local input  = {}
      local target = {}

      -- prepare meta-data
      local denseMetadata  = train[1].new(sgdOpt.miniBatchSize, self.info.metaDim or 0)
      local sparseMetadata = {}

      local i = 1
      local noRatings = 0


      for k, oneInput in pairs(train) do

         -- Focus on the prediction aspect
         local oneTarget = test[k]

         -- Ignore data with no testing examples
         if oneTarget ~= nil then

            -- autoencoder input/target
            input[i]  = oneInput

            -- bufferize target
            target[i] = target[i] or oneTarget.new()
            target[i]:resizeAs(oneTarget):copy(oneTarget)

            -- center the target values
            target[i][{{}, 2}]:add(-self.info[k].mean)

            -- append metadata
            if appenderIn then
               denseMetadata[i]  = self.info[k].full
               sparseMetadata[i] = self.info[k].fullSparse
            end

            --compute the current number of ratings
            noRatings  = noRatings + oneTarget:size(1)
            i = i + 1


            --compute loss when minibatch is ready
            if #input == sgdOpt.miniBatchSize then

               --Prepare metadata
               if appenderIn then
                  appenderIn:prepareInput(denseMetadata, sparseMetadata)
               end
               
               local output = network:forward(input)

               rmse = rmse + rmseFct:forward(output, target)
               mae  = mae  + maeFct:forward(output, target)

               --reset minibatch
               input = {}
               i = 1               
 
            end
         end
      end

      -- remaining data for minibatch
      if #input > 0 then
         local _targets = { unpack(target, 1, #input)} --retrieve a subset of targets
         
         if appenderIn then
            local _sparseMetadata = {unpack(sparseMetadata, 1, #input)}
            local _denseMetadata =  denseMetadata[{{1, #input},{}}] 
            
            appenderIn:prepareInput(_denseMetadata, _sparseMetadata)
         end
        
         local output = network:forward(input)

         rmse = rmse + rmseFct:forward(output, _targets)
         mae  = mae  + maeFct:forward(output , _targets)
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







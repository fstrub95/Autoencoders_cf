require("nn")
require("optim")
require("xlua") 

torch.setdefaulttensortype('torch.FloatTensor') 

require("nnsparse")

dofile("tools/SDAECriterionGPU.lua")
dofile("misc/Preload.lua")



----------------------------------
-- Configuration
----------------------------------

local inputSize = 6040

local sgdConfiguration = {
   learningRate      = 0.07, 
   learningRateDecay = 0.3,
   weightDecay       = 0.05,
}

local lossFct = cfn.SDAECriterionGPU(nn.MSECriterion(), {
   alpha     = 1,
   beta      = 0.5,
   hideRatio = 0.25,
}, inputSize)
lossFct.sizeAverage = false


local batchSize = 35
local epoches = 20



----------------------------------
-- Loading data
----------------------------------
print("Start Loading data...")


local train, test = {}, {}

-- Step 1 : Load file
local ratesfile = io.open("ml-1m/ratings.dat", "r")

-- Step 2 : Retrieve ratings
for line in ratesfile:lines() do


   -- parse the file by using regex
   local userIdStr, movieIdStr, ratingStr, _ = line:match('(%d+)::(%d+)::(%d%.?%d?)::(%d+)')

   local userId  = tonumber(userIdStr)
   local itemId  = tonumber(movieIdStr)
   local rating  = tonumber(ratingStr)

   -- normalize the rating between [-1, 1]
   rating = (rating-3)/2

   -- we are going to autoencode the item with a training ratio of 0.9
   if torch.uniform() < 0.9 then
      train[itemId] = train[itemId] or nnsparse.DynamicSparseTensor()
      train[itemId]:append(torch.Tensor{userId, rating})
   else
      test[itemId] = test[itemId] or nnsparse.DynamicSparseTensor()
      test[itemId]:append(torch.Tensor{userId, rating})
   end 

end

-- Step 3 : Build the final sparse matrices
for k, oneTrain in pairs(train) do train[k] = oneTrain:build():ssortByIndex() end
for k, oneTest  in pairs(test) do test[k]  = oneTest:build():ssortByIndex() end


-- Step 4 : remove mean
for k, oneTrain in pairs(train) do 
   local mean = oneTrain[{ {},2 }]:mean()
   train[k][{ {},2 }]:add(-mean) 
   
   if test[k] then test[k] [{ {},2 }]:add(-mean) end 
end



----------------------------------
-- Building the network
----------------------------------
print("Start Building the network...")

local network = nn.Sequential()
network:add(nnsparse.SparseLinearBatch(inputSize, 770)) --There are 6040 users in movieLens-1M
network:add(nn.Tanh())
network:add(nn.Linear(770, inputSize))
network:add(nn.Tanh())

print(network)


----------------------------------
-- Training the network
----------------------------------

local function trainNN(network, t)

   -- Create minibatch
   local input, minibatch = {}, {}

   --shuffle the indices of the inputs to create the minibatch 
   local shuffle = torch.randperm(inputSize)
   shuffle:apply(function(k)
      if train[k] then
         input[#input+1] = train[k] 
         if #input == batchSize then
            minibatch[#minibatch+1] = input
            input = {}  
         end
      end
   end)
   if #input > 0 then 
      minibatch[#minibatch+1] = input 
   end


   local w, dw = network:getParameters()
   lossFct.sizeAverage = false

   -- Classic training 
   for _, input in pairs(minibatch) do
      local function feval(x)

         -- Reset gradients and losses
         network:zeroGradParameters()

         -- AutoEncoder targets
         local target = input

         -- Compute noisy input for Denoising autoencoders
         local noisyInput = lossFct:prepareInput(input) 

         -- FORWARD
         local output = network:forward(noisyInput)
         local loss   = lossFct:forward(output, target)

         -- BACKWARD
         local dloss = lossFct:backward(output, target)
         local _     = network:backward(noisyInput, dloss)

         -- Return loss and gradients
         return loss/batchSize, dw:div(batchSize)
      end
      
      sgdConfiguration.evalCounter = t
      optim.sgd (feval, w, sgdConfiguration )

   end  
   
end





----------------------------------
-- Testing the network
----------------------------------

local function testNN(network)

   local criterion = nnsparse.SparseCriterion(nn.MSECriterion())


   -- Create minibatch
   local noRatings = 0
   local input, target, minibatch = {}, {}, {}
   
   for k, _ in pairs(train) do
   
     if test[k] ~= nil then --ignore when there is no target 
       input [#input  +1] = train[k] 
       target[#target +1] = test[k]
       
       noRatings = noRatings + test[k]:size(1)
     
       if #input == batchSize then
         minibatch[#minibatch+1] = {input = input, target = target}
         input, target = {}, {}  
       end
     end
   end
   if #input > 0 then 
      minibatch[#minibatch+1] = {input = input, target = target} 
   end

   -- define the testing criterion
   local criterion = nnsparse.SparseCriterion(nn.MSECriterion())
   criterion.sizeAverage = false

   -- Compute the RMSE by predicting the testing dataset thanks to the training dataset
   local err = 0
   for _, oneBatch in pairs(minibatch) do
     local output = network:forward(oneBatch.input)
     err = err + criterion:forward(output, oneBatch.target)
   end
   
   err = err/noRatings

   print("Current RMSE : " .. math.sqrt(err) * 2)

end
   
print("Start Training the network...")   
for t = 1, epoches do
   xlua.progress(t, epoches)
   trainNN(network, t)
   testNN(network)
end
   

   
require("nn")
require("torch")

dofile ("tools.lua")


local AbsCriterion2 = torch.class('nn.AbsCriterion2')

function AbsCriterion2:forward(x,y)
   self.output = self.output or x.new()
   self.output:resizeAs(x):copy(x):add(-1,y):abs()
   return self.output:sum()
end



function algoTrain(train, test, algo, conf)

   local Usize = train.U.size
   local Vsize = train.V.size

   algo:init(train.U, train.V, conf)


   local M       = torch.Tensor(Usize, Vsize):fill(NaN)
   local lossFct = nnsparse.SparseCriterion(nn.MSECriterion())
   local algoLoss = 0

   local bestLoss = 999

   local noEpoch = conf.epoches or 15

   for t = 1, noEpoch do

      algo:eval(M)

      local algoLoss = 0
      local noRating = 0
      for i, target in pairs(test.U.data) do
            local size = target:size(1)
            algoLoss = algoLoss + lossFct:forward(M[{i,{}}], target)*size
            noRating = noRating + size
      end
      algoLoss = algoLoss / noRating
      

      print("Loss = " .. math.sqrt(algoLoss)*2)

      if algoLoss > bestLoss then
         print("early stopping")
         break
      else
         bestLoss = algoLoss
      end 

   end

   print("Algo loss = " .. math.sqrt(bestLoss)*2)
   return math.sqrt(bestLoss), algo.U, algo.V

end




function FlatNetwork(network)

   function FlatNetworkRecursive(network, layers)
      for i = 1, network:size() do
         local layer = network:get(i)
         if torch.type(layer) == "nn.Sequential" then
            FlatNetworkRecursive(layer, layers)
         else
            layers[#layers+1] = layer
         end
      end
   end
   
   local layers = {}
   FlatNetworkRecursive(network, layers)
   
   
   local flatNetwork = nn.Sequential()
   for _, layer in pairs(layers) do
      flatNetwork:add(layer)
   end
   
   return flatNetwork
end


function sortSparse(X)
   
   local _ , index = X[{{},1}]:sort()
   local sX = torch.Tensor():resizeAs(X)
   
   for k = 1, index:size(1) do
      sX[k] = X[index[k]]   
   end

   return sX
end



local function GetnElement(X) 
   if torch.isTensor(X)  then 
      return X:nElement()
   elseif torch.type(X) == "table" then 
      local size = 0
      for _, _ in pairs(X) do size = size + 1 end
      return size
   else return nil
   end

end

local Batchifier2, parent = torch.class('nnsparse.Batchifier2')

function Batchifier2:__init(network, outputSize, appenderIn, info)
   self.network    = network
   self.outputSize = outputSize
   self.appenderIn = appenderIn
   self.info = info
end

function Batchifier2:forward(data, batchSize)
   
   -- no need for batch for dense Tensor
   if torch.isTensor(data) then
   
      if self.appenderIn then
         local denseInfo  = data[1].new(self.info.size, self.info.metaDim)
         for k = 1, data:size(1) do
            denseInfo[k] = self.info[k] or 0
         end
         self.appenderIn:prepareInput(denseInfo)
      end
      
      return self.network:forward(data)
   end
      
   batchSize = batchSize or 20
   
   local nFrame    = GetnElement(data)

   --Prepare minibatch
   local inputs   = {}
   local outputs  = data[1].new(nFrame, self.outputSize) 
   
   local denseInfo  = data[1].new(batchSize, self.info.metaDim):zero()
   local sparseInfo = {}
   
   assert(torch.type(data) == "table")

   local i      = 1
   local cursor = 0
   for k, input in pairs(data) do

      inputs[i]  = input   
      
      if self.appenderIn then
          denseInfo[i]  = self.info[k].full
          sparseInfo[i] = self.info[k].fullSparse
      end
      
      i = i + 1

      --compute loss when minibatch is ready
      if #inputs == batchSize then
         local start =  cursor   *batchSize + 1
         local stop  = (cursor+1)*batchSize

         if self.appenderIn then
            self.appenderIn:prepareInput(denseInfo,sparseInfo)
         end
         
         outputs[{{start,stop},{}}] = self.network:forward(inputs)
         
         inputs = {}
         
         i = 1
         cursor = cursor + 1       
      end
   end

   if #inputs > 0 then
      local start = nFrame-(i-1) + 1
      local stop  = nFrame

      if self.appenderIn then
         self.appenderIn:prepareInput(denseInfo[{{1, #inputs},{}}], {unpack(sparseInfo, 1, #inputs)})
      end

      outputs[{{start,stop},{}}] = self.network:forward(inputs)
   end  

   return outputs

end
   

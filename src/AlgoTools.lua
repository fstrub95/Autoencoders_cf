require("nn")
require("torch")

dofile ("tools.lua")

require("nnsparse")




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






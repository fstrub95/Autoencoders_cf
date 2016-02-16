function algoTrain(train, test, algo, conf)

   local Usize = train.U.info.size
   local Vsize = train.V.info.size

   algo:init(train.U.info, train.V.info, conf)

   local lossFct = nnsparse.SparseCriterion(nn.MSECriterion())
   lossFct.sizeAverage = false
   
   
   local bestLoss = 999

   local noEpoch = conf.epoches or 15

   for t = 1, noEpoch do

      local U, V = algo:eval()

      local curLoss = 0
      local noRatings = 0
      
      local line = torch.Tensor(Vsize)
      
      for i, target in pairs(test.U.data) do
            --compute one dense line
            line:mv(V, U[i])
            
            --compute the loss
            curLoss  = curLoss + lossFct:forward(line, target)
            noRatings = noRatings + target:size(1)
      end
      curLoss = curLoss / noRatings
      

      print("Loss = " .. math.sqrt(curLoss)*2)

      if curLoss > bestLoss then
         print("early stopping")
         break
      else
         bestLoss = curLoss
      end 

   end

   bestLoss = math.sqrt(bestLoss)*2

   print("Algo loss = " .. bestLoss)
   return bestLoss, algo.U, algo.V

end

require("nn")
require("torch")

torch.setdefaulttensortype('torch.FloatTensor') 
require("nnsparse")

dofile ("../tools.lua")
dofile ("BenchmarkTools.lua")


GradDescent = {} 

function GradDescent:new(trainU, trainV)
   newObj = 
      {
         lossFct = nn.MSECriterion()
      }

   --precompute data
   local function computeSpareMatrix(train)
   
      local noRatings = 0
      for _, oneTrain in pairs(train.data) do
         noRatings = noRatings + oneTrain:size(1)
      end
      
      
      local M = torch.Tensor(noRatings, 3)
      
      
      local cursor = 1
      
      for i, ratings in pairs(train.data) do
         for k = 1, ratings:size(1) do
            
            local j =  ratings[k][1]
            local t =  ratings[k][2]
            
            M[cursor] = torch.Tensor{i,j,t}
            
            cursor = cursor + 1
         end
      end
         
      return M
   end
   
   newObj.targets = computeSpareMatrix(trainU)


   self.__index = self                      
   return setmetatable(newObj, self)        
end


   
function GradDescent:init(infoU, infoV, conf)

   --store conf
   self.lambda = conf.lambda
   self.rank   = conf.rank
   self.lrt    = conf.lrt

   



   -- initialize U.V
   self.U = torch.Tensor(infoU.size, self.rank):uniform(-0.01, 0.01)
   self.V = torch.Tensor(infoV.size, self.rank):uniform(-0.01, 0.01)
   
   for i, info in pairs(infoU) do
      if torch.type(info) == "table" then
        self.U[i][1] = info.mean or 0
      end
   end
   self.V[{{}, 1}] = 1
   
   
   -- preallocate memory
   self.du = torch.Tensor(self.rank)
   self.dv = torch.Tensor(self.rank)
end


function GradDescent:eval()


   local targets = self.targets 
   local shuffle = torch.randperm(targets:size(1))

   -- main algo
   for k = 1, shuffle:size(1) do
      local indexShuffle = shuffle[k]
      local target = targets[indexShuffle]

      local i = target[1]
      local j = target[2]
      local t = target[3]

      local u = self.U[i]
      local v = self.V[j]

      local err = u:dot(v)-t --MSE 

      self.du:copy(v):mul(err):add(self.lambda, u)
      self.dv:copy(u):mul(err):add(self.lambda, v)

      u:add(-self.lrt, self.du)
      v:add(-self.lrt, self.dv)

   end

   return self.U, self.V

end








function pickBestGrad(train, test, ranks, lambdas, learningRates)


ranks = ranks or
{
--7,
--8,
--9,
10,
--11,
12,
--15,
20
} 

lambdas = lambdas or
{ 
  0.005,
  0.01,
  0.02,
  0.03,
--  0.04,
  0.05,
--  0.06,
--  0.07,
--  0.08,
--  0.09,
  0.1,
--  0.11,
--  0.12,
  0.15,
  0.2,
  
}

learningRates = learningRates or
{
   0.01,
   0.02,
   0.05,
   0.1 ,
} 


   local algoLoss = 999
   local algo  = GradDescent:new(train.U, train.V)
   local algoU, algoV
   local lrt, rank, lambda


   for h = 1, #learningRates do
      print("[Grad] new lrt: " .. learningRates[h] )

      for i = 1, #ranks do
         print("[Grad] new rank: " .. ranks[i] )

         for j = 1, #lambdas do
            print("[Grad] new lambda: " .. lambdas[j] )

            -- compute SVD 
            local loss, U, V = algoTrain(train, test, algo, {
               lrt = learningRates[h],
               rank = ranks[i],
               lambda = lambdas[j], 
               epoches = 50
            })

            if loss < algoLoss then
               algoLoss = loss
               algoU  = U
               algoV  = V
               lrt    = learningRates[h]
               rank   = ranks[i]
               lambda = lambdas[j]
            end

         end
      end
   end

   print("")
   print("-----------------------------------------")
   print("Best Grad : " .. algoLoss)
   print(" - lrt    = " .. lrt)
   print(" - rank   = " .. rank)
   print(" - lambda = " .. lambda)
   print("-----------------------------------------")
   print("")
   
   return algoU, algoV, algoLoss, rank, lambda

end


----------------------------------------------------------------------
-- parse command-line options
--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Basic SVD with gradient Descent')
cmd:text('Warning : This benchmark was not optimized. Prefer mahout for big dataset.')
cmd:text()
cmd:text('Options')
-- general options:
cmd:option('-file'        , './movieLens-1M.t7' , 'The relative path to your data file')
cmd:option('-rank'        , 15                  , 'Rank of the final matrix')
cmd:option('-lambda'      , 0.02                , 'Regularisation')
cmd:option('-lrt'         , 0.02                , 'Learning rate')
cmd:option('-seed'        , 0                   , 'The seed')
cmd:text()



local params = cmd:parse(arg)

print("Options: ")
for key, val in pairs(params) do
   print(" - " .. key  .. "  \t : " .. val)
end


if params.seed > 0 then
   torch.manualSeed(params.seed)
else
   torch.manualSeed(torch.seed())
end


--Load data
print("loading data...")
local data = torch.load(params.file) 
local train = data.train
local test  = data.test

print(data.train.U.info.size .. " Users loaded")
print(data.train.V.info.size .. " Items loaded")

   
local U, V = pickBestGrad(train, test, {params.rank}, {params.lambda}, {params.lrt})





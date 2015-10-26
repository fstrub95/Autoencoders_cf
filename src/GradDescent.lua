require("nn")
require("torch")

torch.setdefaulttensortype('torch.FloatTensor') 
require("nnsparse")

dofile("data.lua")
dofile ("tools.lua")
dofile ("AlgoTools.lua")

GradDescent = {} 

function GradDescent:new(trainU, trainV)
   newObj = 
      {
         lossFct = nn.MSECriterion()
      }

   --precompute data
   local function computeSpareMatrix(train)
      local M = {} 
      for i, ratings in pairs(train.data) do
         for k = 1, ratings:size(1) do
            
            local j =  ratings[k][1]
            local t =  ratings[k][2]
            
            table.insert(M, {
               i = i,
               j = j,
               val = t,
            })
         end
      end
         
      return M
   end
   
   newObj.targets = computeSpareMatrix(trainU)


   self.__index = self                      
   return setmetatable(newObj, self)        
end


   
function GradDescent:init(trainU, trainV, conf)

   --store conf
   self.lambda = conf.lambda
   self.rank   = conf.rank
   self.lrt    = conf.lrt


   -- initialize U.V
   self.U = torch.Tensor(trainU.size, self.rank):uniform(-0.01, 0.01)
   self.V = torch.Tensor(trainV.size, self.rank):uniform(-0.01, 0.01)
   
   for i, info in pairs(trainU.info) do
      self.U[i][1] = info.mean 
   end
   self.V[{{}, 1}] = 1
   
   
   -- preallocate memory
   self.du = torch.Tensor(self.rank)
   self.dv = torch.Tensor(self.rank)
end


function GradDescent:eval(M)


   local targets = self.targets 
   local shuffle = torch.randperm(#targets)

   -- main algo
   for k = 1, shuffle:size(1) do
      local indexShuffle = shuffle[k]
      local target = targets[indexShuffle]

      local i = target.i
      local j = target.j
      local t = target.val

      local u = self.U[i]
      local v = self.V[j]

      local err = u:dot(v)-t --MSE 

      self.du:copy(v):mul(err):add(self.lambda, u)
      self.dv:copy(u):mul(err):add(self.lambda, v)

      u:add(-self.lrt, self.du)
      v:add(-self.lrt, self.dv)

   end

   if M then
      M:copy(self.U*self.V:t())
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

            -- compute the ALS
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
   print("Best Grad : " .. algoLoss*2)
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
local arg = {}
cmd = torch.CmdLine()
cmd:text()
cmd:text('Learn SDAE network for collaborative filtering')
cmd:text()
cmd:text('Options')
-- general options:
cmd:option('-fileType'    , "movieLens"                        , 'The data file format (jester/movieLens/classic)')
cmd:option('-file'        , '../data/movieLens/ratings-1M.dat' , 'The relative path to your data file')
cmd:option('-ratio'       , 0.9                                , 'The training ratio')
cmd:option('-rank'        , 15                                 , 'Rank of the final matrix')
cmd:option('-lambda'      , 0.02                               , 'Regularisation')
cmd:option('-lrt'         , 0.02                               , 'Learning rate')
cmd:option('-seed'        , 1234                               , 'The seed')
cmd:option('-out'         , '../out.csv'                       , 'The path to store the final matrix (csv) ')
cmd:text()



local params = cmd:parse(arg)


torch.manualSeed(params.seed)
math.randomseed(params.seed)


--Load data
local train, test = LoadData(
   {
      type  = params.fileType,
      ratio = params.ratio,
      file  = params.file,
   })
   
local U, V = pickBestGrad(train, test, {params.rank}, {params.lambda}, {params.lrt})


print("Saving Matrix...")
tensorToCsv(U*V:t(), params.out)
print("done!")




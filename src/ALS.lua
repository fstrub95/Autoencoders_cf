require("torch")
require("nn")

torch.setdefaulttensortype('torch.FloatTensor')

require("nnsparse")

dofile ("tools.lua")
dofile ("AlgoTools.lua")

-- ALS, implementing
-- http://www.grappa.univ-lille3.fr/~mary/cours/stats/centrale/reco/paper/MatrixFactorizationALS.pdf


ALS = {}

function ALS:new(trainU, trainV)
   newObj = { }

   local function computeSpareRates(train, nSize)

      local res = {}

      for i = 1, #train do

         if train[i] ~= nil then

            local index = train[i][{{},1}]:long()
            local rates = train[i][{{},2}]
            local noRatedElem = rates:size(1)

            res[i] = {}

            res[i].index       = index
            res[i].noRatedElem = noRatedElem
            res[i].rates       = rates

         else
            --print(i .. " ignored")
         end

      end

      return res
   end

   --precompute data
   newObj.sparseRates = {}

   newObj.sparseRates["U"] = computeSpareRates(trainU.data, trainU.dimension)
   newObj.sparseRates["V"] = computeSpareRates(trainV.data, trainV.dimension)


   self.__index = self

   return setmetatable(newObj, self)       
end

function ALS:init(trainU, trainV, conf)

   self.lambda = conf.lambda
   self.rank   = conf.rank


   -- Matrix decomposition
   self.U = torch.Tensor(trainU.size, self.rank):uniform(-0.01, 0.01)
   self.V = torch.Tensor(trainV.size, self.rank):uniform(-0.01, 0.01)
  
   self.U[{{}, 1}] = 1
--   self.V[{{}, 1}] = 1
  

end


--            if self.useBias then
--               local bias = self.bias[torch.pointer(Y)][mask] -- Ybias
--               bias:add(self.bias[torch.pointer(X)][i])       -- Xbias
--               b:addmv(-1, Y_, bias)                          -- b = b - Y_*(Xbias[i] + Ybias[:])
--            end

  
function ALS:eval(M)

   --- Solve the ALS linear system:
   --
   -- sum( x'y - r)*y_k + Reg = 0
   --
   -- @param X (in/out) vector to update
   -- @param data (in) input data
   -- @param Y (in) vector that is set
   local function updateVector(X, Y, sparseRate)

      self.Y_   = self.Y_  or X.new()
      self.Yt_  = self.Yt_ or X.new()
      
      self.Regu = self.Regu or X.new(self.rank)
      
      self.A = self.A or X.new(self.rank, self.rank)
      self.b = self.b or X.new()
      
      local mask = torch.Tensor():byte():resize(Y:size(1),1)

     
      for i = 1, X:size(1) do

         local data = sparseRate[i]

         if data ~= nil then

            -- retrieve the sparsed rated items/users
            local r_             = data.rates
            local index          = data.index
            local noRatedElem    = data.noRatedElem
  
            --compute mask with no memory allocation
            mask:zero():indexFill(1, index, 1)
           

            --move Y with no memory allocation
            self.Y_ :resize(noRatedElem, self.rank)
            self.Y_:maskedSelect(Y, mask:expand(Y:size()))
            
            --resize Y and its transpose to fit the problem dimension  
            self.Y_:resize(noRatedElem, self.rank)
            self.Yt_ = self.Y_:t()

            -- compute the regularization
            self.Regu:fill(self.lambda*noRatedElem)
            
            self.A:diag(self.Regu):addmm(self.Yt_, self.Y_)

            -- step 2 : Solve the linear system
            self.A:diag(self.Regu):addmm(self.Yt_, self.Y_)
            self.b:resize(self.rank):mv(self.Yt_, r_)

            -- solve linear system
            X[i]:copy(torch.gesv(self.b:view(-1,1),self.A))

         end
        
      end
   end

   -- Main algorithm
   updateVector(self.U, self.V, self.sparseRates["U"])
   updateVector(self.V, self.U, self.sparseRates["V"])

   if M then
      M:copy(self.U*self.V:t())
   end

   return self.U, self.V

end














function pickBestAls(train, test, ranks, lambdas)


ranks = ranks or
{
--7,
--8,
--9,
--10,
--11,
--12,
--15,
20
}

lambdas = lambdas or
{
  0.005,
  0.01,
  0.02,
  0.03,
  0.04,
  0.05,
  0.06,
  0.07,
  0.08,
  0.09,
  0.1,
  0.11,
  0.12,
  0.15,
  0.2,
}


local alsLoss = 999
local als  = ALS:new(train["U"], train["V"])
local alsU, alsV
local rank, lambda



for i = 1, #ranks do
   print("[ALS] new rank: " .. ranks[i] )

   for j = 1, #lambdas do
      print("[ALS] new lambda: " .. lambdas[j] )

      -- compute the ALS
      local loss, U, V = algoTrain(train, test, als, {
         lambda = lambdas[j],
         rank = ranks[i]
      })

      if loss < alsLoss then
         alsLoss = loss
         alsU    = U
         alsV    = V
         rank    = ranks[i]
         lambda  = lambdas[j]
      end

   end
end

   print("")
   print("-----------------------------------------")
   print("ALS loss : " .. alsLoss*2)
   print(" - rank   = " .. rank)
   print(" - lambda = " .. lambda)
   print("-----------------------------------------")
   print("")

return alsU, alsV, alsLoss, rank, lambda

end





----------------------------------------------------------------------
-- parse command-line options
--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Learn SDAE network for collaborative filtering')
cmd:text()
cmd:text('Options')

--cmd:option('-file'        , './movieLens-1M.t7'  , 'The relative path to your data file')
cmd:option('-file'        , './movieLens-1M.t7'  , 'The relative path to your data file')
cmd:option('-rank'        , 15                   , 'Rank of the final matrix')
cmd:option('-lambda'      , 0.03                 , 'Regularisation')
cmd:option('-seed'        , 1234                 , 'The seed')

cmd:text()


local params = cmd:parse(arg)

print("Options: ")
for key, val in pairs(params) do
   print(" - " .. key  .. "  \t : " .. val)
end


torch.manualSeed(params.seed)
math.randomseed(params.seed)


--Load data
--Load data
print("loading data...")
local data = torch.load(params.file)
local train = data.train
local test  = data.test

print(train.U.size .. " Users loaded")
print(train.V.size .. " Items loaded")

local U, V = pickBestAls(train, test, {params.rank}, {params.lambda})
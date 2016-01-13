-- Load global libraries
require("nn")
require("optim")
require("xlua") 

torch.setdefaulttensortype('torch.FloatTensor') 

require("nnsparse")

dofile("AlgoTools.lua")

dofile("AutoEncoderTrainer.lua")
dofile("LearnU.lua")
dofile("SDAECriterionGPU.lua")
dofile("Appender.lua")

----------------------------------------------------------------------
-- parse command-line options
--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Learn SDAE network for collaborative filtering')
cmd:text()
cmd:text('Options')
-- general options:
cmd:option('-file'           , './dummy.t7'            , 'The relative path to your data file (torch format)')
cmd:option('-conf'           , "grid.template.lua"     , 'The relative path to the lua configuration file')
cmd:option('-seed'           , 1234                    , 'The seed')
cmd:option('-gpu'            , 0                       , 'use gpu')
cmd:text()


local params = cmd:parse(arg)

print("Options: ")
for key, val in pairs(params) do
   print(" - " .. key  .. "  \t : " .. tostring(val))
end

dofile(params.conf)


--Load data
print("loading data...")
local data = torch.load(params.file) 
local train = data.train
local test  = data.test

print(train.U.size .. " Users loaded")
print(train.V.size .. " Items loaded")



if params.gpu > 0 then 

   USE_GPU = params.gpu
   cutorch.setDevice(params.gpu)  

   print("Loading cunn...")
   require("cunn")

   print("Loading data to GPU")
   local function toGPU(type)
      local _train = train[type]
      local _test  = test [type]

      for k, _ in pairs(train[type].data) do

         _train.data[k] = _train.data[k]:cuda()

         if _test .data[k] then  
            _test .data[k] = _test .data[k]:cuda()
         end

         if _train.info.metaDim then
            _train.info[k].full = _train.info[k].full:cuda()
         end
      end
   end

   toGPU("U")
   toGPU("V")

end



-- unbias U
for k, u in pairs(train.U.data) do
   u[{{}, 2}]:add(-train.U.info[k].mean) --center input
end

for k, v in pairs(train.V.data) do
   train.V.info[k] = train.V.info[k] or {}     
   train.V.info[k].mean = v[{{}, 2}]:mean()

   v[{{}, 2}]:add(-train.V.info[k].mean) --center input
end



print()

for _, _networkType   in pairs(networkType)     do
for _, _useMetaInfo   in pairs(useMetaInfo)     do
for _, _batchSize     in pairs(batchSize)       do
for _, _layerSize2    in pairs(layerSize2)      do
for _, _layerSize1    in pairs(layerSize1)      do
for _, _batchSize     in pairs(batchSize)       do
for _, _hideratio     in pairs(hideratio)       do 
for _, _alpha         in pairs(alpha)           do  
for _, _beta          in pairs(beta)            do
for _, _learningRate  in pairs(learningRate)   do
for _, _learningDecay in pairs(learningDecay)   do
for _, _weightDecay   in pairs(weightDecay)    do

   __networkType    = _networkType
   __useMetaInfo    = _useMetaInfo
   __batchSize      = _batchSize
   __layerSize1     = _layerSize1
   __layerSize2     = _layerSize2
   __weightDecay    = _weightDecay
   __learningRate   = _learningRate
   __learningDecay  = _learningDecay
   __alpha          = _alpha
   __beta           = _beta
   __hideratio      = _hideratio

   
   print("New network: " )
   print("- networkType   : " .. tostring(__networkType))
   print("- useMetaInfo   : " .. tostring(__useMetaInfo))
   print("- batchSize     : " .. tostring(__batchSize))
   print("- layerSize1    : " .. tostring(__layerSize1))
   print("- layerSize2    : " .. tostring(__layerSize2))
   print("- weightDecay   : " .. tostring(__weightDecay))
   print("- learningRate  : " .. tostring(__learningRate))
   print("- learningDecay : " .. tostring(__learningDecay))
   print("- alpha         : " .. tostring(__alpha))
   print("- beta          : " .. tostring(__beta))
   print("- hideratio     : " .. tostring(__hideratio))
   print("")

   dofile(params.conf)

   if     __networkType == "U" then trainU(train, test, __config)
   elseif __networkType == "V" then trainV(train, test, __config)           
   else   
      error("Invalid network type")
   end
   
   
end
end
end
end
end
end
end
end
end
end
end
end

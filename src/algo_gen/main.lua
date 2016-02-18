-- Load global libraries
require("nn")
require("optim")
require("xlua") 

torch.setdefaulttensortype('torch.FloatTensor') 

require("nnsparse")

dofile("../misc/AutoEncoderTrainer.lua")
dofile("../misc/TrainNetwork.lua")
dofile("../misc/Preload.lua")

dofile("../tools/CFNTools.lua")
dofile("../tools/LuaTools.lua")
dofile("../tools/SDAECriterionGPU.lua")
dofile("../tools/Appender.lua")

dofile("AlgoGen.lua")
dofile("NNGen.lua")


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
cmd:option('-seed'           , 1234                    , 'The seed')
cmd:option('-gpu'            , 1                       , 'use gpu')
cmd:option('-type'           , "V"                     , 'autoencoder type')
cmd:option('-meta'           , 1                       , 'use side information')
cmd:text()

local params = cmd:parse(arg)

print("Options: ")
for key, val in pairs(params) do
   print(" - " .. key  .. "  \t : " .. tostring(val))
end

if params.seed > 0 then
   torch.manualSeed(params.seed)
else
   torch.manualSeed(torch.seed())
end

GPU_DEVICE = params.gpu
NN_TYPE    = params.nn 
NO_THREAD  = params.noThread

local genConf = 
{
   noGenes     = 20,
   noEpoch     = 10,  
   ratioBest   = 1/10,
   ratioCross  = 2/10,
   ratioMutate = 3/10,
   ratioNew    = 4/10,
   sigma       = 0.01,
   file        = params.file,
   gpu         = params.gpu,
   type        = params.type,
   meta        = params.meta,
   noThread    = params.noThread
}


local searchGen = NNGen.new(genConf)
searchGen:Start(genConf)

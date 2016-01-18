-- Load global libraries
require("nn")
require("optim")
require("xlua") 

torch.setdefaulttensortype('torch.FloatTensor') 

require("nnsparse")

dofile("AlgoTools.lua")
dofile("tools.lua")

dofile("AutoEncoderTrainer.lua")
dofile("LearnU.lua")
dofile("SDAECriterionGPU.lua")
dofile("Appender.lua")


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
cmd:option('-gpu'            , 0                       , 'use gpu')
cmd:text()

local params = cmd:parse(arg)

print("Options: ")
for key, val in pairs(params) do
   print(" - " .. key  .. "  \t : " .. tostring(val))
end

torch.manualSeed(params.seed)
math.randomseed(params.seed)


local genConf = 
{
   noGenes = 20,
   noEpoch = 10,  
   ratioBest   = 1/10,
   ratioCross = 2/10,
   ratioMutate = 3/10,
   ratioNew    = 4/10,
   file = params.file
}


local searchGen = NNGen.new(genConf)
searchGen:Start(genConf)
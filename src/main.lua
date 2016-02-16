
-- Load global libraries
require("nn")
require("optim")
require("xlua") 

torch.setdefaulttensortype('torch.FloatTensor') 

require("nnsparse")

dofile("misc/Preload.lua")
dofile("misc/AutoEncoderTrainer.lua")
dofile("misc/TrainNetwork.lua")


dofile("tools/Appender.lua")
dofile("tools/SDAECriterionGPU.lua")
dofile("tools/CFNTools.lua")
dofile("tools/LuaTools.lua")

----------------------------------------------------------------------
-- parse command-line options
--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Learn SDAE network for collaborative filtering')
cmd:text()
cmd:text('Options')
-- general options:
cmd:option('-file'           , '../data/movieLens-10M.t7'         , 'The relative path to your data file (torch format). Please use data.lua to create such file.')
cmd:option('-conf'           , "../conf/conf.movieLens.10M.V.lua" , 'The relative path to the lua configuration file')
cmd:option('-seed'           , 0                                  , 'The seed. random = 0')
cmd:option('-meta'           , 1                                  , 'use metadata false = 0, true 1')
cmd:option('-type'           , 'V'                                , 'Pick either the U/V Autoencoder.')
cmd:option('-gpu'            , 1                                  , 'use gpu. CPU = 0, GPU > 0 with GPU the index of the device')
cmd:option('-save'           , ''                                 , "Store the final network in an external file")
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


print("Load training configuration...")
local config = dofile(params.conf)

--Append command line configuration
config.use_meta       = params.meta > 0
config.use_gpu        = params.gpu  > 0



print("Load data...")
local train, test, info = LoadData(params.file, params)


print("Training network")
local rmse, network = TrainNetwork(train, test, info, config)


if #params.save > 0 then
   print("Saving final network on Disk...")
   torch.save(params.save, network)
end


print("Done!!!")




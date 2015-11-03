-- Load global libraries
require("nn")
require("optim")
require("xlua") 

torch.setdefaulttensortype('torch.FloatTensor') 

require("nnsparse")

dofile("data.lua")

dofile("AlgoTools.lua")

dofile("AutoEncoderTrainer.lua")
dofile("LearnU.lua")




----------------------------------------------------------------------
-- parse command-line options
--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Learn SDAE network for collaborative filtering')
cmd:text()
cmd:text('Options')
-- general options:
cmd:option('-file'        , '../data/movieLens/ratings-1M.dat' , 'The relative path to your data file')
cmd:option('-conf'        , "config.template.lua"              , 'The relative path to the lua configuration file')
cmd:option('-ratio'       , 0.9                                , 'The training ratio')
cmd:option('-fileType'    , "movieLens"                        , 'The data file format (jester/movieLens/classic)')
cmd:option('-seed'        , 1234                               , 'The seed')
cmd:option('-out'         , '../out.csv'                       , 'The path to store the final matrix (csv) ')
cmd:text()




local params = cmd:parse(arg)

print("Options: ")
for key, val in pairs(params) do
   print(" - " .. key  .. "  \t : " .. val)
end



torch.manualSeed(params.seed)
math.randomseed(params.seed)


--Load data
local train, test = LoadData(
   {
      type  = params.fileType,
      ratio = params.ratio,
      file  = params.file,
   })
   
   
   
--Load configuration
dofile(params.conf)


--compute neural network
local estimate
if configU then

   -- unbias U
   for k, u in pairs(train.U.data) do
      u[{{}, 2}]:add(-train.U.info[k].mean) --center input
   end

   _, estimate = trainU(train, test, configU)
   
elseif configV then

   --unbias V
   for k, v in pairs(train.V.data) do
      train.V.info[k] = train.V.info[k] or {}     
      train.V.info[k].mean = v[{{}, 2}]:mean()
   
      v[{{}, 2}]:add(-train.V.info[k].mean) --center input
   end

   _, estimate = trainV(train, test, configV)
end



print("Saving Matrix...")
tensorToCsv(estimate, params.out)

print("done!")
























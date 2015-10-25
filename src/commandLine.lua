-- Load global libraries
require("nn")
require("optim")
require("sys") 
require("xlua") 
local json = require('cjson')

dofile("data.lua")
dofile("AutoEncoderTrainer.lua")
dofile("ALS.lua")
dofile("AlgoTools.lua")

dofile("SnifferLayer.lua")
dofile("SparseLinearBatch.lua")
dofile("SDAECriterion.lua")
dofile("SDAESparseCriterion.lua")

dofile("tsnePloter.lua")

dofile("LearnU.lua")

require("gnuplot")



gfx, Plot = dofile("ploter.lua")
gfx.clear()
disp = require("display")





 




torch.manualSeed(1234)
math.randomseed(1234)

local root = '..'
outputDir = root .. '/output/res'  


--SplitJester('../data/jester/jester-data-1-sparse-150*100.t7', 
--SplitJester('../data/jester/jester-data-1-sparse-full.t7', 
--{
--  train = 3,
--   test  = 1,
--  ratio = 0.20
--}, "full")


--SplitMovieLens("../data/movieLens/ratings-1M.dat", 
--{
--   train = 2,
--   test =  2,
--   ratio = 0.20
--})



--local train = LoadData(
--   {
--      name = "jester",
--      rates = '../data/jester/jester-train-full-0.2-density.t7'
--   })
--
--local test = LoadData(
--   {
--      name = "jester",
--      rates = '../data/jester/jester-test-full-0.2-density.t7'
--   })
   
 
--   




----------------------------------------------------------------------
-- parse command-line options
--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Learn SDAE network')
cmd:text()
cmd:text('Options')
-- general options:
cmd:option('-seed', 1234, 'initial random seed')


-- data:
--cmd:option('-type', '', 'jester / movieLens')
--cmd:option('-path', '', '')



-- data:
cmd:option('-betas'       , {}, 'beta for recronstruction')
cmd:option('-hideRatios'  , {}, 'hide ratio')
cmd:option('-flipRatios'  , {}, 'flipRatios')
cmd:option('-dropouts'    , {}, 'dropouts')
cmd:option('-batchSizes'  , {}, 'batchSizes')
cmd:option('-weightDecays', {}, 'weightDecays')

cmd:text()




local params = cmd:parse()

torch.manualSeed(params.seed)
math.randomseed(params.seed)



local train = LoadData(
   {
      name   = "movieLens",
      rates  = '../data/movieLens/ratings-1M-train.t7',
      movies = '../data/movieLens/movies-1M.dat',
      users  = '../data/movieLens/users-1M.dat',
   })

local test = LoadData(
   {
      name   = "movieLens",
      rates  = '../data/movieLens/ratings-1M-test.t7',
      movies = '../data/movieLens/movies-1M.dat',
      users  = '../data/movieLens/users-1M.dat',
   })  





local batchSizes     = json.decode(params.batchSizes) or {5,10,20,50}
local weightDecays   = json.decode(params.batchSizes) or {0,0.001,0.0005,0.0001,0.00005}

local betas      = json.decode(params.betas)       or {}
local hideRatios = json.decode(params.hideRatios)  or {}
local flipRatios = json.decode(params.flipRatios)  or {}
local dropouts   = json.decode(params.dropouts)    or {}

if #betas      == 0 then for x = 0, 1.0, 0.1 do betas     [#betas+1]      = x end end
if #hideRatios == 0 then for x = 0, 0.8, 0.1 do hideRatios[#hideRatios+1] = x end end
if #flipRatios == 0 then for x = 0, 0.5, 0.1 do flipRatios[#flipRatios+1] = x end end
if #dropouts   == 0 then for x = 0, 0.6, 0.2 do dropouts  [#dropouts+1]   = x end end







for _, _dropout     in pairs(dropouts)     do
for _, _batchSize   in pairs(batchSizes)   do
for _, _weightDecay in pairs(weightDecays) do
for _, _beta        in pairs(betas)        do
for _, _hideRatio   in pairs(hideRatios)   do

   beta        = _beta
   hideRatio   = _hideRatio
   batchSize   = _batchSize
   weightDecay = _weightDecay
   dropout     = _dropout
   
   dofile("config.template.lua")

   -- Prepare a new directory to store the results and backup previous files
   configPrinter(outputDir)
   
   print("beta:        " .. beta)
   print("hideRatio:   " .. hideRatio)
   print("bathSize:    " .. batchSize)
   print("weightDecay: " .. weightDecay)
   print("dropout:     " .. dropout)
   
   print("")
   print(configU)
   
   -- load config
   local err, UnetworkU, UnetworkV = trainU(train["U"], test["U"], configU)


end
end
end
end
end




























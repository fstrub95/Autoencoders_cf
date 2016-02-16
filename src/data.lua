require("sys")
require("torch")

torch.setdefaulttensortype('torch.FloatTensor') 

require("nnsparse")

dofile("data/DataLoader.lua")

dofile("data/MovieLensLoader.lua")
dofile("data/DoubanLoader.lua")
dofile("data/DummyLoader.lua")
dofile("data/TemplateLoader.lua")
dofile("data/ClassicLoader.lua")

dofile ("tools/LuaTools.lua")

----------------------------------------------------------------------
-- parse command-line options
--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Store Data SDAE network for collaborative filtering')
cmd:text()
cmd:text('Options')
-- general options:
cmd:option('-ratings'        , '../data/movieLens/ratings-1M.dat' , 'The relative path to your data file')
cmd:option('-metaUser'       , ''                                 , 'The relative path to your metadata file for users')
cmd:option('-metaItem'       , ''                                 , 'The relative path to your metadata file for items')
cmd:option('-tags'           , ''                                 , 'The relative path to your tag file')
cmd:option('-fileType'       , "movieLens"                        , 'The data file format (movieLens/douban/classic)')
cmd:option('-out'            , "./movieLens-1M.t7"                , 'The data file format (movieLens/douban/classic)')
cmd:option('-ratio'          , 0.9                                , 'The training ratio')
cmd:option('-seed'           , 0                                  , 'seed')
cmd:text()


local params = cmd:parse(arg)

print("Options: ")
for key, val in pairs(params) do
   print(" - " .. key  .. "  \t : " .. tostring(val))
end


if params.seed > 0 then
   torch.manualSeed(params.seed)
else
   torch.manualSeed(torch.initialSeed())
end

local dataLoader
if     params.fileType == "movieLens" then dataLoader = cfn.movieLensLoader:new()
elseif params.fileType == "classic"   then dataLoader = cfn.classicLoader:new()
elseif params.fileType == "douban"    then dataLoader = cfn.doubanLoader:new()
elseif params.fileType == "dummy"     then dataLoader = cfn.dummyLoader:new()
else
   error("Unknown data format, it must be :  movieLens / douban / classic  ")
end

local train, test = dataLoader:LoadData(params.ratio,params)

print('Successfuly saved : ' .. params.out)

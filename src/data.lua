require("sys")
require("torch")
dofile ("tools.lua")

torch.setdefaulttensortype('torch.FloatTensor') 

require("nnsparse")

dofile("DataLoader.lua")
dofile("movieLensLoader.lua")



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
cmd:option('-fileType'       , "movieLens"                        , 'The data file format (jester/movieLens/classic)')
cmd:option('-out'            , "./movieLens-1M.t7"                , 'The data file format (jester/movieLens/classic)')
cmd:option('-ratio'          , 0.8                                , 'The training ratio')
cmd:option('-seed'           , 1234                               , 'The seed')
cmd:text()



local params = cmd:parse(arg)

print("Options: ")
for key, val in pairs(params) do
   print(" - " .. key  .. "  \t : " .. tostring(val))
end


local dataLoader
if     params.fileType == "movieLens" then dataLoader = movieLensLoader:new()
elseif params.fileType == "jester"    then dataLoader = jesterLoader:new()
elseif params.fileType == "classic"   then dataLoader = classicLoader:new()
elseif params.fileType == "douban"    then dataLoader = doubanLoader:new()
elseif params.fileType == "dummy"     then dataLoader = dummyLoader:new()
else
   error("Unknown data format, it must be :  movieLens / jester / douban / none ")
end

local train, test = dataLoader:LoadData(params.ratio,params)


 

print('Successfuly saved : ' .. params.out)

   



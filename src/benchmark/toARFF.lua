-- Load global libraries
require("nn")
require("optim")
require("xlua")

torch.setdefaulttensortype('torch.FloatTensor')

require("nnsparse")

dofile("../tools/LuaTools.lua")
dofile("../tools/BenchmarkTools.lua")


cmd = torch.CmdLine()
cmd:text()
cmd:text('Turn Torch input file into SVDFeature input file')
cmd:text('Options')
-- general options:
cmd:option('-file'           , 'movieLens-10M.t7'    ,  'The relative path to your data file (torch format)')
cmd:option('-outSplit'       , 'movieLens-10M.split' ,  'The relative path to output file with split test/train')
cmd:option('-outArff'        , 'movieLens-10M.arff'  ,  'The relative path to output file ARFF')
cmd:text()

local params = cmd:parse(arg)

print("Options: ")
for key, val in pairs(params) do
   print(" - " .. key  .. "  \t : " .. tostring(val))
end


--Load data
print("loading data...")
local torchdata = torch.load(params.file)

   print("")
   print("Users loaded      : " .. torchdata.train.U.info.size)
   print("Items loaded      : " .. torchdata.train.V.info.size)
   print("")
   print("No Train ratings  : " .. torchdata.train["U"].info.noRating)
   print("No Test  ratings  : " .. torchdata.test["U"] .info.noRating)
   print("Training ratio    : " .. torchdata.train["U"].info.noRating / (torchdata.test["U"].info.noRating + torchdata.train["U"].info.noRating))
   print("")

local train   = torchdata.train["U"].data
local test    = torchdata.test ["U"].data


--create split file
local splitFile = io.open(params.outSplit, "w")
local arfFile   = io.open(params.outArff, "w")

arfFile:write("@DATA", "\n")

for idUser, oneTrain in pairs(train) do
   
   xlua.progress(idUser, #train)
   
   local data = oneTrain
   
   local oneTest = test[idUser]
   
   --write split file
   if oneTest then
    for k = 1, oneTest:size(1) do
        local idItem = oneTest[k][1]
        splitFile:write(idUser .. "\t" .. idItem, "\n")
    end
    
      --recreate full dataset
      data = data:cat(oneTest, 1):ssortByIndex()
   end
   
   local line = "{0 " .. idUser
   for k = 1, data:size(1) do
      local idItem = data[k][1]
      local rating = data[k][2]
      line = line .. ", " .. idItem .. " " .. rating 
   end
   line = line .. "}"
   
   arfFile:write(line, "\n")
   
end



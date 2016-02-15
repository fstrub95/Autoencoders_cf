-- Load global libraries
require("nn")
require("optim")
require("xlua")

torch.setdefaulttensortype('torch.FloatTensor')

require("nnsparse")

dofile("../AlgoTools.lua")
dofile("../tools.lua")



cmd = torch.CmdLine()
cmd:text()
cmd:text('Turn Torch input file into SVDFeature input file')
cmd:text('WARNING : Beware that SVDFeature use a file format requiring a lot of memory. For instance, the Douban dataset requires 50Gb of memory!')
cmd:text()
cmd:text('Options')
-- general options:
cmd:option('-file'           , 'movieLens-10M.t7'      ,  'The relative path to your data file (torch format)')
cmd:option('-out'            , 'movieLens-10M.svd.csv' ,  'The relative path to output file SVDFeature')
cmd:text()

local params = cmd:parse(arg)


print("Options: ")
for key, val in pairs(params) do
   print(" - " .. key  .. "  \t : " .. tostring(val))
end


--Load data
print("loading data...")
local data = torch.load(params.file)

print(data.train.U.info.size .. " Users loaded")
print(data.train.V.info.size .. " Items loaded")
print("No Train rating : " .. data.train.U.info.noRating)
print("No Test  rating : " .. data.test.U.info.noRating)

local train   = data.train["U"].data
local test    = data.test ["U"].data

local allInfoU    = data.train["U"].info
local allInfoV    = data.train["V"].info


-- This function retrieve side information from data.train.info and turn it into a string
-- String are stored to  
local function getInfo(oneInfo, size)

   if oneInfo and oneInfo.full and oneInfo.line == nil then
   
      local newInfo = oneInfo.full:sparsify(0, size) 

      -- Encode side information for the user/item i in SVDFeature format
      local oneLine = "" 
      local dim     = 0
      if newInfo:nDimension() > 0 then
         dim = newInfo:size(1)
         for k = 1, newInfo:size(1) do
            oneLine = oneLine .. " " .. newInfo[k][1] .. ":" .. newInfo[k][2]
         end
      end
      oneInfo.line = oneLine
      oneInfo.dim  = dim

   end
   
   return oneInfo or { line = "", dim = 0 }
end



local function computeFile(samples, path)

   local f = io.open(path, "w")
  
   local line = ""
   local uLine = ""
   local vLine = ""

   for i, oneU in pairs(samples) do
   
      xlua.progress(i, #samples)
   
      for k = 1, oneU:size(1) do
         local j = oneU[k][1]
         local r = oneU[k][2]
        
         local uDim = 1
         local vDim = 1
      
         -- Provide the rating for user i and item j
         uLine = i .. ":1"
         vLine = j .. ":1"


         local uInfo = getInfo(allInfoU[i], data.train.U.size)
         uDim  = uDim + uInfo.dim   
         uLine = uLine .. uInfo.line

         local vInfo = getInfo(allInfoV[j], data.train.V.size)
         vDim  = vDim + vInfo.dim   
         vLine = vLine .. vInfo.line


         -- format : rating \t #globalInfo \t #uInfo \t #vInfo \t gIngo(Ex 23:1) uInfo vInfo
         line = r .. "\t0\t" .. uDim .. "\t" .. vDim .. "\t" .. uLine .. " " .. vLine
         f:write(line, "\n")
      end
   end
  
   f:close()

end

computeFile(train, params.out .. ".train")
computeFile(test , params.out .. ".test")

-- Load global libraries
require("nn")
require("optim")
require("xlua")

torch.setdefaulttensortype('torch.FloatTensor')

require("nnsparse")

dofile("AlgoTools.lua")

cmd = torch.CmdLine()
cmd:text()
cmd:text('Learn SDAE network for collaborative filtering')
cmd:text()
cmd:text('Options')
-- general options:
cmd:option('-file'           , 'movieLens-10M.t7'    ,  'The relative path to your data file (torch format)')
cmd:option('-out'            , 'movieLens-10M.svd.csv'    ,  'The relative path to output file SVDFeature')
cmd:text()

local params = cmd:parse(arg)

print("Options: ")
for key, val in pairs(params) do
   print(" - " .. key  .. "  \t : " .. tostring(val))
end


--Load data
print("loading data...")
local data = torch.load(params.file)

print(data.train.U.size .. " Users loaded")
print(data.train.V.size .. " Items loaded")
print("No Train rating : " .. data.train.U.noRating)
print("No Test  rating : " .. data.test.U.noRating)

local train   = data.train["U"].data
local test    = data.test ["U"].data

local allInfoU    = data.train["U"].info
local allInfoV    = data.train["V"].info


function computeFile(samples, path)

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
        
         uLine = i .. ":1"
         vLine = j .. ":1"

         if allInfoU[i] and allInfoU[i].fullSparse then

            if allInfoU[i].line == nil then
               local infoU = allInfoU[i].full:sparsify(0, data.train.U.size) 

               local oneLine = "" 
               local dim     = 0
               if infoU:nDimension() > 0 then
                  dim = infoU:size(1)
                  for k=1, infoU:size(1) do
                     oneLine = oneLine .. " " .. infoU[k][1] .. ":" .. infoU[k][2]
                  end
               end

               allInfoU[i].line = oneLine
               allInfoU[i].dim  = dim
            end
            
            uDim  = uDim + allInfoU[i].dim   
            uLine = uLine .. " " .. allInfoU[i].line

         end


         if allInfoV[j] and allInfoV[j].fullSparse then

            if allInfoV[j].line == nil then
               local infoV = allInfoV[j].full:sparsify(0, data.train.V.size) 

               local oneLine = "" 
               local dim     = 0
               if infoV:nDimension() > 0 then
                  dim = infoV:size(1)
                  for k=1, infoV:size(1) do
                     oneLine = oneLine .. " " .. infoV[k][1] .. ":" .. infoV[k][2]
                  end
               end

               allInfoV[j].line = oneLine
               allInfoV[j].dim  = dim
            end
            
            vDim  = vDim      +     allInfoV[j].dim   
            vLine = vLine .. " " .. allInfoV[j].line

         end

         line = r .. "\t0\t" .. uDim .. "\t" .. vDim .. "\t" .. uLine .. " " .. vLine
         f:write(line, "\n")
      end
   end
  
   f:close()

end

computeFile(train, params.out .. ".train")
computeFile(test , params.out .. ".test")

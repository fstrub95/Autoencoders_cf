-- Load global libraries
require("nn")
require("optim")
require("xlua") 

torch.setdefaulttensortype('torch.FloatTensor') 

require("nnsparse")

dofile("tools/CFNTools.lua")
dofile("tools/Appender.lua")
dofile("misc/Preload.lua")

----------------------------------------------------------------------
-- parse command-line options
--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Compute final metrics for network')
cmd:text()
cmd:text('Options')
-- general options:
cmd:option('-file'              , ''  ,  'The relative path to your data file (torch format)')
cmd:option('-network'           , ""  , 'The relative path to the lua configuration file')
cmd:option('-type'              , ""  , 'The network type U/V')
cmd:option('-gpu'               , 1   , 'use gpu')
cmd:option('-ratioStep'         , 0.2   , 'use gpu')
cmd:text()



--the following code was not clean... sorry for that!



local ratioStep = 0.2
local params = cmd:parse(arg)

print("Options: ")
for key, val in pairs(params) do
  print(" - " .. key  .. "  \t : " .. tostring(val))
end

local ratioStep = params.ratioStep

--Load data
print("Loading data...")
local train, test, info = LoadData(params.file, params)


-- start evaluating
print("Loading network...")
local network = torch.load(params.network)
network:evaluate()
print(network)


--look for appenderIn
local appenderIn
for k = 1, network:size() do
  local layer = network:get(k)
  if torch.type(layer) == "cfn.AppenderOut" then
    appenderIn = layer.appenderIn
    print("AppenderIn found")
    break
  end
end






--Sort samples by their number of ratings 
local noRatings = nnsparse.DynamicSparseTensor(10000)
local size = 0
for k, oneTrain in pairs(train) do
  size = size + 1
  noRatings:append(torch.Tensor{k, oneTrain:size(1)})
end
noRatings = noRatings:build():ssort()
local sortedIndex = noRatings[{{},1}]


-- compute the number of valid training samples
local ignore = 0
for kk = 1, size do
   if train[sortedIndex[kk]] == nil then ignore = ignore + 1 end
end



-- Configure prediction error
local rmseFct = nnsparse.SparseCriterion(nn.MSECriterion())
local maeFct  = nnsparse.SparseCriterion(nn.AbsCriterion())

rmseFct.sizeAverage = false
maeFct.sizeAverage  = false

local batchSize = 20 
local curRatio  = ratioStep
local rmse, mae = 0,0





--this method compute the error with the sparse matrix 
local transposeError = {}
function computeTranspose(outputs, targets, reverseIndex)
   for cursor, oneTarget in pairs(targets) do
       local i = reverseIndex[cursor]
   
       for k = 1, oneTarget:size(1) do
   
         local j = oneTarget[k][1]
   
         local y = outputs[cursor][j]
         local t = oneTarget[k][2]
   
         local mse = ( y - t )^2
   
         local transpose = transposeError[j] or nnsparse.DynamicSparseTensor(500)
         transpose:append(torch.Tensor{i, mse})
         transposeError[j] = transpose
   
       end
     end
end

function computeTranposeRatio(transposeError)

   --Sort samples by number of ratings
   local noRatings = nnsparse.DynamicSparseTensor(10000)
   local size  = 0
   for k, oneTranspose in pairs(transposeError) do
     transposeError[k] = oneTranspose:build():ssortByIndex()
     oneTranspose  = transposeError[k]
     size = size + 1
     noRatings:append(torch.Tensor{k, oneTranspose:size(1)})
   end
   
   noRatings = noRatings:build():ssort()
   local index = noRatings[{{},1}]
   
   
   local ignore = 0
   for kk = 1, size do
      if transposeError[index[kk]] == nil then ignore = ignore + 1 end
   end
   
   print("TRANSPOSE !!!")
   
   local curRatio = ratioStep
   local rmse   = 0
   local rmseInterval = 0
   local noSample = 0
   local noSampleInterval = 0
   
   for kk = 1, index:size(1) do
      local k    = index[kk]
      local data = transposeError[k][{{}, 2}]
      
      noSample         = noSample         + data:size(1)
      noSampleInterval = noSampleInterval + data:size(1)
      
      rmse         = rmse         + data:sum()
      rmseInterval = rmseInterval + data:sum()
      
      if kk >= curRatio * (size-ignore) then
           local curRmse = math.sqrt(rmse/noSample)*2
           rmseInterval  = math.sqrt(rmseInterval/noSampleInterval)*2
           print( kk .."/" ..  (size-ignore)  .. "\t ratio [".. curRatio .."] : " .. curRmse .. "\t Interval [".. (curRatio - ratioStep) .. "-".. curRatio .. "]: " .. rmseInterval)
           curRatio = curRatio + ratioStep 
           rmseInterval = 0
           noSampleInterval = 0
      end 
      
   end
   
   rmse = math.sqrt(rmse/noSample) * 2 
   
   print("Final RMSE: " .. rmse)

end





local i = 1
local noSample = 0
local rmseInterval = 0
local noSampleInterval = 0

--Prepare minibatch
local inputs  = {}
local targets = {}

-- prepare meta-data
local denseMetadata  = train[1].new(batchSize, info.metaDim or 0)
local sparseMetadata = {}


------------MAIN!!!
local reverseIndex = {}

--for k, input in pairs(train) do
for kk = 1, size do
  local k = sortedIndex[kk]

  -- Focus on the prediction aspect
  local input  = train[k]
  local target = test[k]

  -- Ignore data with no testing examples
  if target ~= nil then

    -- keep the original index
    reverseIndex[i] = k

    
    inputs[i]  = input

    targets[i] = targets[i] or target.new()
    targets[i]:resizeAs(target):copy(target)

    -- center the target values
    targets[i][{{}, 2}]:add(-info[k].mean)

    if appenderIn then
      denseMetadata[i]  = info[k].full
      sparseMetadata[i] = info[k].fullSparse
    end
    
    noSample         = noSample         + target:size(1)
    noSampleInterval = noSampleInterval + target:size(1)
    i = i + 1



    --compute loss when minibatch is ready
    if #inputs == batchSize then

      --Prepare metadata
      if appenderIn then
        appenderIn:prepareInput(denseMetadata, sparseMetadata)
      end

      local outputs = network:forward(inputs)

      -- compute MAE
      mae = mae + maeFct:forward(outputs, targets)

      --compute RMSE
      local rmseCur = rmseFct:forward(outputs, targets)
      rmse         = rmse        +  rmseCur
      rmseInterval = rmseInterval + rmseCur

      --reset minibatch
      inputs = {}
      i = 1
      
      -- if the ratio
      if kk >= curRatio * (size-ignore) then
      
        local curRmse = math.sqrt(rmse/noSample)*2
        rmseInterval  = math.sqrt(rmseInterval/noSampleInterval)*2
        
        print( kk .."/" ..  (size-ignore)  .. "\t ratio [".. curRatio .."] : " .. curRmse .. "\t Interval [".. (curRatio - ratioStep) .. "-".. curRatio .. "]: " .. rmseInterval)
        
        -- increment next ratio
        curRatio = curRatio + ratioStep
        
        -- reset interval
        rmseInterval     = 0
        noSampleInterval = 0
      end
      
      
      computeTranspose(outputs, targets, reverseIndex)
      reverseIndex = {}

    end

  end
end

-- remaining data for minibatch
if #inputs > 0 then
  local _targets = {unpack(targets, 1, #inputs)} --retrieve a subset of targets

  if appenderIn then
    local _sparseMetadata = {unpack(sparseMetadata, 1, #inputs)}
    local _denseMetadata =  denseMetadata[{{1, #inputs},{}}]

    appenderIn:prepareInput(_denseMetadata, _sparseMetadata)
  end

  local outputs = network:forward(inputs)

  mae  = mae  + maeFct:forward(outputs , _targets)


  local rmseCur = rmseFct:forward(outputs, _targets)
  rmse         = rmse        +  rmseCur
  rmseInterval = rmseInterval + rmseCur

  local curRmse  = math.sqrt(rmse/noSample )*2
  rmseInterval   = math.sqrt(rmseInterval/noSampleInterval)*2

  computeTranspose(outputs, _targets, reverseIndex)
  
  
end



rmse = math.sqrt(rmse/noSample) * 2 
mae  = mae/noSample * 2

print( (size-ignore) .."/" ..  (size-ignore)  .. "\t ratio [1.0] : " .. rmse .. "\t Interval [0.8-1.0]: " .. rmseInterval)

print("Final RMSE: " .. rmse)
print("Final MAE : " .. mae)



computeTranposeRatio(transposeError)




      

 

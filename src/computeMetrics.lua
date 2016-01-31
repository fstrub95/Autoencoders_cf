-- Load global libraries
require("nn")
require("optim")
require("xlua") 

torch.setdefaulttensortype('torch.FloatTensor') 

require("nnsparse")

dofile("AlgoTools.lua")

dofile("AutoEncoderTrainer.lua")
dofile("SDAECriterionGPU.lua")
dofile("LearnU.lua")
dofile("Appender.lua")


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
cmd:text()




local params = cmd:parse(arg)

print("Options: ")
for key, val in pairs(params) do
  print(" - " .. key  .. "  \t : " .. tostring(val))
end


--Load data
print("Loading data...")
local data = torch.load(params.file) 
local train = data.train
local test  = data.test

print(train.U.size .. " Users loaded")
print(train.V.size .. " Items loaded")
print("No Train rating : " .. train.U.noRating)
print("No Test  rating : " .. test.U.noRating)


SHOW_PROGRESS = true
USE_GPU        = params.gpu > 0


if USE_GPU then
  print("Loading cunn...")
  require("cunn")

  cutorch.setDevice(params.gpu)
  
  print("Loading data to GPU...")
  local type = params.type
  local _train = train[type]
  local _test  = test [type]

  for k, _ in pairs(train[type].data) do

    _train.data[k] = _train.data[k]:cuda()

    if _train.info.metaDim then
      _train.info[k].full       = _train.info[k].full:cuda()
      _train.info[k].fullSparse = _train.info[k].fullSparse:cuda()
    end
  end

  for k, _ in pairs(test[type].data) do

    _test .data[k] = _test .data[k]:cuda()

    if _train.info.metaDim then
      _train.info[k].full       = _train.info[k].full:cuda()
      _train.info[k].fullSparse = _train.info[k].fullSparse:cuda()
    end
  end


end




print("Loading network...")
local network = torch.load(params.network)
local train   = data.train[params.type].data
local test    = data.test [params.type].data
local info    = data.train[params.type].info


-- start evaluating
network:evaluate()


--look for appenderIn
local appenderIn
for k = 1, network:size() do
  local layer = network:get(k)
  if torch.type(layer) == "nnsparse.AppenderOut" then
    appenderIn = layer.appenderIn
    print("AppenderIn found")
    break
  end
end


inf = 1/0




-- Configure prediction error
local rmseFct = nnsparse.SparseCriterion(nn.MSECriterion())
local maeFct  = nnsparse.SparseCriterion(nn.AbsCriterion())

rmseFct.sizeAverage = false
maeFct.sizeAverage  = false

local batchSize = 3
local curRatio = 0.1
local rmse, mae = 0,0

--Prepare minibatch
local inputs  = {}
local targets = {}

-- prepare meta-data
local denseMetadata  = train[1].new(batchSize, info.metaDim or 0)
local sparseMetadata = {}


local i = 1
local noSample = 0

--Prepare RMSE interval
local noRatings = nnsparse.DynamicSparseTensor(10000)
local size = 0
for k, oneTrain in pairs(train) do
  size = size + 1
  noRatings:append(torch.Tensor{k, oneTrain:size(1)})
end
noRatings = noRatings:build():ssort()
local index = noRatings[{{},1}]

local rmseInterval = 0
local noSampleInterval = 0

local transposeY = {}

local ignore = 0
for kk = 1, size do
   if train[index[kk]] == nil then ignore = ignore + 1 end
end


local reverseIndex = {}


------------MAIN!!!

--for k, input in pairs(train) do
for kk = 1, size do
  local k = index[kk]

  -- Focus on the prediction aspect
  local input = train[k]
  local target = test[k]

  -- Ignore data with no testing examples
  if target ~= nil then

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
      --mae = mae + maeFct:forward(outputs, targets)

      --compute RMSE
      local rmseCur = rmseFct:forward(outputs, targets)
      rmse         = rmse        +  rmseCur
      rmseInterval = rmseInterval + rmseCur

      --reset minibatch
      inputs = {}
      i = 1
      if kk >= curRatio * (size-ignore) then
        local curRmse = math.sqrt(rmse/noSample)*2
        rmseInterval  = math.sqrt(rmseInterval/noSampleInterval)*2
        print( kk .."/" ..  (size-ignore)  .. "\t ratio [".. curRatio .."] : " .. curRmse .. "\t Interval [".. (curRatio - 0.1) .. "-".. curRatio .. "]: " .. rmseInterval)
        curRatio = curRatio + 0.1
        rmseInterval = 0
        noSampleInterval = 0
      end

      for cursor, oneTarget in pairs(targets) do
         local i = reverseIndex[cursor]

         for k = 1, oneTarget:size(1) do

            local j = oneTarget[k][1]
         
            local y = outputs[cursor][j]
            local t = oneTarget[k][2]
            
            local mse = ( y - t )^2

            local transpose = transposeY[j] or nnsparse.DynamicSparseTensor(500)
            transpose:append(torch.Tensor{i, mse})
            transposeY[j] = transpose

         end
      end
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

  --mae  = mae  + maeFct:forward(outputs , _targets)


  local rmseCur = rmseFct:forward(outputs, _targets)
  rmse         = rmse        +  rmseCur
  rmseInterval = rmseInterval + rmseCur

  local curRmse  = math.sqrt(rmse/noSample )*2
  rmseInterval   = math.sqrt(rmseInterval/noSampleInterval)*2
  print( size .."/" ..  (size-ignore)  .. "\t ratio [1.0] : " .. curRmse .. "\t Interval [0.9-1.0]: " .. rmseInterval)
  
  
  for cursor, oneTarget in pairs(_targets) do
    local i = reverseIndex[cursor]

    for k = 1, oneTarget:size(1) do

      local j = oneTarget[k][1]

      local y = outputs[cursor][j]
      local t = oneTarget[k][2]

      local mse = ( y - t )^2

      local transpose = transposeY[j] or nnsparse.DynamicSparseTensor(500)
      transpose:append(torch.Tensor{i, mse})
      transposeY[j] = transpose

    end
  end
end


--Prepare RMSE interval
local noRatings = nnsparse.DynamicSparseTensor(10000)
local size  = 0
for k, oneTranspose in pairs(transposeY) do
  transposeY[k] = oneTranspose:build():ssortByIndex()
  oneTranspose  = transposeY[k]
  size = size + 1
  noRatings:append(torch.Tensor{k, oneTranspose:size(1)})
end

noRatings = noRatings:build():ssort()
local index = noRatings[{{},1}]


--local ignore = 0
--for kk = 1, size do
--   if transposeY[index[kk]] == nil then ignore = ignore + 1 end
--end
print("TRANSPOSE !!!")

local curRatio = 0.1
local rmse   = 0
local rmseInterval = 0
local noSample = 0
local noSampleInterval = 0

for kk = 1, index:size(1) do
   local k    = index[kk]
   local data = transposeY[k][{{}, 2}]
   
   noSample         = noSample         + data:size(1)
   noSampleInterval = noSampleInterval + data:size(1)
   
   rmse         = rmse         + data:sum()
   rmseInterval = rmseInterval + data:sum()
   
   if kk >= curRatio * size then
        local curRmse = math.sqrt(rmse/noSample)*2
        rmseInterval  = math.sqrt(rmseInterval/noSampleInterval)*2
        print( kk .."/" ..  size  .. "\t ratio [".. curRatio .."] : " .. curRmse .. "\t Interval [".. (curRatio - 0.1) .. "-".. curRatio .. "]: " .. rmseInterval)
        curRatio = curRatio + 0.1
        rmseInterval = 0
        noSampleInterval = 0
   end 
   
end






rmse = math.sqrt(rmse/noSample) * 2 
mae  = mae/noSample * 2
      
print("Final RMSE: " .. rmse)
print("Final MAE : " .. mae)
      
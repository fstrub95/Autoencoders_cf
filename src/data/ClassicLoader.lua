local classicLoader, parent = torch.class('cfn.classicLoader', 'cfn.DataLoader')

function classicLoader:LoadRatings(conf)

   --no pre-process/post-processing
   function preprocess(x)  return x end
   function postprocess(x) return x end

   local i = 0

   local function appendRatings(fileName, appendFct)
      local ratesfile = io.open(fileName, "r")
      for line in ratesfile:lines() do
   
         local userIdStr, movieIdStr, ratingStr = line:match('(%d+) (%d+) (%-?%d%.?%d*)')
   
         local userId  = tonumber(userIdStr)
         local itemId  = tonumber(movieIdStr)
         local rating  = tonumber(ratingStr)
   
         -- use hash table to have continuous indices
         local itemIndex = self:getItemIndex(itemId)
         local userIndex = self:getUserIndex(userId)
   
         rating = preprocess(rating)
   
         appendFct(self, userIndex, itemIndex, rating)
         
         i = i + 1
         if math.fmod(i, 100000) == 0 then
            print(i .. " ratings loaded...")
         end
         
      end
      ratesfile:close()
   end

   appendRatings(conf.ratings .. ".train", self.appendTrain)
   appendRatings(conf.ratings .. ".test" , self.appendTest)
   
end

-- Lua does not support advance regex such as "(foo)+"... we have to compute the exact regex
local __regexTable = {}
local function getRegex(noElem)
      assert(noElem > 0)
      local regex = __regexTable[noElem]
      if regex == nil then
         regex = "(%d:%-?%d%.?%d*)" --regex for -2.126 --> no standard regex for numbers in lua :(
         for k = 1, noElem-1 do regex = regex .. " (%d:%-?%d%.?%d*)" end
         __regexTable[noElem] = regex 
      end

     return regex
end

local function densify(sparseTensor, size)
   local denseTensor = torch.zeros(size)
   local index = sparseTensor[{{},1}]
   local data  = sparseTensor[{{},2}]
    denseTensor:indexCopy(1, index:long(), data)
end

function classicLoader:LoadMeta(fileName, type, GetXindex) 

   local maxId  = 0
   local offset = self.train[type].info.size

   local file = io.open(fileName, "r")
   for line in file:lines() do

      -- parse side information
      local objIdStr, noInfoStr, allInfoStr = line:match('(%d+) (%d+) ?(.*)')
      local objId  = tonumber(objIdStr)
      local noInfo = tonumber(noInfoStr)

      if noInfo > 0 then
         --split side information into a table
         local regex = getRegex(noInfo)
         local infoTable = { allInfoStr:match(regex) }
   
         --append side inforamtion in a sparse Tensor
         local sparseTensor = nnsparse.DynamicSparseTensor.new(200)
         for _, infoStr in pairs(infoTable) do
            local idInfoStr, valInfoStr = infoStr:match("(%d+):(%-?%d%.?%d*)")
            
            assert(idInfoStr, "Fail to parse the following line: " .. allInfoStr)
   
            local idInfo  = tonumber(idInfoStr)
            local valInfo = tonumber(valInfoStr)
   
            maxId = math.max(maxId, idInfo)
   
            sparseTensor:append(torch.Tensor{idInfo + offset, valInfo})
         end
   
   
         -- retrieve the info table
         local objIndex = GetXindex(self,objId)
         local info     = self.train[type].info[objIndex] or {}
   
         -- fill it
         info.fullSparse = sparseTensor:build():ssortByIndex() 
   
         -- update the info
         self.train[type].info[objIndex] = info
      end
   end
   file:close()


   --compute dense side information by using the maximum id of side information
   for _, oneInfo in pairs(self.train[type].info) do
      if oneInfo.fullSparse then
         oneInfo.full = densify(oneInfo.fullSparse,maxId)
      else
         oneInfo.full       = torch.zeros(maxId)
         oneInfo.fullSparse = torch.Tensor()
      end
   end 

   self.train[type].info.metaDim = maxId

end

function classicLoader:LoadMetaU(conf) 
   if #conf.metaUser > 0 then
      self:LoadMeta(conf.metaUser, "U", self.getUserIndex) 
   end
end

function classicLoader:LoadMetaU(conf) 
   if #conf.metaItem > 0 then
      self:LoadMeta(conf.metaItem, "V", self.getItemIndex) 
   end
end


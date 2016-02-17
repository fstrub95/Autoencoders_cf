cfn = cfn or {}

local DataLoader = torch.class('cfn.DataLoader')
--This class has a poor design but it does the work. (And Lua is not helping to prototype nice class)
--WARNING : NVI idiom (cf C++)


-- Public Interface to implement
function DataLoader:LoadRatings(conf) end
function DataLoader:LoadMetaU  (conf) end
function DataLoader:LoadMetaV  (conf) end


-- Public Interface to call
function DataLoader:LoadData(ratioTraining, conf) 

   -- First initlialize clean storage
   self:__reset()

   self.__ratioTraining = ratioTraining

   print("Step 1 : Loading ratings...")
   self:LoadRatings(conf)

   print("Step 2 : PostProcessig ratings...")
   self:__PostProcessRating()

   --Load MetaData
   print("Step 3 : Load user metadata...")   
   self:LoadMetaU(conf)

   print("Step 4 : Load item metadata...")
   self:LoadMetaV(conf)
   
   print("Step 5 : Saving data in torch format...")
   local data = {train = self.train, test = self.test}
   torch.save(conf.out, data)

   print("Done...")
   return self.train, self.test

end


-- Protected Method (helper)
function DataLoader:AppendOneRating(userId, itemId, rating)
   
   --store the rating in either the training or testing set
   if torch.uniform() < self.__ratioTraining then
      self:appendTrain(userId, itemId, rating)
   else
      self:appendTest(userId, itemId, rating)
   end
   
   if math.fmod(self.__noRating, 100000) == 0 then
         print(self.__noRating .. " ratings loaded...")
   end
end

-- Protected Method (helper)
function DataLoader:appendTrain(userId, itemId, rating)

      -- bufferize sparse tensors
      if self.train.U.data[userId] == nil then self.train.U.data[userId] = nnsparse.DynamicSparseTensor(200) end
      if self.train.V.data[itemId] == nil then self.train.V.data[itemId] = nnsparse.DynamicSparseTensor(200) end 

      self.train.U.data[userId]:append(torch.Tensor{itemId,rating})
      self.train.V.data[itemId]:append(torch.Tensor{userId,rating})

      --update the training mean
      self.__noRating = self.__noRating + 1
      self.__n    =  self.__n + 1
      self.__mean = (self.__n*self.__mean + rating) / ( self.__n + 1 )
      
      --store the matrix size by keeping the max Id
      self.__Usize = math.max(self.__Usize, userId)
      self.__Vsize = math.max(self.__Vsize, itemId)
end

-- Protected Method (helper)
function DataLoader:appendTest(userId, itemId, rating)

      -- bufferize sparse tensors
      if self.test.U.data[userId] == nil then self.test.U.data[userId] = nnsparse.DynamicSparseTensor.new(200) end
      if self.test.V.data[itemId] == nil then self.test.V.data[itemId] = nnsparse.DynamicSparseTensor.new(200) end 

      self.test.U.data[userId]:append(torch.Tensor{itemId,rating})
      self.test.V.data[itemId]:append(torch.Tensor{userId,rating})
      
      self.__noRating = self.__noRating + 1

      --store the matrix size by keeping the max Id
      self.__Usize = math.max(self.__Usize, userId)
      self.__Vsize = math.max(self.__Vsize, itemId)
end


--private method
function DataLoader:__reset() 
   self.train = {U = { data = {}, info = {} }, V = { data = {}, info = {}}}
   self.test  = {U = { data = {}, info = {} }, V = { data = {}, info = {}}}
   
   self.__Usize = 0
   self.__Vsize = 0
   self.__mean  = 0
   self.__n     = 0
   
   self.__noRating = 0
end


function DataLoader:__PostProcessRating()

   -- sort sparse vectors (This is required to make nn.SparseLinear works)
   local function build(X) 
      for k, x in pairs(X.data) do 
         X.data[k] = torch.Tensor.ssortByIndex(x:build())
      end 
   end
   
   build(self.train.U)
   build(self.train.V)
   build(self.test.U)
   build(self.test.V)


   --store mean, globalMean and std for every row/column
   local function computeBias(X,gMean)
      for k, x in pairs(X.data) do
         X.info[k] = X.info[k] or {}
         X.info[k].mean  = x[{{},2}]:mean()
         X.info[k].std   = x[{{},2}]:std()
         X.info[k].gMean = gMean
      end
   end
   
   computeBias(self.train.U, self.__mean)
   computeBias(self.train.V, self.__mean)

   --Provide external information
   self.train.U.info.size, self.test.U.info.size = self.__Usize, self.__Usize
   self.train.V.info.size, self.test.V.info.size = self.__Vsize, self.__Vsize

   self.train.U.info.dimension, self.test.U.info.dimension = self.__Vsize, self.__Vsize
   self.train.V.info.dimension, self.test.V.info.dimension = self.__Usize, self.__Usize
   
   self.train.U.info.noRating, self.test.U.info.noRating = self.__n, self.__noRating - self.__n
   self.train.V.info.noRating, self.test.V.info.noRating = self.__n, self.__noRating - self.__n
   
   print(self.__Usize .. " users were loaded.")
   print(self.__Vsize .. " items were loaded.")

end


function DataLoader:getUserIndex(userId)
   self.userHash    =  self.userHash    or {}
   self.userCounter =  self.userCounter or 1

   local userIndex = self.userHash[userId]
   if userIndex == nil then
      self.userHash[userId] = self.userCounter
      userIndex             = self.userCounter
      self.userCounter      = self.userCounter + 1
   end

   return userIndex
end

function DataLoader:getItemIndex(itemId)
   self.itemHash    = self.itemHash    or {}
   self.itemCounter = self.itemCounter or 1

   local itemIndex = self.itemHash[itemId]
   if itemIndex == nil then
      self.itemHash[itemId] = self.itemCounter
      itemIndex             = self.itemCounter
      self.itemCounter      = self.itemCounter + 1
   end

   return itemIndex
end



   

   



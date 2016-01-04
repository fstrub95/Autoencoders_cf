
require("sys")
require("torch")

dofile ("tools.lua")

require("nnsparse")

local function computeTestAndTrain(userRating, ratioTraining)

   local train = {U = { data = {}, info = {} }, V = { data = {}, info = {}}}
   local test  = {U = { data = {}, info = {} }, V = { data = {}, info = {}}} 

   local mean = 0
   local n    = 0

   local Usize = 0
   local Vsize = 0
   

   for userId, ratings in pairs(userRating) do

      xlua.progress(userId, #userRating)

      for _, oneRating in pairs(ratings) do

         local itemId = oneRating.itemId
         local rating = oneRating.rating
   
         --store the matrix size by keeping the max Id
         Usize = math.max(Usize, userId)
         Vsize = math.max(Vsize, itemId)
   
         --store the rating in either the training or testing set
         if math.random() < ratioTraining then
         
            if train.U.data[userId] == nil then train.U.data[userId] = nnsparse.DynamicSparseTensor(200) end
            if train.V.data[itemId] == nil then train.V.data[itemId] = nnsparse.DynamicSparseTensor(200) end 
            
            train.U.data[userId]:append(torch.Tensor{itemId,rating})
            train.V.data[itemId]:append(torch.Tensor{userId,rating})
            
            n    = n + 1
            mean = (n*mean + rating) / ( n + 1 )
   
         else
            if test.U.data[userId] == nil then test.U.data[userId] = nnsparse.DynamicSparseTensor.new(200) end
            if test.V.data[itemId] == nil then test.V.data[itemId] = nnsparse.DynamicSparseTensor.new(200) end 
            
            test.U.data[userId]:append(torch.Tensor{itemId,rating})
            test.V.data[itemId]:append(torch.Tensor{userId,rating})
         end
      end
   end


   -- sort sparse vectors (This is required)
   local function build(X) 
      for k, x in pairs(X.data) do 
         X.data[k] = torch.Tensor.ssortByIndex(x:build())
         if USE_GPU then X.data[k] = X.data[k]:cuda() end 
      end 
   end
   
   build(train.U)
   build(train.V)
   build(test.U)
   build(test.V)


   --store mean, globalMean and std for every row/column
   local function computeBias(X,gMean)
      for k, x in pairs(X.data) do
         X.info[k] = X.info[k] or {}
         X.info[k].mean  = x[{{},2}]:mean()
         X.info[k].std   = x[{{},2}]:std()
         X.info[k].gMean = gMean
      end
   end
   
   computeBias(train.U,mean)
   computeBias(train.V,mean)

   --Provide external information
   train.U.size, test.U.size = Usize, Usize
   train.V.size, test.V.size = Vsize, Vsize

   train.U.dimension, test.U.dimension = Vsize, Vsize
   train.V.dimension, test.V.dimension = Usize, Usize
   
  
   print(Usize .. " users were loaded.")
   print(Vsize .. " items were loaded.")

   return train, test
end





local function LoadRatings(rates, ratio, regex)

   --no pre-process/post-processing
   function preprocess(x)  return (x-3)/2 end
   function postprocess(x) return 2*x+3 end


   -- step 3 : load ratings
   local ratesfile = io.open(rates, "r")

   -- Step 1 : Retrieve movies'scores...th
   print("Step 1 : Retrieve movies'scores...")
   local userRating = {}
   for line in ratesfile:lines() do
      local userIdStr, movieIdStr, ratingStr = line:match(regex)

      local userId  = tonumber(userIdStr)
      local itemId  = tonumber(movieIdStr)
      local rating  = tonumber(ratingStr)

      rating = preprocess(rating)

      if userRating[userId] == nil then 
         userRating[userId] = {} 
      end

      table.insert(userRating[userId], 
         {
            itemId = itemId, 
            rating = rating
         })

   end
   ratesfile:close()
   

   -- Step 2 : Separate training set and testing set in sparse matrix
   print("Step 2 : Separate training set and testing set in sparse matrix...")
   local train, test = computeTestAndTrain(userRating, ratio)

   return train, test
end



local function LoadBidon(file, ratio)


   local data = torch.Tensor(200,100):uniform(-1,1):apply(function(x) if torch.uniform() < 0.6 then return 0 else return x end end)

   function preprocess(x)  return (x) end
   function postprocess(x) return (x) end


   -- Step 1 : Retrieve rating by jokes
   print("Step 1 : Retrieve rating by jokes...")

   local userRating = {}

   for i = 1, data:size(1) do
      for j = 1, data:size(2) do

         local t = data[i][j]

         if t ~= 0 then

            local userId = i
            local itemId = j

            local rating =  preprocess(t)

            if userRating[userId] == nil then 
               userRating[userId] = {} 
            end

            table.insert(userRating[userId], 
                {
                  itemId = itemId, 
                  rating = rating
               })

         end  
      end
   end




   -- Step 2 : Separate training set and testing set in sparse matrix
   print("Step 2 : Separate training set and testing set in sparse matrix...")
   local train, test = computeTestAndTrain(userRating, ratio)

   return train, test

end


local function LoadJester(file, ratio)


   local file = torch.DiskFile(file, "r")
   local data = file:readObject()


   function preprocess(x)  return (x)/10 end
   function postprocess(x) return (x)*10 end


   -- Step 1 : Retrieve rating by jokes
   print("Step 1 : Retrieve rating by jokes...")

   local userRating = {}

   for i = 1, data:size(1) do
      for j = 1, data:size(2) do

         local t = data[i][j]

         if t ~= 99 then

            local userId = i
            local itemId = j

            local rating =  preprocess(t)

            if userRating[userId] == nil then 
               userRating[userId] = {} 
            end

            table.insert(userRating[userId], 
               {
                  itemId = itemId, 
                  rating = rating
               })

         end  
      end
   end




   -- Step 2 : Separate training set and testing set in sparse matrix
   print("Step 2 : Separate training set and testing set in sparse matrix...")
   local train, test = computeTestAndTrain(userRating, ratio)

   return train, test

end





function LoadData(conf)

   if     conf.type == "movieLens" then
      return LoadRatings(conf.file, conf.ratio, '(%d+)::(%d+)::(%d+)::(%d+)')
   elseif conf.type == "jester" then
      return LoadJester(conf.file, conf.ratio)
   elseif conf.type == "classic" then
      return LoadRatings(conf.file, conf.ratio, '(%d+) (%d+) (%d+)')
   elseif conf.type == "bidon" then
      return LoadBidon(conf.file, conf.ratio, '(%d+) (%d+) (%d+)')
   else
      error("Unknown data format, it must be :  movieLens / jester / none ")
   end

end



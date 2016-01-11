
require("sys")
require("torch")

dofile ("tools.lua")

require("nnsparse")







--This class has very poor design but it does the work. (And Lua is not helping to prototype nice class)
--WARNING : NVI idiom (cf C+++)

local DataLoader = torch.class('DataLoader')


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
   
   print("Done...")

   return self.train, self.test

end



-- Protected Method (helper)
function DataLoader:AppendOneRating(userId, itemId, rating)

   --store the matrix size by keeping the max Id
   self.__Usize = math.max(self.__Usize, userId)
   self.__Vsize = math.max(self.__Vsize, itemId)


   --store the rating in either the training or testing set
   if math.random() < self.__ratioTraining then

      if self.train.U.data[userId] == nil then self.train.U.data[userId] = nnsparse.DynamicSparseTensor(200) end
      if self.train.V.data[itemId] == nil then self.train.V.data[itemId] = nnsparse.DynamicSparseTensor(200) end 

      self.train.U.data[userId]:append(torch.Tensor{itemId,rating})
      self.train.V.data[itemId]:append(torch.Tensor{userId,rating})

      --update the training mean
      self.__n    =  self.__n + 1
      self.__mean = (self.__n*self.__mean + rating) / ( self.__n + 1 )

   else
      if self.test.U.data[userId] == nil then self.test.U.data[userId] = nnsparse.DynamicSparseTensor.new(200) end
      if self.test.V.data[itemId] == nil then self.test.V.data[itemId] = nnsparse.DynamicSparseTensor.new(200) end 

      self.test.U.data[userId]:append(torch.Tensor{itemId,rating})
      self.test.V.data[itemId]:append(torch.Tensor{userId,rating})
   end
   
end


--private method
function DataLoader:__reset() 
   self.train = {U = { data = {}, info = {} }, V = { data = {}, info = {}}}
   self.test  = {U = { data = {}, info = {} }, V = { data = {}, info = {}}}
   
   self.__Usize = 0
   self.__Vsize = 0
   self.__mean  = 0
   self.__n     = 0
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
   self.train.U.size, self.test.U.size = self.__Usize, self.__Usize
   self.train.V.size, self.test.V.size = self.__Vsize, self.__Vsize

   self.train.U.dimension, self.test.U.dimension = self.__Vsize, self.__Vsize
   self.train.V.dimension, self.test.V.dimension = self.__Usize, self.__Usize
   
   print(self.__Usize .. " users were loaded.")
   print(self.__Vsize .. " items were loaded.")

end


----------------------------------------------------------------------------
----------------------------------------------------------------------------

local movieLensLoader, parent = torch.class('movieLensLoader', 'DataLoader')

function movieLensLoader:LoadRatings(conf)

   --no pre-process/post-processing
   function preprocess(x)  return (x-3)/2 end
   function postprocess(x) return 2*x+3 end

   -- step 3 : load ratings
   local ratesfile = io.open(conf.ratings, "r")

   -- Step 1 : Retrieve movies'scores...th
   local i = 0
   for line in ratesfile:lines() do

      local userIdStr, movieIdStr, ratingStr = line:match('(%d+)::(%d+)::(%d%.?%d?)::(%d+)')

      local userId  = tonumber(userIdStr)
      local itemId  = tonumber(movieIdStr)
      local rating  = tonumber(ratingStr)

      rating = preprocess(rating)

      self:AppendOneRating(userId, itemId, rating)

      i = i + 1
      
      if math.fmod(i, 100000) == 0 then
         print(i .. " ratings loaded...")
      end

   end
   ratesfile:close()

end


function movieLensLoader:LoadMetaU(conf) 

   if #conf.metaUser > 0 then

      local usersfile = io.open(conf.metaUser, "r")

      for line in usersfile:lines() do

         local userIdStr, sex, age, job, ZIP = line:match('(%d+)::(%a)::(%d+)::(%d+)::(%d+)')
         --local userIdStr, age, sex, job, ZIP = line:match('(%d+)|(%d+)|(%a)|(%a+)|(.-)') --ignore code zip since it is ill formated

         local userId = tonumber(userIdStr)
           
         local info = self.U.info[userId] or {}
         
         info.sex    = sexToBinary(sex)
         info.age    = ageToBinary(age)
         info.job    = jobToBinary(job)
         
         info.full   = torch.cat({info.sex, info.age, info.job})

         self.U.info[userId] = info

      end
      usersfile:close()

      self.U.info.metaDim = 2 + 7 + 21

   end

end

function movieLensLoader:LoadMetaV(conf) 


   if #conf.metaItem > 0 then

      local moviesfile = io.open(conf.metaItem, "r")

      for line in moviesfile:lines() do

         local movieIdStr, title, genre = line:match('(%d+)::(.*)::(.*)')
         --local movieIdStr, title, day, month, year, url, genreStr = line:match('(%d+)|(.*)|(%d+)-(%a+)-(%d+)||(.-)|(.*)')

         if movieIdStr ~= nil then 

            local movieId = tonumber(movieIdStr)
            
            
            local info = self.V.info[movieId] or {}

            info.title  = title
            info.genre  = genreToBinary(genre)

            info.full   = info.genre

            self.V.info[movieId] = info   

         else
            local movieIdStr = line:match('(%d+)|')
            local movieId = tonumber(movieIdStr)
            print("unable to parse movie (".. movieId .. ") : " .. line)
            self.V.info[movieId] = {}
         end
      end
      
     moviesfile:close()
     
     self.V.info.metaDim = 18

   end

end




----------------------------------------------------------------------------
----------------------------------------------------------------------------

local jesterLoader, parent = torch.class('jesterLoader', 'DataLoader')

function jesterLoader:LoadRatings(conf)

   function preprocess(x)  return (x)/10 end
   function postprocess(x) return (x)*10 end

   local file = torch.DiskFile(conf.ratings, "r")
   local data = file:readObject()

   -- Step 1 : Retrieve rating by jokes
   for i = 1, data:size(1) do
      for j = 1, data:size(2) do

         local t = data[i][j]

         if t ~= 99 then

            local userId = i
            local itemId = j
            local rating =  preprocess(t)
   
            self:AppendOneRating(userId, itemId, rating)
         end 
      end
   end

end


----------------------------------------------------------------------------
----------------------------------------------------------------------------

local dummyLoader, parent = torch.class('dummyLoader', 'DataLoader')

function dummyLoader:__init(noUsers, noItems, sparseRate)
   self.sparseRate = 0.4
   self.noUsers = 200
   self.noItems = 200
   self.sparsifier = function(x) if torch.uniform() < self.sparseRate then return 0 else return x end end
end

function dummyLoader:LoadRatings(conf)

   function preprocess(x)  return x end
   function postprocess(x) return x end

   local data = torch.Tensor(self.noUsers, self.noItems):uniform(-1, 1)
   data:apply(self.sparsifier)
  
   for i = 1, data:size(1) do
      for j = 1, data:size(2) do

         local t = data[i][j]

         if t ~= 0 then
   
            local userId = i
            local itemId = j
            local rating = preprocess(t)
   
            self:AppendOneRating(userId, itemId, rating)
         end 
      end
   end

end





   ----------------------------------------------------------------------
   -- parse command-line options
   --
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Store Data SDAE network for collaborative filtering')
   cmd:text()
   cmd:text('Options')
   -- general options:
   cmd:option('-ratings'        , '../data/movieLens/ratings-1M.dat', 'The relative path to your data file')
   cmd:option('-metaUser'       , ''                                 , 'The relative path to your metadata file for users')
   cmd:option('-metaItem'       , ''                                 , 'The relative path to your metadata file for items')
   cmd:option('-fileType'       , "movieLens"                        , 'The data file format (jester/movieLens/classic)')
   cmd:option('-out'            , "./movieLens-1M.t7"                 , 'The data file format (jester/movieLens/classic)')
   cmd:option('-ratio'          , 0.9                                , 'The training ratio')
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
   elseif params.fileType == "dummy"     then dataLoader = dummyLoader:new()
   else
      error("Unknown data format, it must be :  movieLens / jester / none ")
   end

   local train, test = dataLoader:LoadData(params.ratio,params)

   print("Saving data in torch format...")
   local data = {train = train, test = test}
   torch.save(params.out, data) 
   print('Successfuly saved : ' .. params.out)
   





--- this file provide a template to implement your parser
-- Lua is a dynamic typed language. It means that the variables are checked at run-time
-- One must be careful to follow the program syntax
-- 
-- Data Loader return this object
--    self.train = {U = { data = {}, info = {} }, V = { data = {}, info = {}}}
--    self.test  = {U = { data = {}, info = {} }, V = { data = {}, info = {}}}  
--    
--    where
--      "data'  is a sparse representation of the input
--      "info" is the side information
--      info = { id = idItem, full = torch.Tensor(), fullSparse = torch.Tensor(.,2) }
--          where "full" is dense representation of side information
--          where "fullSparse" is a sparse representation of side information
--          
--      "info[.]" will also automatically contains the training mean/std for every items/users




local templateLoader, parent = torch.class('cfn.templateLoader', 'cfn.DataLoader')

function templateLoader:LoadRatings(conf)

   -- 1 : provide a preprocess/postprocess function
   function preprocess(x)  return x end
   function postprocess(x) return x end

   -- 2 : Load your file
   local ratesfile = io.open(conf.ratings, "r")


   -- 3 : Retrieve ratings
   local i = 0
   for line in ratesfile:lines() do

      -- use some regex
      local userIdStr, movieIdStr, ratingStr = line:match('(%d+) (%d+) (%d+)')

      -- turn string into id 
      local userId  = tonumber(userIdStr)
      local itemId  = tonumber(movieIdStr)
      local rating  = tonumber(ratingStr)


      -- use the helper to have continuous indices
      local itemIndex = self:getItemIndex(itemId)
      local userIndex = self:getUserIndex(userId)

      -- do not forget to preprocess your rating 
      rating = preprocess(rating)

      -- append one rating by using the training ratio
      self:AppendOneRating(userIndex, itemIndex, rating)
      
      -- you can also directly use append train/test for benchmarking purpose 
      self:appendTrain(userIndex, itemIndex, rating)
      self:appendTest(userIndex, itemIndex, rating)

   end
   ratesfile:close()
   
end


function templateLoader:LoadMetaU(conf) 

   if #conf.metaUser > 0 then

      local usersfile = io.open(conf.metaUser, "r")

      for line in usersfile:lines() do

         -- parse your side information
         local userIdStr, ______ = line:match('(%d+)::(%a)::(%d+)::(%d+)::(%d+)')

         local userId    = tonumber(userIdStr)
         local userIndex = self:getUserIndex(userId)

         -- retrieve the info table
         local info = self.train.U.info[userIndex] or {}

         -- fill it
         info.id     = userId
         ______      = ______


         --It is very important to provide a dense and a sparse representation of your input
         -- the dense representation MUST have the same dimension for every users/items
         info.full       = torch.cat({info.sex, info.age, info.job})
         info.fullSparse = info.full:sparsify(0, self.train.U.dimension) -- the second argument is the offset

         -- update the info
         self.train.U.info[userIndex] = info

      end
      usersfile:close()

      -- This line provides the dimension of side information to update the size of network layers   
      self.train.U.info.metaDim = self.train.U.info[1].full:size(1)

   end

end


function templateLoader:LoadMetaV(conf) 

   if #conf.metaItem > 0 then

      --if you are reading a cvs files
      local csv2tensor = require 'csv2tensor'
      local itemTensor  = csv2tensor.load(conf.metaItem)

      --Warning columns are sorted by alphabetical order. We advise to use the user/item id as the first column 

      for i = 1, itemTensor:size(1) do

         -- idMovie, tag1, tag2, tag3 etc.
         local itemId = itemTensor[i][1]
         local data   = itemTensor[{i, {2, itemTensor:size(2)}}]

         local itemIndex = self.itemHash[itemId]
         
         if itemIndex ~= nil then
            local info = self.train.V.info[itemIndex] or {}

            info.data  = data
            
            --It is very important to provide a dense and a sparse representation of your input
            -- the dense representation MUST have the same dimension for every users/items
            info.full       = data
            info.fullSparse = data:sparsify(0, self.train.V.dimension) -- the second argument is the offset

            self.train.V.info[itemIndex] = info
         end

      end
      
      -- This line provides the dimension of side information to update the size of network layers  
      self.train.V.info.metaDim = self.train.V.info[1].full:size(1)
      
   end
end


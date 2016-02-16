local doubanLoader, parent = torch.class('cfn.doubanLoader', 'cfn.DataLoader')

function doubanLoader:LoadRatings(conf)

   --no pre-process/post-processing
   function preprocess(x)  return (x-3)/2 end
   function postprocess(x) return 2*x+3 end

   -- step 3 : load ratings
   local ratesfile = io.open(conf.ratings, "r")

 
   -- Step 1 : Retrieve movies'scores...th
   for line in ratesfile:lines() do

      local userIdStr, movieIdStr, ratingStr = line:match('(%d+) (%d+) (%d+)')

      local userId  = tonumber(userIdStr)
      local itemId  = tonumber(movieIdStr)
      local rating  = tonumber(ratingStr)

      local itemIndex = self:getItemIndex(itemId)
      local userIndex = self:getUserIndex(userId)

      rating = preprocess(rating)

      self:AppendOneRating(userIndex, itemIndex, rating)

   end
   ratesfile:close()

end


function doubanLoader:LoadMetaU(conf) 

   if #conf.metaUser > 0 then

      local csv2tensor = require 'csv2tensor'
      local friendTensor  = csv2tensor.load(conf.metaUser)

      for i = 1, friendTensor:size(1) do

         -- idUser, friends1, friends2, friends3 etc.
         local userId  = friendTensor[i][1]
         local friends = friendTensor[{i, {2, friendTensor:size(2)}}]

         local userIndex = self:getUserIndex(userId)

         local info = self.train.U.info[userIndex] or {}

         info.friends    = friends

         info.full       = friends
         info.fullSparse = friends:sparsify(0, self.train.U.dimension)

         self.train.U.info[userIndex] = info

      end
      self.train.U.info.metaDim = self.train.U.info[1].full:size(1)
   end
end
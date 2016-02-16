local classicLoader, parent = torch.class('cfn.classicLoader', 'cfn.DataLoader')

function classicLoader:LoadRatings(conf)

   --no pre-process/post-processing
   function preprocess(x)  return x end
   function postprocess(x) return x end

   -- step 3 : load ratings
   local ratesfile = io.open(conf.ratings, "r")

   -- Step 1 : Retrieve movies'scores...th
   for line in ratesfile:lines() do

      local userIdStr, movieIdStr, ratingStr = line:match('(%d+) (%d+) (%d+)')

      local userId  = tonumber(userIdStr)
      local itemId  = tonumber(movieIdStr)
      local rating  = tonumber(ratingStr)

      -- use hash table to have continuous indices
      local itemIndex = self:getItemIndex(itemId)
      local userIndex = self:getUserIndex(userId)

      rating = preprocess(rating)

      self:AppendOneRating(userIndex, itemIndex, rating)

   end
   ratesfile:close()
   
end
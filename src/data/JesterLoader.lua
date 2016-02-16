local jesterLoader, parent = torch.class('cfn.jesterLoader', 'cfn.DataLoader')

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
local dummyLoader, parent = torch.class('cfn.dummyLoader', 'cfn.DataLoader')

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
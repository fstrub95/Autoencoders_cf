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



local function jobToBinary(jobStr, offset)
   local jobs = torch.zeros(21)

   local jobId = tonumber(jobStr)+1
   jobs[jobId] = 1

   return jobs;
end

local function sexToBinary(sexStr, offset)
   local sex = torch.zeros(2)

   if     sexStr == 'M' then sex[1] = 1 
   elseif sexStr == 'F' then sex[2] = 1 
   end

   return sex; 

end

local function ageToBinary(ageStr, offset)
   local age = torch.zeros(7)
   
   local ageId = tonumber(ageStr)

   if     ageId == 1  then age[1] = 1 
   elseif ageId == 18 then age[2] = 1 
   elseif ageId == 25 then age[3] = 1 
   elseif ageId == 35 then age[4] = 1 
   elseif ageId == 45 then age[5] = 1 
   elseif ageId == 50 then age[6] = 1 
   elseif ageId == 56 then age[7] = 1 
   end

   return age; 

end


function movieLensLoader:LoadMetaU(conf) 

   if #conf.metaUser > 0 then

      local usersfile = io.open(conf.metaUser, "r")

      for line in usersfile:lines() do

         local userIdStr, sex, age, job, ZIP = line:match('(%d+)::(%a)::(%d+)::(%d+)::(%d+)')
         --local userIdStr, age, sex, job, ZIP = line:match('(%d+)|(%d+)|(%a)|(%a+)|(.-)') --ignore code zip since it is ill formated

         local userId = tonumber(userIdStr)
           
         local info = self.train.U.info[userId] or {}
         
         info.sex    = sexToBinary(sex)
         info.age    = ageToBinary(age)
         info.job    = jobToBinary(job)
         
         info.full       = torch.cat({info.sex, info.age, info.job})
         info.fullSparse = info.full:sparsify(0, self.train.U.dimension)   

         self.train.U.info[userId] = info

      end
      usersfile:close()

      self.train.U.info.metaDim = 2 + 7 + 21

   end

end


  local genreList = {
   "Action",
   "Adventure",
   "Animation",
   "Children's",
   "Comedy",
   "Crime",
   "Documentary",
   "Drama",
   "Fantasy", 
   "Film-Noir",
   "Horror",
   "Musical",
   "Mystery",
   "Romance",
   "Sci-Fi",
   "Thriller",
   "War",
   "Western",
 }
 
 
 local genreToIndex = {}
 for k, genre in pairs(genreList) do
   genreToIndex[genre] = k
 end



local function genreToBinary(genreStr, offset)

   local genre = torch.zeros(#genreList)

   local genreTable = string.split(genreStr, "|")

   for _,itemGenre in pairs(genreTable) do 
      genre[genreToIndex[itemGenre]] = 1
   end
   
   return genre
end




function movieLensLoader:LoadMetaV(conf) 


   if #conf.metaItem > 0 then

      local moviesfile = io.open(conf.metaItem, "r")

      for line in moviesfile:lines() do

         local movieIdStr, title, genre = line:match('(%d+)::(.*)::(.*)')
         --local movieIdStr, title, day, month, year, url, genreStr = line:match('(%d+)|(.*)|(%d+)-(%a+)-(%d+)||(.-)|(.*)')

         if movieIdStr ~= nil then 

            local movieId = tonumber(movieIdStr)
            
            
            local info = self.train.V.info[movieId] or {}

            info.title  = title
            info.genre  = genreToBinary(genre)

            info.full       = info.genre
            info.fullSparse = info.full:sparsify(0, self.train.V.dimension)

            self.train.V.info[movieId] = info   

         else
            local movieIdStr = line:match('(%d+)|')
            local movieId = tonumber(movieIdStr)
            print("unable to parse movie (".. movieId .. ") : " .. line)
            self.train.V.info[movieId] = {}
         end
      end
      
     moviesfile:close()
     
     self.train.V.info.metaDim = 18

   end

end

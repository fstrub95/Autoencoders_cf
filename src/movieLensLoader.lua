----------------------------------------------------------------------------
----------------------------------------------------------------------------


local movieLensLoader, parent = torch.class('movieLensLoader', 'DataLoader')

function movieLensLoader:LoadRatings(conf)

   --no pre-process/post-processing
   function preprocess(x)  return (x-3)/2 end
   function postprocess(x) return 2*x+3 end

   -- step 3 : load ratings
   local ratesfile = io.open(conf.ratings, "r")


   self.movieHash = {}
   self.userHash  = {}

   local itemCounter = 1
   local userCounter = 1
 

   -- Step 1 : Retrieve movies'scores...th
   local i = 0
   for line in ratesfile:lines() do

      local userIdStr, movieIdStr, ratingStr = line:match('(%d+)::(%d+)::(%d%.?%d?)::(%d+)')

      local userId  = tonumber(userIdStr)
      local itemId  = tonumber(movieIdStr)
      local rating  = tonumber(ratingStr)

      local itemIndex = self.movieHash[itemId]
      if itemIndex == nil then
         self.movieHash[itemId] = itemCounter
         itemIndex   = itemCounter
         itemCounter = itemCounter + 1
      end

      local userIndex = self.userHash[userId]
      if userIndex == nil then
         self.userHash[userId] = userCounter
         userIndex   = userCounter
         userCounter = userCounter + 1
      end


      rating = preprocess(rating)

      self:AppendOneRating(userIndex, itemIndex, rating)

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

         local userId    = tonumber(userIdStr)
         local userIndex = self.userHash[userId] 

         if userIndex ~= nil then

            local info = self.train.U.info[userIndex] or {}

            info.id     = userId
            info.sex    = sexToBinary(sex)
            info.age    = ageToBinary(age)
            info.job    = jobToBinary(job)

            info.full       = torch.cat({info.sex, info.age, info.job})
            info.fullSparse = info.full:sparsify(0, self.train.U.dimension)

            self.train.U.info[userIndex] = info
         else
            print("No ratings for user : " .. userId )
         end

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
   "IMAX"
 }
 
 
 local genreToIndex = {}
 for k, genre in pairs(genreList) do
   genreToIndex[genre] = k
 end
 genreToIndex["Children"] = 4



local function genreToBinary(genreStr, offset)

   local genre = torch.zeros(#genreList)

   local genreTable = string.split(genreStr, "|")

   for _,itemGenre in pairs(genreTable) do 
      
      --typo exception
      if itemGenre == "Children" then itemGenre = "Children's" end
      
      local genreIndex = genreToIndex[itemGenre]
      if genreIndex ~= nil then
        genre[genreIndex] = 1
      else
        error("Unknow genre : " .. itemGenre)
      end
      
   end
   
   return genre
end




function movieLensLoader:LoadMetaV(conf) 


  if #conf.tags > 0 then

    local csv2tensor = require 'csv2tensor'
    local tagTensor  = csv2tensor.load(conf.tags)

    for i = 1, tagTensor:size(1) do

      -- idMovie, tag1, tag2, tag3 etc.
      local movieId = tagTensor[i][1]
      local tag     = tagTensor[{i, {2, tagTensor:size(2)}}]

      local movieIndex = self.movieHash[movieId]
      if movieIndex ~= nil then
        local info = self.train.V.info[movieIndex] or {}
        
        info.tag  = tag

        info.full = tag
        info.fullSparse = tag:sparsify(0, self.train.V.dimension)

        self.train.V.info[movieIndex] = info
      end

    end
  end


   if #conf.metaItem > 0 then

      local moviesfile = io.open(conf.metaItem, "r")

      for line in moviesfile:lines() do

         local movieIdStr, title, genreStr = line:match('(%d+)::(.*)::(.*)')
         --local movieIdStr, title, day, month, year, url, genreStr = line:match('(%d+)|(.*)|(%d+)-(%a+)-(%d+)||(.-)|(.*)')

         if movieIdStr ~= nil then 

            local movieId    = tonumber(movieIdStr)     
            local movieIndex = self.movieHash[movieId]

            if movieIndex ~= nil then
                
                local info = self.train.V.info[movieIndex] or {}

                info.id     = movieId
                info.title  = title
                info.genre  = genreToBinary(genreStr)


                --if there is some tags, append the genre to the tags
                if info.full then  info.full = info.genre:cat(info.full)
                else               info.full = info.genre
                end 

                info.fullSparse = info.full:sparsify(0, self.train.V.dimension)

                self.train.V.info[movieIndex] = info

            else
                print("No ratings for movieId : " .. movieId .. " - " .. title)
            end

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

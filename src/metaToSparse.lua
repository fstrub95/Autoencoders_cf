function jobToBinary(jobStr, offset)
      local sparseJob = torch.ones(1,2)
   
      local jobId = tonumber(jobStr)+1
      sparseJob[1][1] = jobId + offset
      
      return sparseJob;
   end

   function sexToBinary(sexStr, offset)
      local sparseSex = torch.ones(1,2)
   
      if     sexStr == 'M' then sparseSex[1][1] = 1 
      elseif sexStr == 'F' then sparseSex[1][1] = 2 
      end
   
      sparseSex[1][1] = sparseSex[1][1] + offset
   
      return sparseSex; 
   
   end

   function ageToBinary(ageStr, offset)
      local sparseAge = torch.ones(1,2)
   
       local ageId = tonumber(ageStr)
   
      if     ageId == 1  then sparseAge[1][1] = 1 
      elseif ageId == 18 then sparseAge[1][1] = 2 
      elseif ageId == 25 then sparseAge[1][1] = 3 
      elseif ageId == 35 then sparseAge[1][1] = 4 
      elseif ageId == 45 then sparseAge[1][1] = 5 
      elseif ageId == 50 then sparseAge[1][1] = 6 
      elseif ageId == 56 then sparseAge[1][1] = 7 
      end
      
      sparseAge[1][1] = sparseAge[1][1] + offset
   
      return sparseAge; 
   
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
 for k = 1, #genreList do
   genreToIndex[genreList[k]] = k
 end



function genreToBinary(genreStr, offset)

   local genreTable = string.split(genreStr, "|")
   local genreSparse = torch.ones(#genreTable, 2)

   local k = 1
   for _,itemGenre in pairs(genreTable) do 
      genreSparse[k][1] = genreToIndex[itemGenre] + offset
      k = k + 1
   end
   
   genreSparse:sort()
   
   return genreSparse
end
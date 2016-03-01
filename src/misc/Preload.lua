function LoadData(file, params)

   local use_gpu = params.gpu > 0
   local type    = params.type
   
   if type ~= "U" and type ~= "V" then
      error("Invalid network type : " .. type .. ". It should either be U or V")
   end 
  
   --Load data
   print("Loading data from disk...")
   local data = torch.load(params.file) 

   -- keep only the useful data for the network
   local train   = data.train[type].data
   local test    = data.test [type].data
   local info    = data.train[type].info
   
   print("")
   print("Users loaded      : " .. data.train.U.info.size)
   print("Items loaded      : " .. data.train.V.info.size)
   print("")
   print("No Train ratings  : " .. data.train[type].info.noRating)
   print("No Test  ratings  : " .. data.test[type] .info.noRating)
   print("Training ratio    : " .. data.train[type].info.noRating / (data.test[type].info.noRating + data.train[type].info.noRating))
   print("Training density  : " .. info.noRating / (info.size*info.dimension) )
   print("")


   if use_gpu then
      print("Loading cunn...")
      require("cunn")


      if params.seed and params.seed > 0 then
         torch.manualSeed(params.seed)
      else
         torch.manualSeed(torch.seed())
      end


      print("Loading data to GPU...")
      for k, _ in pairs(train) do      --dirty code, but it does the job

         train[k] = train[k]:cuda()

         -- put info on GPU
         if info.metaDim then
            info[k]            = info[k]            or {}
            info[k].full       = info[k].full       or torch.Tensor(info.metaDim):zero():cuda()
            info[k].fullSparse = info[k].fullSparse or torch.Tensor():cuda()

            info[k].full       = info[k].full:cuda()
            info[k].fullSparse = info[k].fullSparse:cuda()
         end
      end

      for k, _ in pairs(test) do

         test[k] = test[k]:cuda()

         -- put info on GPU
         if info.metaDim then
            info[k]            = info[k]            or {}
            info[k].full       = info[k].full       or torch.Tensor(info.metaDim):zero():cuda()
            info[k].fullSparse = info[k].fullSparse or torch.Tensor():cuda()
            
            info[k].full       = info[k].full:cuda()
            info[k].fullSparse = info[k].fullSparse:cuda()
         end
      end
   end   


   print("Unbias the data...")

   if type == "U" then -- unbias V
      for k, u in pairs(train) do
         info[k]      = info[k]      or {}     
         info[k].mean = info[k].mean or u[{{}, 2}]:mean()
      
         u[{{}, 2}]:add(-info[k].mean) --center input
      end
   else -- unbias V
      for k, v in pairs(train) do
         info[k]      = info[k]      or {}     
         info[k].mean = info[k].mean or v[{{}, 2}]:mean()

         v[{{}, 2}]:add(-info[k].mean) --center input
      end
   end

   print("Data was successfully preloaded...")

   return train, test, info
end

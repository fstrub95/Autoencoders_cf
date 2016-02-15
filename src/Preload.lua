

function LoadData(file, params)

   local use_gpu = params.gpu > 0
   local type    = params.type
   
   if type ~= "U" or type ~= "V" then
      error("Invalid network type : " .. type .. " . It should either be U or V")
   end 
  
   --Load data
   print("Loading data from disk...")
   local data = torch.load(params.file) 
   local train = data.train
   local test  = data.test
   
   
   print(train.U.size .. " Users loaded")
   print(train.V.size .. " Items loaded")
   print("")
   print("Training ratio   : " .. train.U.noRating / (train.U.noRating + test.U.noRating))
   print("No Train ratings : " .. train.U.noRating)
   print("No Test  ratings : " .. test.U.noRating)
   print("")


   if use_gpu then
      print("Loading cunn...")
      require("cunn")


      if params.seed > 0 then
         torch.manualSeed(params.seed)
      else
         torch.manualSeed(torch.seed())
      end


      print("Loading data to GPU...")
      local _train = train[type]
      local _test  = test [type]


      --dirty code, but it does the job
      for k, _ in pairs(train[type].data) do

         _train.data[k] = _train.data[k]:cuda()

         -- put info on GPU
         if _train.info.metaDim then
            _train.info[k]            = _train.info[k]            or {}
            _train.info[k].full       = _train.info[k].full       or torch.Tensor(_train.info.metaDim):zero():cuda()
            _train.info[k].fullSparse = _train.info[k].fullSparse or torch.Tensor():cuda()

            _train.info[k].full       = _train.info[k].full:cuda()
            _train.info[k].fullSparse = _train.info[k].fullSparse:cuda()
         end
      end

      for k, _ in pairs(test[type].data) do

         _test .data[k] = _test.data[k]:cuda()

         -- put info on GPU
         if _train.info.metaDim then
            _train.info[k]            = _train.info[k]            or {}
            _train.info[k].full       = _train.info[k].full       or torch.Tensor(_train.info.metaDim):zero():cuda()
            _train.info[k].fullSparse = _train.info[k].fullSparse or torch.Tensor():cuda()
            
            _train.info[k].full       = _train.info[k].full:cuda()
            _train.info[k].fullSparse = _train.info[k].fullSparse:cuda()
         end

      end

   end   


   print("Unbias the data...")
   if type == "U" then -- unbias V
      for k, u in pairs(train.U.data) do
         train.U.info[k]      = train.U.info[k]      or {}     
         train.U.info[k].mean = train.U.info[k].mean or u[{{}, 2}]:mean()
      
         u[{{}, 2}]:add(-train.U.info[k].mean) --center input
      end
   else -- unbias V
      for k, v in pairs(train.V.data) do
         train.V.info[k]      = train.V.info[k]      or {}     
         train.V.info[k].mean = train.V.info[k].mean or v[{{}, 2}]:mean()

         v[{{}, 2}]:add(-train.V.info[k].mean) --center input
      end
   end


   -- keep only the data usefull for the network
   local train   = data.train[type].data
   local test    = data.test [type].data
   local info    = data.train[type].info

   print("Data was successfully preload...")

   return train, test, info
end
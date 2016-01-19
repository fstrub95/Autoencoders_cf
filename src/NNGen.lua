local NNGen, parent = torch.class("NNGen", "AlgoGen")

local threads = require 'threads'

function NNGen:GenerateOne()           

   self.nnType = "V"

   local gene = {}
   gene.layer1       = 700--torch.random (400 , 600)
   gene.alpha1       = torch.uniform(0.8 , 1.2)
   gene.beta1        = torch.uniform(0.5 , 1)
   gene.hide1        = torch.uniform(0   , 0.5)
   gene.batch1       = torch.random(10, 50)
   gene.lrt1         = torch.uniform(0.1, 0.0001)
   gene.lrtDecay1    = torch.uniform(-0.2, 0.4)
   gene.weigthDecay1 = torch.uniform(0.1, 0.0001)
   --gene.grad1        = "sgd"

   gene.layer2       = 500--torch.random (500 , 800)
   gene.alpha2       = torch.uniform(0.8 , 1.2)
   gene.beta2        = torch.uniform(0.5 , 1)
   gene.hide2        = torch.uniform(0   , 0.5)
   gene.batch2       = torch.random(10, 50)
   gene.lrt2         = torch.uniform(0.1, 0.0001)
   gene.lrtDecay2    = torch.uniform(-0.2, 0.4)
   gene.weigthDecay2 = torch.uniform(0.1, 0.0001)
   --gene.grad2        = "sgd"


   return gene
end


function NNGen:LoadGene(gene)

   local conf = 
      {
         layer1 = 
         {      
            layerSize = gene.layer1,
            { 
               criterion = nnsparse.SDAECriterionGPU(nn.MSECriterion(),
                  {
                     alpha     = gene.alpha1,
                     beta      = gene.beta1,
                     hideRatio = gene.hide1,
                  }), 
               noEpoch           = 10, 
               miniBatchSize     = gene.batch1,
               learningRate      = gene.lrt1,  
               learningRateDecay = gene.lrtDecay1,
               weightDecay       = gene.weigthDecay1,
            },

         },

         layer2 = 
         {
            layerSize = gene.layer2,
            { 
               criterion = nnsparse.SDAECriterionGPU(nn.MSECriterion(),
                  {
                     alpha = 1,
                     beta  = 0.8,
                     noiseRatio = 0.2,
                     noiseStd  = 0.02, 
                  }),
               noEpoch = 4, 
               miniBatchSize = 20,
               learningRate  = 5e-5,  
               learningRateDecay = 0.1,
               weightDecay = 0.2,
               momentum = 0.8
            },

            {
               criterion = nnsparse.SDAECriterionGPU(nn.MSECriterion(),
                  {
                     alpha     = gene.alpha2,
                     beta      = gene.beta2,
                     hideRatio = gene.hide2,
                  }), 
               noEpoch           = 8, 
               miniBatchSize     = gene.batch2,
               learningRate      = gene.lrt2,  
               learningRateDecay = gene.lrtDecay2,
               weightDecay       = gene.weigthDecay2,
            },
         },

      }

   return conf

end



function NNGen:EvaluateAll(genes)      

   local nthread = 2 
   local njob = table.Count(genes)
   local pool = threads.Threads(
      nthread,
      function(threadid)
         print('starting a new thread/state number ' .. threadid)

         require("nn")
         require("optim")
         require("xlua")
         require("sys")
         
         USE_GPU = true
         
         if USE_GPU then
            require("cunn")
         end
        
         torch.setdefaulttensortype('torch.FloatTensor') 
         require("nnsparse")

         dofile("AlgoTools.lua")
         dofile("tools.lua")


         dofile("AlgoGen.lua")
         dofile("NNGen.lua")

         dofile("SDAECriterionGPU.lua")
         dofile("AutoEncoderTrainer.lua")
         dofile("LearnU.lua")
         dofile("Appender.lua")
      end
   )

   local jobdone = 0
   for i = 1,njob do

      pool:addjob(

            function()
               local noGPUDevice = math.fmod(i,2) + 1
               cutorch.setDevice(noGPUDevice)
               
               print(string.format('START Training on thread %x with GPU %d',  __threadid, noGPUDevice))

               local GPUTrain = table.Copy(self.train)
               local GPUTest  = table.Copy(self.test)

               print("Loading data to GPU [".. noGPUDevice .. "/" .. cutorch.getDevice() .."]...")

               local function toGPU(type)
                  local _train = self.train[type]
                  local _test  = self.test [type]

                  for k, _ in pairs(_train.data) do

                     GPUTrain[type].data[k] = _train.data[k]:cuda()

                     if GPUTrain[type].info.metaDim then
                        GPUTrain[type].full               = _train.info[k].full:cuda()
                        GPUTrain[type].info[k].fullSparse = _train.info[k].fullSparse:cuda()
                     end
                  end

                  for k, _ in pairs(_test.data) do

                     GPUTest[type].data[k] = _test.data[k]:cuda()

                     if GPUTest[type].info.metaDim then
                        GPUTest[type].full               = _test.info[k].full:cuda()
                        GPUTest[type].info[k].fullSparse = _test.info[k].fullSparse:cuda()
                     end
                  end

               end

               toGPU("U")
               toGPU("V")

               SHOW_PROGRESS = true

               --------------------------------------------------------------------------------

               local gene = genes[i].gene
               local conf = self:LoadGene(gene) 

               local fitness = 999

               if     self.nnType == "U" then 
			fitness = trainU(GPUTrain, GPUTest, conf)
               elseif self.nnType == "V" then 
			fitness = trainV(GPUTrain, GPUTest, conf)           
               else   
                  error("Invalid network type")
               end

               return __threadid, fitness
            end,

            function(id, fitness)
               genes[i].score = fitness
               
               print(string.format("Training gene %d finished (thread ID %x). Score : %f" , i, id, fitness))
               jobdone = jobdone + 1
            end
      )
   end

   pool:synchronize()
   pool:terminate() 

   return genes

end


function NNGen:Preconfigure(genConf)   

   --Load data
   print("loading data...")
   local data = torch.load(genConf.file) 
   self.train = data.train
   self.test  = data.test

   print(self.train.U.size .. " Users loaded")
   print(self.train.V.size .. " Items loaded")

   -- unbias U
   for k, u in pairs(self.train.U.data) do
      u[{{}, 2}]:add(-self.train.U.info[k].mean) --center input
   end

   --unbias V
   for k, v in pairs(self.train.V.data) do
      self.train.V.info[k] = self.train.V.info[k] or {}     
      self.train.V.info[k].mean = v[{{}, 2}]:mean()

      v[{{}, 2}]:add(-self.train.V.info[k].mean) --center input
   end


   SHOW_PROGRESS = false
   USE_GPU       = true 

end



























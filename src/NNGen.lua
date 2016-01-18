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
               noEpoch           = 5, 
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
               noEpoch = 2, 
               miniBatchSize = 10,
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
               noEpoch           = 2, 
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

   local nthread = 3
   local njob    = table.Count(genes)
   local pool = threads.Threads(
      nthread,
      function(threadid)
         print('starting a new thread/state number ' .. threadid)

         require("nn")
         require("optim")
         require("xlua")
         require("sys")

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

               print(string.format('START Training on thread %x',  __threadid))

               local gene = genes[i].gene
               local conf = self:LoadGene(gene) 

               local fitness = 999

               if     self.nnType == "U" then fitness = trainU(self.train, self.test, conf)
               elseif self.nnType == "V" then fitness = trainV(self.train, self.test, conf)           
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

   SHOW_PROGRESS = false
   USE_GPU       = false

      if USE_GPU then
        print("Loading cunn...")
        require("cunn")
        
        print("Loading data to GPU...")
        local function toGPU(type)
           local _train = train[type]
           local _test  = test [type]
           
           for k, _ in pairs(train[type].data) do
           
               _train.data[k] = _train.data[k]:cuda()
               
               if _test .data[k] then  
                  _test .data[k] = _test .data[k]:cuda()
               end
                    
               if _train.info.metaDim then
                  _train.info[k].full       = _train.info[k].full:cuda()
                  _train.info[k].fullSparse = _train.info[k].fullSparse:cuda()
               end
           end
        end
        
        toGPU("U")
        toGPU("V")
        
      end

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

end




























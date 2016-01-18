local ExampleGenConf = 
{
   noGenes = 20,
   noEpoch = 10,  
   ratioBest   = 1/10,
   rationCross = 2/10,
   ratioMutate = 4/10,
   ratioNew    = 5/10,
}



local AlgoGen = torch.class('AlgoGen')

function AlgoGen:__init(genConf)
   self.genConf = genConf
   
   self.ratioBest   = genConf.ratioBest
   self.ratioCross  = genConf.ratioCross
   self.ratioMutate = genConf.ratioMutate
   self.ratioNew    = genConf.ratioNew
   
   assert(self.ratioBest + self.ratioCross + self.ratioMutate + self.ratioNew == 1)
      
   self.noGenes = genConf.noGenes
   self.noEpoch = genConf.noEpoch
   
end

--To implement
function AlgoGen:Preconfigure(genConf)   end
function AlgoGen:GenerateOne()           return nil   end
function AlgoGen:EvaluateOne(gene)       return 0     end







function AlgoGen:PrintOne(gene)             
   print('Gene:')
   for key, value in pairs(gene) do
      print(" - " .. key .. ' : ' .. value)
   end
end


function AlgoGen:MutateOne(gene) 
   
   --Generate a new Gene and ramdomly pick one of his element to apply it to the mutated gene
   local geneBuf = self:GenerateOne()
   local toMutate math.random( 1, #geneBuf )   
   gene[toMutate] = geneBuf[toMutate]

   return gene  
end


function AlgoGen:CrossOver(geneA, geneB) 
   local geneC = {}
   for key, _ in pairs(geneA) do
      if type(value) == "number" then
         geneC[key] = 2/3 * geneA[key] + 1/3 * geneB[key] 
      else
         geneC[key] = geneA[key]
      end
   end
   return geneC 
end



function AlgoGen:Start(genConf)
   self:Preconfigure(genConf)
   return self:Learn()
end



function AlgoGen:Learn()


   print("Generate initial genes")
   local genes = {}
   for k = 1, self.noGenes do
      local newGene = {}
      newGene.gene  = self:GenerateOne()
      newGene.score = 999
      
      genes[#genes+1] = newGene
   end


   print("Start Alogo Gen")
   for t = 1, self.noEpoch + 1 do

      print("Evaluate genes...")
      for k, oneGene in pairs(genes) do 
           oneGene.score = self:EvaluateOne(oneGene.gene)
      end
      
      
      print("Sort the final scores") 
      table.sort(genes, function(geneA,geneB) return geneA.score > geneB.score end)      

      print("-------------------------------------------------------------------------------")  
      print("-------------------------------------------------------------------------------")  
      print("Best scores:")    
      print(" - No1: " .. genes[1].score) print("") self:PrintOne(genes[1].gene) print("")
      print(" - No2: " .. genes[2].score) print("") self:PrintOne(genes[2].gene) print("")
      print(" - No3: " .. genes[3].score) print("") self:PrintOne(genes[2].gene) print("")
      print("-------------------------------------------------------------------------------")  
      print("-------------------------------------------------------------------------------")  
      
      
      --Stop creating new families when the number of epoch is over
      if t == self.noEpoch then break end
      
      
      print("Epoch : " .. t)
      
      -- Start Creating a new family of genes
      local newGenes = {}
      local noBest   = math.floor(#genes * self.ratioBest)
      local noCross  = math.floor(#genes * self.ratioCross)
      local noMutate = math.floor(#genes * self.ratioMutate)
      local noNew    = math.floor(#genes * self.ratioNew)


      -- Create two sets containing the best genes 
      local S1 = { unpack(genes,             1, self.noBest) }
      local S2 = { unpack(genes, self.noBest+1, self.noCross) }


      -- Copy Best genes
      newGenes = table.concat(newGenes, S1)
  
      -- Cross Over set1 and set2
      for _, fatherGene in pairs(S2) do
         local motherGene =  S1[math.random( 1, #S1 )]
         local newGene1 = self:CrossOne(motherGene, fatherGene)
         local newGene2 = self:CrossOne(fatherGene, motherGene)
         table.insert(newGenes, newGene1)
         table.insert(newGenes, newGene2)
      end
      
      
      -- Mutate gene in set1
      for k = 1, noMutate do
         local motherGene =  S1[math.random( 1, #S1 )]
         local newGene = self:MutateOne(motherGene)
         table.insert(newGenes, newGene)
      end
      
      
      -- Generate new genes
      for k = 1, noNew do
         local newGene = self:GenerateOne()
         table.insert(newGenes, newGene)
      end
   
   
      --Replace old genes by new one
      genes = newGenes
   
   end

   return genes

end



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
   
   self.sigma       = genConf.sigma
   
   assert(self.ratioBest + self.ratioCross + self.ratioMutate + self.ratioNew == 1)
      
   self.noGenes = genConf.noGenes
   self.noEpoch = genConf.noEpoch
   
end

--To implement
function AlgoGen:Preconfigure(genConf)   end
function AlgoGen:GenerateOne()           return nil   end
function AlgoGen:EvaluateAll(genes)      return nil   end










function AlgoGen:PrintOne(gene)             
    local bufKey = {}
    for key, _ in pairs(gene) do table.insert(bufKey, key) end
    table.sort(bufKey)
    for _, key in pairs(bufKey) do 
       print(" - " .. key .. ' : ' .. gene[key])
    end

end


function AlgoGen:MutateOne(gene) 
   
   
   --Generate a new Gene and ramdomly pick one of his element to apply it to the mutated gene
   local newGene  = table.Copy(gene)
   for key, value in pairs(geneA) do
      if type(value) == "number" then
         newGene[key] = torch.normal(value, self.std)
      end
   end
   return newGene  
end


function AlgoGen:CrossOne(geneA, geneB) 
   local geneC = {}
   for key, value in pairs(geneA) do
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
      newGene.score = NaN
      
      genes[#genes+1] = newGene
   end


   print("Start Alogo Gen")
   for t = 1, self.noEpoch + 1 do

      print("Evaluate genes...")
        genes = self:EvaluateAll(genes, self.genConf)
--      for k, oneGene in pairs(genes) do 
--           oneGene.score = self:EvaluateOne(oneGene.gene)
--      end
      
      
      print("Sort the final scores") 
      table.sort(genes, function(geneA,geneB) return geneA.score < geneB.score end)      

      print("-------------------------------------------------------------------------------")  
      print("-------------------------------------------------------------------------------")  
      print("Best scores:")    
      print("#### Score No1: " .. genes[1].score)  self:PrintOne(genes[1].gene) print("")
      print("#### Score No2: " .. genes[2].score)  self:PrintOne(genes[2].gene) print("")
      print("#### Score No3: " .. genes[3].score)  self:PrintOne(genes[3].gene) print("")
      print("-------------------------------------------------------------------------------")  
      print("-------------------------------------------------------------------------------")  
      
      
      --Stop creating new families when the number of epoch is over
      if t == self.noEpoch then break end
      
      
      print("Epoch : " .. t)
      
      -- Start Creating a new family of genes
      local newGenes = {}
      local noBest   = math.floor(table.Count(genes) * self.ratioBest)
      local noCross  = math.floor(table.Count(genes) * self.ratioCross)
      local noMutate = math.floor(table.Count(genes) * self.ratioMutate)
      local noNew    = math.floor(table.Count(genes) * self.ratioNew)


      -- Create two sets containing the best genes 
      local S1 = { unpack(genes,          1, noBest) }
      local S2 = { unpack(genes, noBest + 1, noBest + noCross) }


      -- Copy Best genes
      table.merge(newGenes, S1)
  
  
     local val, key = table.Random( geneBuf )   
   local newGene  = table.Copy(gene)
  
  
      -- Cross Over set1 and set2
      for k = 1, (noCross/2 + 0.5)  do
         local geneA = S1[math.random( 1, #S1 )].gene
         local geneB = S2[math.random( 1, #S2 )].gene
         local newGene1 = { gene = self:CrossOne(geneA, geneB), score = NaN }
         local newGene2 = { gene = self:CrossOne(geneB, geneA), score = NaN } 
         table.insert(newGenes, newGene1)
         table.insert(newGenes, newGene2)
      end
      
      
      -- Mutate gene in set1
      self.std = self.sigma / math.pow(table.Count(genes), 1/table.Count(genes[1].gene))
      for k = 1, noMutate do
         local geneA =  S1[math.random( 1, #S1 )].gene
         local newGene = { gene = self:MutateOne(geneA), score = NaN }
         table.insert(newGenes, newGene)
      end
      
      
      -- Generate new genes
      for k = 1, noNew do
         local newGene = { gene = self:GenerateOne(), score = NaN}
         table.insert(newGenes, newGene)
      end
   
   
      --Replace old genes by new one
      genes = newGenes
   
   end

   return genes

end



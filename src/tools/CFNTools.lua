require("nn")
require("torch")


cfn = cfn or {}


function cfn.FlatNetwork(network)

   function FlatNetworkRecursive(network, layers)
      for i = 1, network:size() do
         local layer = network:get(i)
         if torch.type(layer) == "nn.Sequential" then
            FlatNetworkRecursive(layer, layers)
         else
            layers[#layers+1] = layer
         end
      end
   end
   
   local layers = {}
   FlatNetworkRecursive(network, layers)
   
   
   local flatNetwork = nn.Sequential()
   for _, layer in pairs(layers) do
      flatNetwork:add(layer)
   end
   
   return flatNetwork
end



local Batchifier, parent = torch.class('cfn.Batchifier')

function Batchifier:__init(network, outputSize, appenderIn, info)
   self.network    = network
   self.outputSize = outputSize
   self.appenderIn = appenderIn
   self.info = info
end

function Batchifier:forward(data, batchSize)
   
   -- no need for batch for dense Tensor
   if torch.isTensor(data) then
   
      if self.appenderIn then
         local denseInfo  = data[1].new(self.info.size, self.info.metaDim)
         for k = 1, data:size(1) do
            denseInfo[k] = self.info[k] or 0
         end
         self.appenderIn:prepareInput(denseInfo)
      end
      
      return self.network:forward(data)
   end
      
   batchSize = batchSize or 20
   
   local nFrame    = GetnElement(data)

   --Prepare minibatch
   local inputs   = {}
   local outputs  = data[1].new(nFrame, self.outputSize) 
   
   local denseInfo  = data[1].new(batchSize, self.info.metaDim):zero()
   local sparseInfo = {}
   
   assert(torch.type(data) == "table")

   local i      = 1
   local cursor = 0
   for k, input in pairs(data) do

      inputs[i]  = input   
      
      if self.appenderIn then
          denseInfo[i]  = self.info[k].full
          sparseInfo[i] = self.info[k].fullSparse
      end
      
      i = i + 1

      --compute loss when minibatch is ready
      if #inputs == batchSize then
         local start =  cursor   *batchSize + 1
         local stop  = (cursor+1)*batchSize

         if self.appenderIn then
            self.appenderIn:prepareInput(denseInfo,sparseInfo)
         end
         
         outputs[{{start,stop},{}}] = self.network:forward(inputs)
         
         inputs = {}
         
         i = 1
         cursor = cursor + 1       
      end
   end

   if #inputs > 0 then
      local start = nFrame-(i-1) + 1
      local stop  = nFrame

      if self.appenderIn then
         self.appenderIn:prepareInput(denseInfo[{{1, #inputs},{}}], {unpack(sparseInfo, 1, #inputs)})
      end

      outputs[{{start,stop},{}}] = self.network:forward(inputs)
   end  

   return outputs

end
   

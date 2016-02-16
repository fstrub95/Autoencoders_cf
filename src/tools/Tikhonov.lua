nnsparse = nnsparse or {}

local Tikhonov = torch.class('nnsparse.Tikhonov')


local function IsLinearLayer(layer)

   local type = torch.type(layer)

   if type == "nn.Linear" or
      type == "nn.SparseLinear" or
      type == "nn.SparseLinearBatch"
   then return true
   else return false
   end
end



function Tikhonov:__init(lambda, network)

   -- retrieve input/output layers
   local inLayer
   for k = 1, network:size() do
      inLayer = network:get(k)
      if IsLinearLayer(inLayer) then break end
   end
   
   local outLayer
   for k = network:size(), 1, -1 do
      outLayer = network:get(k)
      if IsLinearLayer(outLayer) then break end
   end

   local inParams = inLayer.weight:nElement() + inLayer.bias:nElement()
   local outParams = outLayer.weight:nElement() + outLayer.bias:nElement()
   local networkParams = network:getParameters():nElement()
   collectgarbage()
      
   -- memory pre-allocation
   self.lambda     = lambda
   self.lambdaBuf  = inLayer.weight.new(inLayer.weight:size(2))
   self.lambdas    = inLayer.weight.new(networkParams):fill(lambda)

   
   -- provide the output layer delimitation   
   local output = {}
   
   output.w = {}
   output.w.start = networkParams - outParams + 1
   output.w.stop  = networkParams - outLayer.bias:nElement()
   output.w.size  = outLayer.weight:size()
   
   
   output.b = {}
   output.b.start = networkParams - outLayer.bias:nElement() + 1
   output.b.stop  = networkParams 
   output.b.size  = outLayer.bias:size() 
   
   self.output = output


   -- provide the input layer delimitation   
   local input = {}
   
   input.w = {}
   input.w.start = 1
   input.w.stop  = inLayer.weight:nElement()
   input.w.size  = inLayer.weight:size()
   
   self.input = input
   
   
   -- provide the hidden layer delimitation
--   local hidden = {}
--   hidden.start = inParams + 1
--   hidden.stop  = networkParams - outParams
--      
--   self.hidden = hidden

end

function Tikhonov:computeLambda(target)

   -- STEP 1 : compute the regulizer such as reg = lambda_i * n_(occurence_of_i) / batchSize 
   
   self.lambdaBuf:zero()
--   self.ones = self.ones or self.lambdaBuf.new(1):fill(1)
   
   for i, oneTarget in pairs(target) do
      local index = oneTarget[{{},1}]
      if torch.type(index) ~= "torch.CudaTensor" then
         index = index:long()
      end 
--      self.lambdaBuf:indexAdd(1, index ,self.ones:expandAs(index))
        self.lambdaBuf:indexFill(1, index , self.lambda)
   end
-- self.lambdaBuf:mul(self.lambda/#target)


   -- STEP 2 : compute the lambdas vector for weight decay

   --out layer --> same lambda are discontiguous in memory
   self.lambdas[{{self.output.w.start, self.output.w.stop}}] = self.lambdaBuf:view(-1,1):expand(self.output.w.size)
   self.lambdas[{{self.output.b.start, self.output.b.stop}}] = self.lambdaBuf

   --in layer --> same lambda are contiguous in memory. 
   self.lambdas[{{self.input.w.start, self.input.w.stop}}]:repeatTensor(self.lambdaBuf, self.input.w.size[1])
   
   
   --hidden layers
   --self.lambdas[{{self.hidden.start, self.hidden.stop}}]:fill(self.lambda) 

   --self.lambdas:fill(self.lambda)

   return self.lambdas
end

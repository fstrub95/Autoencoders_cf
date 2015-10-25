local SDAESparseCriterion, parent = torch.class('nn.SDAESparseCriterion', 'nn.Criterion')

function SDAESparseCriterion:__init(criterion, SDAEconf)
   parent.__init(self)

   self.criterion = nn.SDAECriterion(criterion, SDAEconf)

end 


function SDAESparseCriterion:prepareInput(inputs)

   assert(torch.type(inputs) == "table")

   self.inputsBuf = {}
   self.masks     = {}   

   for k, input in pairs(inputs) do

      -- Store a new input
      local inputBuf = input:clone()

      -- modify the input
      inputBuf[{{},2}]:copy(self.criterion:prepareInput(input[{{},2}]))

      -- store the prepared input
      self.inputsBuf[k] = inputBuf

      --store masks
      local mask = {}
      mask.alpha = self.criterion.maskAlpha:clone()
      mask.beta  = self.criterion.maskBeta:clone()
      self.masks[k] = mask

   end

   return self.inputsBuf

end


function SDAESparseCriterion:updateOutput(estimates, targets)

   assert(torch.type(targets) == "table")

   self.estimateBuf = self.predictBuf or torch.Tensor()

   local loss = 0
   local totalSize = 0
   for k, target in pairs(targets) do

      -- retrieve expected target
      local index = target[{{},1}]
      local t     = target[{{},2}]
      local size  = target:size(1)

      local estimate = estimates[k]

      -- only conserve the prediction with target index
      local i = 0
      self.estimateBuf:resizeAs(index)
      self.estimateBuf:apply(function()
         i = i + 1
         return estimate[index[i]] end
      )

      -- update SDAE criterion 
      self.criterion.maskAlpha = self.masks[k].alpha
      self.criterion.maskBeta  = self.masks[k].beta

      loss = loss + self.criterion:updateOutput(self.estimateBuf, t)*size 

      totalSize = totalSize + size

   end
   
   loss = loss/totalSize

   return loss

end




function SDAESparseCriterion:updateGradInput(estimates, targets)


   assert(torch.type(targets) == "table")

   self.dloss = self.dloss or estimates[1].new()
   self.dloss:resizeAs(estimates):zero()

   self.targetBuf = self.targetBuf or torch.Tensor()


   for k, target in pairs(targets) do

      -- retrieve expected targets
      local index = target[{{},1}]
      local t     = target[{{},2}]
      
      local estimate = estimates[k]
   
      self.targetBuf:resizeAs(estimate):copy(estimate)
      self.targetBuf:indexCopy(1, index:long(), t)
   
      -- update SDAE criterion 
      self.criterion.maskAlpha = self.masks[k].alpha
      self.criterion.maskBeta  = self.masks[k].beta
   
      self.dloss[k] = self.criterion:updateGradInput(estimate, self.targetBuf, index)
   
   end

   return self.dloss

end

function SDAESparseCriterion:SetAlpha(alpha) self.criterion.alpha = alpha end
function SDAESparseCriterion:SetBeta(beta)   self.criterion.beta  = beta end


function SDAESparseCriterion:__tostring__()
   return "Sparse " .. self.criterion:__tostring__()
end


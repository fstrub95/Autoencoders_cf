function nn.Criterion:prepareInput(input)
   return input
end
function nn.Criterion:training()
   self.train = true
end


function nn.Criterion:evaluate()
   self.train = false
end



local SDAECriterion, parent = torch.class('nn.SDAECriterion', 'nn.Criterion')

function SDAECriterion:__init(criterion, SDAEconf)
   parent.__init(self)

   self.criterion = criterion

   self.alpha = SDAEconf.alpha or 1   
   self.beta  = SDAEconf.beta  or 0

   self.noiseRatio = SDAEconf.noiseRatio or 0
   self.noiseMean = SDAEconf.noiseMean or 0
   self.noiseStd = SDAEconf.noiseStd or 0.2

   self.flipRatio = SDAEconf.flipRatio or 0
   self.flipRange = SDAEconf.flipRange


   self.hideRatio = SDAEconf.hideRatio or 0

   
   self.maskAlpha    = torch.Tensor():byte()
   self.maskBeta     = torch.Tensor():byte()

end


function SDAECriterion:prepareInput(input)

   self.input = self.input or input.new()
   self.input:resizeAs(input):copy(input)
   
   self.maskAlpha:resize(input:size()):byte()
   self.maskBeta:resize(input:size()):byte()


   --enable mini-batch by linearizing data
   local viewInput = input
   local viewAlpha = self.maskAlpha
   local viewBeta  = self.maskBeta
   if input:nDimension() > 1 then
      viewInput = input:view(-1)
      viewAlpha = self.maskAlpha:view(-1)
      viewBeta  = self.maskBeta:view(-1)
   end
   
   local i = 0
   self.input:apply(function(x)

         i = i + 1

         viewAlpha[i] = 1
         viewBeta[i]  = 0

         local r = torch.uniform()
         if      r < self.noiseRatio                                   then return x + torch.normal(self.noiseMean, self.noiseStd)  -- add gaussian noise
         elseif  r < self.noiseRatio + self.flipRatio                  then return self.flipRange[torch.uniform() > 0.5 and 1 or 2] -- either return min/max
         elseif  r < self.noiseRatio + self.flipRatio + self.hideRatio then return 0                                                -- remove data
         else                                                           
            viewAlpha[i] = 0
            viewBeta[i]  = 1
            return x
         end

   end)
   
   return self.input

end



function SDAECriterion:updateOutput(estimate, target)

      local loss = 0
      local totalSize = 0
       
      --loss = loss + self.alpha * self.criterion:updateOutput(estimate[self.maskAlpha], target[self.maskAlpha])
      --loss = loss + self.beta  * self.criterion:updateOutput(estimate[self.maskBeta] , target[self.maskBeta])
      
      local _estimate = estimate[self.maskAlpha]
      if _estimate:nDimension() > 0 then 
         loss = loss + self.alpha * self.criterion:updateOutput(_estimate, target[self.maskAlpha])*_estimate:size(1)
      end 
      
      local _estimate = estimate[self.maskBeta]
      if _estimate:nDimension() > 0 then 
         loss = loss + self.beta  * self.criterion:updateOutput(_estimate , target[self.maskBeta])* _estimate:size(1)
      end
      
      return loss/estimate:size(1)
end



function SDAECriterion:updateGradInput(estimate, target, index)

   local dloss = self.criterion:updateGradInput(estimate , target)

   if index == nil then -- not-sparse

      local viewAlpha = self.maskAlpha
      if viewAlpha:nDimension() > 1 then
         viewAlpha = viewAlpha:view(-1)
      end

      local i = 0
      dloss:apply(function(x)
         i = i +1
         if viewAlpha[i] == 1 then
            return self.alpha * x 
         else
            return self.beta  * x
         end
      end
      )

   else --sparse

      local i = 1
      local k = 1

      dloss:apply(function(x)

            if k <= index:size(1) and index[k] == i then 
               if self.maskAlpha[k] == 1 then
                  x = self.alpha * x 
               else
                  x = self.beta * x
               end
               k = k + 1
            end  
            i = i + 1

            return x

      end
      )
   end

   return dloss

end

function SDAECriterion:__len() return 0 end

function SDAECriterion:__tostring__()
   return torch.type(self) .. " alpha: ".. self.alpha .. ", beta: " ..self.beta .. ",noiseRatio: " .. self.noiseRatio .. " ,flipRatio: " .. self.flipRatio .. ", hideratio:" .. self.hideRatio
end


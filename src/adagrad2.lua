function optim.adagrad2(opfunc, x, config, state)

   -- (0) get/update state
   if config == nil and state == nil then
      print('no state table, ADAGRAD initializing')
   end
   
   local config = config or {}
   local lr    = config.learningRate or 1e-3
   local lrd   = config.learningRateDecay or 0
   local wd    = config.weightDecay or 0
   local wds   = config.weightDecays
   
   local state = state or config
   state.evalCounter = state.evalCounter or 0
   local nevals = state.evalCounter

   -- (1) evaluate f(x) and df/dx
   local fx,dfdx = opfunc(x)

   -- (3) learning rate decay (annealing)
   local clr = lr / (1 + nevals*lrd)
      
   -- (4) weight decay with single or individual parameters
   if wd ~= 0 then
      dfdx:add(wd, x)
   elseif wds then
      if not state.decayParameters then
         state.decayParameters = torch.Tensor():typeAs(x):resizeAs(dfdx)
      end
      state.decayParameters:copy(wds):cmul(x)
      dfdx:add(state.decayParameters)
   end  
      
   -- (5) parameter update with single or individual learning rates
   if not state.paramVariance then
      state.paramVariance = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
      state.paramStd = torch.Tensor():typeAs(x):resizeAs(dfdx)
   end
   state.paramVariance:addcmul(1,dfdx,dfdx)
   state.paramStd:resizeAs(state.paramVariance):copy(state.paramVariance):sqrt()
   x:addcdiv(-clr, dfdx,state.paramStd:add(1e-10))

   -- (6) update evaluation counter
   state.evalCounter = state.evalCounter + 1

   -- return x*, f(x) before optimization
   return x,{fx}
end


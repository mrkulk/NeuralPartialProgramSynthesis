function rmsprop(dfdx, x, config, state)
   -- get parameters
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 1e-3
   local b1 = config.momentumDecay or 1
   local b2 = config.updateDecay or 1

   -- local fx, dfdx = opfunc(x)

   state.evalCounter = state.evalCounter or 0
   state.m = state.m or torch.Tensor():typeAs(dfdx):resizeAs(dfdx):fill(0)
   state.v = state.v or torch.Tensor():typeAs(dfdx):resizeAs(dfdx):fill(0)
   
   -- Decay term
   state.m:mul(1 - b1)

   -- New term 
   state.momentum_dfdx = state.momentum_dfdx or torch.Tensor():typeAs(dfdx):resizeAs(dfdx)
   state.momentum_dfdx:copy(dfdx)

   state.m:add(state.momentum_dfdx:mul(b1))

   -- Decay term of update
   state.v:mul(1 - b2)

   -- New update
   dfdx:cmul(dfdx):mul(b2)

   state.v:add(dfdx)

   -- calculate update step
   state.evalCounter = state.evalCounter + 1

   --Create new momentum Tensors for cutorch compatibility
   state.momentum_update = state.momentum_update or torch.Tensor():typeAs(state.m):resizeAs(state.m)
   state.momentum_update:copy(state.m)

   state.update = state.update or torch.Tensor():typeAs(state.v):resizeAs(state.v)
   state.update:copy(state.v)

   state.momentum_update:cdiv(state.update:add(1e-8):sqrt())

   local gamma = (math.sqrt(1 - math.pow(1 - b2,state.evalCounter))/(1 - math.pow(1 - b1, state.evalCounter)))
   state.momentum_update:mul(gamma)

   x:add(-lr, state.momentum_update)

   -- return x*, f(x) before optimization
   return x
end



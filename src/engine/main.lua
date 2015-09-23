-- Tejas D Kulkarni (tejask@mit.edu)

require('cunn')
require('nngraph')
require('base')
require('srctransform')
require('model')
require('utils')
require('unittests')

params = {
  batch_size=20,
  seq_length=20,
  layers=2,
  decay=2,
  rnn_size=200,
  dropout=0,
  init_weight=0.1,
  lr=1,
  input_dim=100,
  max_steps=200,
  max_grad_norm=5,
  rows = 20, -- mem
  cols = 20 -- mem
}

transformer("testprogram.lua")
print('Completed program source transformation ...\n\n')
local file = io.open("tmp/modsrc.lua", "w")
file:write(MODIFIED_SOURCE)
file:close()

require('tmp/modsrc.lua')
-- program(params.batch_size, torch.rand(params.batch_size,1,10))


local function main()
  model = setup()
  reset_state()

  local step = 0
  local epoch = 0
  local total_cases = 0
  local beginning_time = torch.tic()
  local start_time = torch.tic()
  print("Starting training.")

  engine_reset()
  program(params.batch_size, torch.rand(params.batch_size,1,10))

  while step < params.max_steps do
    engine_reset() -- reset internal memory after each execution as well as other pointers
    program(params.batch_size, torch.rand(params.batch_size,1,10))
    -- local perf = fp(state_train)
    -- bp(state_train)
    print(EXTERNAL_IDS, CMD_NUM)
    print('Stepping ...')
    step = step + 1
    -- if math.fmod(step,2) == 0 then
    --   print('epoch = ' .. g_f3(epoch) ..
    --         ', train perp. = ' .. g_f3(torch.exp(perf)) ..
    --         ', dw:norm() = ' .. g_f3(model.norm_dw) ..
    --         ', lr = ' ..  params.lr)
 
    --   eval("Validation", state_valid)
    -- end

    -- if math.fmod(step,50) then --learning rate decay
    --   params.lr = params.lr / params.decay
    -- end

    -- if step % 33 == 0 then
    --   cutorch.synchronize()
    --   collectgarbage()
    -- end
  end

  -- eval("Test", state_test)
  -- print("Training is over.")
end

main()
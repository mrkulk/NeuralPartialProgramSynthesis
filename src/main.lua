-- Tejas D Kulkarni (tejask@mit.edu)

require('cunn')
require('nngraph')
require('base')
require('srctransform')
require('model')
require('utils')
require('unittests')

params = {
  batch_size=5,
  seq_length=20,
  layers=2,
  decay=2,
  rnn_size=200,
  dropout=0,
  init_weight=0.1,
  lr=1e-3,
  input_dim=100,
  max_steps=20000,
  max_grad_norm=5,--5,
  rows = 20, -- mem
  cols = 10, -- mem,
  MEM_gscale = 1
}

transformer("testprogram.lua")
print('Completed program source transformation ...\n\n')
local file = io.open("tmp/modsrc.lua", "w")
file:write(MODIFIED_SOURCE)
file:close()

require('tmp/modsrc.lua')
-- program(params.batch_size, torch.rand(params.batch_size,1,10))

local function extract_externals(cache_eids)
  for i=1,#cache_eids do
    EXTERNAL_CACHED_VALUES[cache_eids[i]] = model.write_val[cache_eids[i]]:float()
  end
end


local function gt_program(BSIZE, args)
  a=torch.zeros(BSIZE,1,10)
  fac = 30
  mult = 5
  dummy = fac * 4
  for indx=1,10 do
    a[{{},{1,1},{indx,indx}}]  = args[{{},{1,1},{indx,indx}}]*fac + args[{{},{1,1},{1,1}}]*mult
  end
  return a
end



local function main()
  local step = 0
  local epoch = 0
  local total_cases = 0
  local beginning_time = torch.tic()
  local start_time = torch.tic()
  print("Starting training.")

  engine_reset()
  program(params.batch_size, torch.rand(params.batch_size,1,10))
  params.seq_length = CMD_NUM
  local cache_eids = EXTERNAL_IDS

  model = setup()
  reset_state()

  state_fake = {data = load_fakedata()}

  while step < params.max_steps do
    engine_reset() -- reset internal memory after each execution as well as other pointers
    fp("fake", state_fake) -- just to get outputs
    extract_externals(cache_eids)
    -- Now holes in the program have been filled by neural network. Execute program.
    local train_data = torch.rand(params.batch_size,1,10)
    program(params.batch_size, train_data)
    local program_out = gt_program(params.batch_size, train_data)
    local predicted_output, target_output, total_err = fp("real", nil, program_out)
    bp("real", nil, program_out)
    
    print('Stepping ...')
    step = step + 1

    local score = (torch.pow(predicted_output - target_output, 2):sum())/(params.batch_size)

    if math.fmod(step,2) == 0 then
      print('epoch = ' .. g_f3(epoch) ..
            ', MSE = ' .. score ..
            ', dw:norm() = ' .. g_f3(model.norm_dw) ..
            ', lr = ' ..  params.lr)
      print('TARGET:\n', target_output[1][2])
      print('PREDICTED:', predicted_output[1][2])
      -- eval("Validation", state_valid)
    end

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
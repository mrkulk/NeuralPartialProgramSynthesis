-- Tejas D Kulkarni (tejask@mit.edu)

require('cunn')
require('nngraph')
require('base')
require('srctransform')
require('model')
require('utils')

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

local state_train, state_valid, state_test
local function load_fakedata()
  return {
    true_read_key = torch.rand(params.seq_length, params.batch_size, params.rows, params.cols):cuda(),
    true_read_val = torch.rand(params.seq_length, params.batch_size, params.rows, params.cols):cuda(),
    true_write_key = torch.rand(params.seq_length, params.batch_size, params.rows, params.cols):cuda(),
    true_write_val = torch.rand(params.seq_length, params.batch_size, params.rows, params.cols):cuda(),
    true_write_erase = torch.rand(params.seq_length, params.batch_size, params.rows, params.cols):cuda() 
  }
end

local function main()
  state_train = {data = load_fakedata()}
  state_valid =  {data=load_fakedata()}
  state_test =  {data=load_fakedata}
  model = setup()
  reset_state()

  local step = 0
  local epoch = 0
  local total_cases = 0
  local beginning_time = torch.tic()
  local start_time = torch.tic()
  print("Starting training.")

  while step < params.max_steps do
    local perf = fp(state_train)
    bp(state_train)
    step = step + 1
    if math.fmod(step,2) == 0 then
      print('epoch = ' .. g_f3(epoch) ..
            ', train perp. = ' .. g_f3(torch.exp(perf)) ..
            ', dw:norm() = ' .. g_f3(model.norm_dw) ..
            ', lr = ' ..  params.lr)
 
      eval("Validation", state_valid)
    end

    if math.fmod(step,50) then --learning rate decay
      params.lr = params.lr / params.decay
    end

    if step % 33 == 0 then
      cutorch.synchronize()
      collectgarbage()
    end
  end

  eval("Test", state_test)
  print("Training is over.")
end

main()


local function local_coretest()
  core_network = create_network()
  core_network:cuda()
  -- print(
  --   core_network:forward({
  --     torch.zeros(params.batch_size, params.rnn_size):cuda(),
  --     torch.zeros(params.batch_size):cuda(),
  --     {torch.zeros(params.batch_size, params.rnn_size):cuda(), torch.zeros(params.batch_size, params.rnn_size):cuda(), torch.zeros(params.batch_size, params.rnn_size):cuda(), torch.zeros(params.batch_size, params.rnn_size):cuda() },
  --     torch.zeros(params.batch_size, params.rows, params.cols):cuda()
  --   })
  -- )  
  print(params)
  local ret = core_network:forward({
    {torch.zeros(params.batch_size, params.rnn_size):cuda(), torch.zeros(params.batch_size, params.rnn_size):cuda(), torch.zeros(params.batch_size, params.rnn_size):cuda(), torch.zeros(params.batch_size, params.rnn_size):cuda() },
    torch.zeros(params.batch_size, params.rows, params.cols):cuda(),
    torch.rand(params.batch_size, params.rows, params.cols):cuda(),
    torch.rand(params.batch_size, params.rows, params.cols):cuda(),
    torch.rand(params.batch_size, params.rows, params.cols):cuda(),
    torch.rand(params.batch_size, params.rows, params.cols):cuda(),
    torch.rand(params.batch_size, params.rows, params.cols):cuda(),
    torch.rand(params.batch_size, params.rows, params.cols):cuda(),
    torch.rand(params.batch_size, params.rows, params.cols):cuda(),
    torch.rand(params.batch_size, params.rows, params.cols):cuda(),
    torch.rand(params.batch_size, params.rows, params.cols):cuda(),
    torch.rand(params.batch_size, params.rows, params.cols):cuda()
    })

  local ret = core_network:backward(
  {
    {torch.zeros(params.batch_size, params.rnn_size):cuda(), torch.zeros(params.batch_size, params.rnn_size):cuda(), torch.zeros(params.batch_size, params.rnn_size):cuda(), torch.zeros(params.batch_size, params.rnn_size):cuda() },
    torch.zeros(params.batch_size, params.rows, params.cols):cuda(),
    torch.rand(params.batch_size, params.rows, params.cols):cuda(),
    torch.rand(params.batch_size, params.rows, params.cols):cuda(),
    torch.rand(params.batch_size, params.rows, params.cols):cuda(),
    torch.rand(params.batch_size, params.rows, params.cols):cuda(),
    torch.rand(params.batch_size, params.rows, params.cols):cuda(),
    torch.rand(params.batch_size, params.rows, params.cols):cuda(),
    torch.rand(params.batch_size, params.rows, params.cols):cuda(),
    torch.rand(params.batch_size, params.rows, params.cols):cuda(),
    torch.rand(params.batch_size, params.rows, params.cols):cuda(),
    torch.rand(params.batch_size, params.rows, params.cols):cuda()
  },
  {
    torch.zeros(1):cuda(),torch.zeros(1):cuda(),torch.zeros(1):cuda(),torch.zeros(1):cuda(),torch.zeros(1):cuda(),
    {torch.zeros(params.batch_size, params.rnn_size):cuda(),torch.zeros(params.batch_size, params.rnn_size):cuda(),
    torch.zeros(params.batch_size, params.rnn_size):cuda(),torch.zeros(params.batch_size, params.rnn_size):cuda()
    },
    torch.zeros(params.batch_size, params.rows, params.cols):cuda(),
    torch.zeros(params.batch_size, params.rows, params.cols):cuda(),
    torch.zeros(params.batch_size, params.rows, params.cols):cuda(),
    torch.zeros(params.batch_size, params.rows, params.cols):cuda(),
    torch.zeros(params.batch_size, params.rows, params.cols):cuda(),
    torch.zeros(params.batch_size, params.rows, params.cols):cuda()
  })
  print(ret)
end
-- local_coretest()
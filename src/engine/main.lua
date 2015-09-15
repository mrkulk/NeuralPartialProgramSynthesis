-- Tejas D Kulkarni (tejask@mit.edu)

require('cunn')
require('nngraph')
require('base')
require('srctransform')

-- Trains 1h and gives test 115 perplexity.
params = {batch_size=20,
                seq_length=20,
                layers=2,
                decay=2,
                rnn_size=200,
                dropout=0,
                init_weight=0.1,
                lr=1,
                input_dim=100,
                max_epoch=4,
                max_max_epoch=13,
                max_grad_norm=5}

require('model')
require('runner')
require('utils')


local state_train, state_valid, state_test
local model = {}
local paramx, paramdx

local function main()
  g_init_gpu(arg)
  state_train = {data=transfer_data(ptb.traindataset(params.batch_size))}
  state_valid =  {data=transfer_data(ptb.validdataset(params.batch_size))}
  state_test =  {data=transfer_data(ptb.testdataset(params.batch_size))}
  print("Network parameters:")
  print(params)
  local states = {state_train, state_valid, state_test}
  for _, state in pairs(states) do
    reset_state(state)
  end
  setup()
  local step = 0
  local epoch = 0
  local total_cases = 0
  local beginning_time = torch.tic()
  local start_time = torch.tic()
  print("Starting training.")
  local words_per_step = params.seq_length * params.batch_size
  local epoch_size = torch.floor(state_train.data:size(1) / params.seq_length)
  local perps
  while epoch < params.max_max_epoch do
    local perp = fp(state_train)
    if perps == nil then
      perps = torch.zeros(epoch_size):add(perp)
    end
    perps[step % epoch_size + 1] = perp
    step = step + 1
    bp(state_train)
    total_cases = total_cases + params.seq_length * params.batch_size
    epoch = step / epoch_size
    if step % torch.round(epoch_size / 10) == 10 then
      local wps = torch.floor(total_cases / torch.toc(start_time))
      local since_beginning = g_d(torch.toc(beginning_time) / 60)
      print('epoch = ' .. g_f3(epoch) ..
            ', train perp. = ' .. g_f3(torch.exp(perps:mean())) ..
            ', wps = ' .. wps ..
            ', dw:norm() = ' .. g_f3(model.norm_dw) ..
            ', lr = ' ..  g_f3(params.lr) ..
            ', since beginning = ' .. since_beginning .. ' mins.')
    end
    if step % epoch_size == 0 then
      run_valid()
      if epoch > params.max_epoch then
          params.lr = params.lr / params.decay
      end
    end
    if step % 33 == 0 then
      cutorch.synchronize()
      collectgarbage()
    end
  end
  run_test()
  print("Training is over.")
end

-- main()

core_network = create_network()
print(params)
print(
  core_network:forward({
    torch.zeros(params.batch_size, params.rnn_size):cuda(),
    torch.zeros(params.batch_size):cuda(),
    {torch.zeros(params.batch_size, params.rnn_size):cuda(), torch.zeros(params.batch_size, params.rnn_size):cuda(), torch.zeros(params.batch_size, params.rnn_size):cuda(), torch.zeros(params.batch_size, params.rnn_size):cuda() }
  })
)
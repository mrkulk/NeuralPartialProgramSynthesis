require 'Normalize'

function transfer_data(x)
  return x:cuda()
end


function lstm(x, prev_c, prev_h)
  -- Calculate all four gates in one go
  local i2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(x)
  local h2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(prev_h)
  local gates = nn.CAddTable()({i2h, h2h})
  
  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates =  nn.Reshape(4,params.rnn_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)
  
  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  return next_c, next_h
end

function create_network()
  local prev_s           = nn.Identity()() -- LSTM
  local MEM              = nn.Identity()() -- memory
  local prev_read_key    = nn.Identity()() -- " "
  local prev_read_val    = nn.Identity()() 
  local prev_write_key   = nn.Identity()()
  local prev_write_val   = nn.Identity()()

  -- targets whenever available (specified fragments of the program)
  local true_read_key    = nn.Identity()()
  local true_read_val    = nn.Identity()() 
  local true_write_key   = nn.Identity()()
  local true_write_val   = nn.Identity()()
  local true_write_erase = nn.Identity()()

  local head_dim = params.rows * params.cols
  local reshape_prev_read_key    = nn.Reshape(params.batch_size, head_dim)(prev_read_key)
  local reshape_prev_read_val    = nn.Reshape(params.batch_size, head_dim)(prev_read_val)
  local reshape_prev_write_key   = nn.Reshape(params.batch_size, head_dim)(prev_write_key)
  local reshape_prev_write_val   = nn.Reshape(params.batch_size, head_dim)(prev_write_val)

  local concat_x = nn.JoinTable(1)({reshape_prev_read_key, reshape_prev_read_val, reshape_prev_write_key , reshape_prev_write_val })


  local module           = nn.gModule({prev_s, MEM, prev_read_key, prev_read_val, prev_write_key, prev_write_val,
                                                    true_read_key, true_read_val, true_write_key, true_write_val, true_write_erase},
                                      { reshape_prev_read_key, 
                                      prev_s, prev_read_key, prev_read_val, prev_write_key, prev_write_val, 
                                      true_read_key, true_read_val, true_write_key, true_write_val, true_write_erase, MEM })
  
  return module

  -- local remapped_x = nn.Linear(4*head_dim, params.rnn_size)(concat_x)

  -- local i                = {[0] = nn.Identity()(remapped_x)}

  -- local next_s           = {}
  -- local split         = {prev_s:split(2 * params.layers)}
  -- for layer_idx = 1, params.layers do
  --   local prev_c         = split[2 * layer_idx - 1]
  --   local prev_h         = split[2 * layer_idx]
  --   local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
  --   local next_c, next_h = lstm(dropped, prev_c, prev_h)
  --   table.insert(next_s, next_c)
  --   table.insert(next_s, next_h)
  --   i[layer_idx] = next_h
  -- end

  -- -- local h2y              = nn.Linear(params.rnn_size, params.input_dim)
  -- -- local dropped          = nn.Dropout(params.dropout)(i[params.layers])
  -- -- local pred             = nn.LogSoftMax()(h2y(dropped))

  -- ---------------------- Memory Ops ------------------
  -- local read_channel = nn.ReLU()(nn.Linear(params.rnn_size,params.rnn_size)(i[params.layers]))
  -- local read_key = nn.Normalize()(nn.SoftMax()(nn.Linear(params.rnn_size, params.rows*params.cols)(read_channel)))
  -- local read_val = nn.DotProduct()({MEM, read_key})

  -- local write_channel = nn.ReLU()(nn.Linear(params.rnn_size,params.rnn_size)(i[params.layers]))
  -- local write_key = nn.Normalize()(nn.SoftMax()(nn.Linear(params.rnn_size, params.rows*params.cols)(write_channel)))
  -- local write_val = nn.Linear(params.rnn_size, params.rows*params.cols)(write_channel) 
  -- local write_erase = nn.Linear(params.rnn_size, params.rows*params.cols)(write_channel) 

  -- local erase_val_interim = nn.DotProduct()({write_key, write_erase})
  -- local erase_val = nn.AddConstant(1)(nn.MulConstant(-1)(erase_val_interim))
  -- local erase_MEM = nn.DotProduct()({MEM, erase_val})

  -- local add_val_interim = nn.DotProduct()({write_key, write_val})
  -- local add_MEM = nn.CAddTable()({erase_MEM, add_val_interim})

  -- local err_rk = nn.MSECriterion()({read_key, true_read_key})
  -- local err_rv = nn.MSECriterion()({read_val, true_read_val})
  -- local err_wk = nn.MSECriterion()({write_key, true_write_key})
  -- local err_wv = nn.MSECriterion()({write_val, true_write_val})
  -- local err_we = nn.MSECriterion()({write_erase, true_write_erase})

  -- local module           = nn.gModule({prev_s, MEM, prev_read_key, prev_read_val, prev_write_key, prev_write_val,
  --                                                   true_read_key, true_read_val, true_write_key, true_write_val, true_write_erase},
  --                                     {err_rk, err_rv, err_wk, err_wv, err_we,
  --                                     nn.Identity()(next_s), add_MEM, read_key, read_val, write_key, write_val, write_erase})
  
  -- return module
end



function setup()
  print("Creating a RNN LSTM network.")
  local core_network = create_network()
  core_network:cuda()
  core_network:getParameters():uniform(-params.init_weight, params.init_weight)
  paramx, paramdx = core_network:getParameters()
  
  local model = {}
  model.s = {}
  model.ds = {}
  model.start_s = {}
  for j = 0, params.seq_length do
    model.s[j] = {}
    for d = 1, 2 * params.layers do
      model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
  end
  for d = 1, 2 * params.layers do
    model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end
  model.core_network = core_network
  model.rnns = g_cloneManyTimes(core_network, params.seq_length)
  model.norm_dw = 0
  model.err = transfer_data(torch.zeros(params.seq_length))
  return model
end

function reset_state(state)
  state.pos = 1
  if model ~= nil and model.start_s ~= nil then
    for d = 1, 2 * params.layers do
      model.start_s[d]:zero()
    end
  end
end

function reset_ds()
  for d = 1, #model.ds do
    model.ds[d]:zero()
  end
end

function fp(state)
  g_replace_table(model.s[0], model.start_s)
  if state.pos + params.seq_length > state.data:size(1) then
    reset_state(state)
  end
  for i = 1, params.seq_length do
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    model.err[i], model.s[i] = unpack(model.rnns[i]:forward({x, y, s}))
    state.pos = state.pos + 1
  end
  g_replace_table(model.start_s, model.s[params.seq_length])
  return model.err:mean()
end

function bp(state)
  paramdx:zero()
  reset_ds()
  for i = params.seq_length, 1, -1 do
    state.pos = state.pos - 1
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    local derr = transfer_data(torch.ones(1))
    local tmp = model.rnns[i]:backward({x, y, s},
                                       {derr, model.ds})[3]
    g_replace_table(model.ds, tmp)
    cutorch.synchronize()
  end
  state.pos = state.pos + params.seq_length
  model.norm_dw = paramdx:norm()
  if model.norm_dw > params.max_grad_norm then
    local shrink_factor = params.max_grad_norm / model.norm_dw
    paramdx:mul(shrink_factor)
  end
  paramx:add(paramdx:mul(-params.lr))
end

require 'Normalize'
require 'componentMul'

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
  local prev_write_erase   = nn.Identity()()

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
  local reshape_prev_write_erase = nn.Reshape(params.batch_size, head_dim)(prev_write_erase)

  local concat_x = nn.JoinTable(2)({reshape_prev_read_key, reshape_prev_read_val, reshape_prev_write_key , reshape_prev_write_val, reshape_prev_write_erase })
  local remapped_x = nn.Linear(5*head_dim, params.rnn_size)(concat_x)

  local i                = {[0] = nn.Identity()(remapped_x)}

  local next_s           = {}
  local split         = {prev_s:split(2 * params.layers)}
  for layer_idx = 1, params.layers do
    local prev_c         = split[2 * layer_idx - 1]
    local prev_h         = split[2 * layer_idx]
    local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
    local next_c, next_h = lstm(dropped, prev_c, prev_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end

  local h2y              = nn.Linear(params.rnn_size, params.input_dim)
  local dropped          = nn.Dropout(params.dropout)(i[params.layers])
  local pred             = nn.LogSoftMax()(h2y(dropped))

  ---------------------- Memory Ops ------------------
  local read_channel = nn.ReLU()(nn.Linear(params.rnn_size,params.rnn_size)(i[params.layers]))
  local read_key = nn.Reshape(params.rows,params.cols)(nn.Normalize()(nn.SoftMax()(nn.Linear(params.rnn_size, params.rows*params.cols)(read_channel))))
  local read_val = nn.componentMul()({MEM, read_key})

  local write_channel = nn.ReLU()(nn.Linear(params.rnn_size,params.rnn_size)(i[params.layers]))
  local write_key = nn.Reshape(params.rows,params.cols)(nn.Normalize()(nn.SoftMax()(nn.Linear(params.rnn_size, params.rows*params.cols)(write_channel))))
  local write_val = nn.Reshape(params.rows,params.cols)(nn.Linear(params.rnn_size, params.rows*params.cols)(write_channel)) 
  local write_erase = nn.Reshape(params.rows,params.cols)(nn.Linear(params.rnn_size, params.rows*params.cols)(write_channel)) 

  local erase_val_interim = nn.componentMul()({write_key, write_erase})
  local erase_val = nn.AddConstant(1)(nn.MulConstant(-1)(erase_val_interim))
  local erase_MEM = nn.componentMul()({MEM, erase_val})

  local add_val_interim = nn.componentMul()({write_key, write_val})
  local add_MEM = nn.CAddTable()({erase_MEM, add_val_interim})

  local err_rk = nn.MSECriterion()({read_key, nn.Reshape(params.rows * params.cols)(true_read_key)})
  local err_rv = nn.MSECriterion()({read_val, nn.Reshape(params.rows * params.cols)(true_read_val)})
  local err_wk = nn.MSECriterion()({write_key, nn.Reshape(params.rows * params.cols)(true_write_key)})
  local err_wv = nn.MSECriterion()({write_val, nn.Reshape(params.rows * params.cols)(true_write_val)})
  local err_we = nn.MSECriterion()({write_erase, nn.Reshape(params.rows * params.cols)(true_write_erase)})

  local module           = nn.gModule({prev_s, MEM, prev_read_key, prev_read_val, prev_write_key, prev_write_val, prev_write_erase,
                                                    true_read_key, true_read_val, true_write_key, true_write_val, true_write_erase},
                                      {err_rk, err_rv, err_wk, err_wv, err_we,
                                      nn.Identity()(next_s), add_MEM, read_key, read_val, write_key, write_val, write_erase})
  
  return module

  -- local module           = nn.gModule({prev_s, MEM, prev_read_key, prev_read_val, prev_write_key, prev_write_val,
  --                                                   true_read_key, true_read_val, true_write_key, true_write_val, true_write_erase},
  --                                     { err_rk,
  --                                     prev_s, prev_read_key, prev_read_val, prev_write_key, prev_write_val, 
  --                                     true_read_key, true_read_val, true_write_key, true_write_val, true_write_erase, MEM })
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
  model.MEM = {}; model.read_key = {}; model.read_val = {}; model.write_key = {}; model.write_val = {}; model.write_erase = {}

  for j = 0, params.seq_length do
    model.s[j] = {}
    for d = 1, 2 * params.layers do
      model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
    
    model.MEM[j] = transfer_data(torch.zeros(params.batch_size, params.rows, params.cols))
    model.read_key[j] = transfer_data(torch.zeros(params.batch_size, params.rows, params.cols))
    model.read_val[j] = transfer_data(torch.zeros(params.batch_size, params.rows, params.cols))
    model.write_key[j] = transfer_data(torch.zeros(params.batch_size, params.rows, params.cols))
    model.write_val[j] = transfer_data(torch.zeros(params.batch_size, params.rows, params.cols))
    model.write_erase[j] = transfer_data(torch.zeros(params.batch_size, params.rows, params.cols))
  end

  model.ds_MEM = transfer_data(torch.zeros( params.batch_size, params.rows, params.cols))
  model.ds_read_key = transfer_data(torch.zeros( params.batch_size, params.rows, params.cols))
  model.ds_read_val = transfer_data(torch.zeros( params.batch_size, params.rows, params.cols))
  model.ds_write_key = transfer_data(torch.zeros( params.batch_size, params.rows, params.cols))
  model.ds_write_val = transfer_data(torch.zeros( params.batch_size, params.rows, params.cols))
  model.ds_write_erase = transfer_data(torch.zeros( params.batch_size, params.rows, params.cols))

  for d = 1, 2 * params.layers do
    model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end

  model.core_network = core_network
  model.rnns = g_cloneManyTimes(core_network, params.seq_length)
  model.norm_dw = 0
  model.err_rk = transfer_data(torch.zeros(params.seq_length)) 
  model.err_rv = transfer_data(torch.zeros(params.seq_length))
  model.err_wk = transfer_data(torch.zeros(params.seq_length))
  model.err_wv = transfer_data(torch.zeros(params.seq_length))
  model.err_we = transfer_data(torch.zeros(params.seq_length))

  return model
end

function reset_state()
  for j = 0, params.seq_length do
    for d = 1, 2 * params.layers do
      model.s[j][d]:zero()
    end
    model.MEM[j]:zero()
    model.read_key[j]:zero()
    model.read_val[j]:zero()
    model.write_key[j]:zero()
    model.write_val[j]:zero()
    model.write_erase[j]:zero()
  end
end

function reset_ds()
  for d = 1, #model.ds do
    model.ds[d]:zero()
  end
  model.ds_MEM:zero()
  model.ds_read_key:zero()
  model.ds_read_val:zero()
  model.ds_write_key:zero()
  model.ds_write_val:zero()
  model.ds_write_erase:zero()
end

function fp(state)
  reset_state()

  for i = 1, params.seq_length do
    model.err_rk[i], model.err_rv[i], model.err_wk[i], model.err_wv[i], model.err_we[i], model.s[i], 
    model.MEM[i], model.read_key[i], model.read_val[i], model.write_key[i], model.write_val[i], model.write_erase[i] = unpack(model.rnns[i]:forward({
      model.s[i-1], model.MEM[i-1], model.read_key[i-1], model.read_val[i-1], model.write_key[i-1], model.write_val[i-1], model.write_erase[i-1],
      state.true_read_key[i], state.true_read_val[i], state.true_write_key[i], state.true_write_val[i], state.true_write_erase[i]
    }))
  end
  g_replace_table(model.start_s, model.s[params.seq_length])
  return model.err_rk:mean() + model.err_rv:mean() + model.err_wk:mean() + model.err_wv:mean() + model.err_we:mean()
end

function bp(state)
  paramdx:zero()
  reset_ds()
  for i = params.seq_length, 1, -1 do
    local derr_rk = transfer_data(torch.ones(1)); local derr_rv = transfer_data(torch.ones(1));
    local derr_wk = transfer_data(torch.ones(1)); local derr_wv = transfer_data(torch.ones(1));
    local derr_we = transfer_data(torch.ones(1))

    local tmp_s, tmp_MEM, tmp_rk, tmp_rv, tmp_wk, tmp_wv, tmp_we, t1,t2,t3,t4,t5  = unpack(model.rnns[i]:backward(
    {
      model.s[i-1], model.MEM[i-1], model.read_key[i-1], model.read_val[i-1], model.write_key[i-1], model.write_val[i-1], model.write_erase[i-1],
      state.true_read_key[i], state.true_read_val[i], state.true_write_key[i], state.true_write_val[i], state.true_write_erase[i]
    },
    {
      derr_rk, derr_rv, derr_wk, derr_wv, derr_we,
      model.ds, model.ds_MEM, model.ds_read_key, model.ds_read_val, model.ds_write_key, model.ds_write_val, model.ds_write_erase
    }
    ))

    g_replace_table(model.ds, tmp_s)
    g_replace_table(model.ds_MEM, tmp_MEM)
    g_replace_table(model.ds_read_key, tmp_rk)
    g_replace_table(model.ds_read_val, tmo_rv)
    g_replace_table(model.ds_write_key, tmp_wk)
    g_replace_table(model.ds_write_val, tmp_wv)
    g_replace_table(model.ds_write_erase, tmp_we)

    cutorch.synchronize()
  end
  model.norm_dw = paramdx:norm()
  if model.norm_dw > params.max_grad_norm then
    local shrink_factor = params.max_grad_norm / model.norm_dw
    paramdx:mul(shrink_factor)
  end
  paramx:add(paramdx:mul(-params.lr))
end


function eval(mode, state)
  reset_state()
  g_disable_dropout(model.rnns)
  local perp = 0
  for i = 1, params.seq_length do
    perp = perp + fp(state)
  end
  print(mode .. " set perplexity : " .. g_f3(torch.exp(perp / len)))
  g_enable_dropout(model.rnns)
end

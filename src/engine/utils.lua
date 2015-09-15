
function transfer_data(x)
  return x:cuda()
end

function setup()
  print("Creating a RNN LSTM network.")
  local core_network = create_network()
  paramx, paramdx = core_network:getParameters()
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




package.path = 'lang/?.lua;?.lua;' .. package.path
local lex_setup = require("lang.lexer")
local reader = require("lang.reader")
local parse = require('lang.parser')
local ast = require('lang.lua-ast').New()
generator = require('lang.luacode-generator')

function split(pString, pPattern)
   local Table = {}  -- NOTE: use {n = 0} in Lua-5.0
   local fpat = "(.-)" .. pPattern
   local last_end = 1
   local s, e, cap = pString:find(fpat, 1)
   while s do
      if s ~= 1 or cap ~= "" then
     table.insert(Table,cap)
      end
      last_end = e+1
      s, e, cap = pString:find(fpat, last_end)
   end
   if last_end <= #pString then
      cap = pString:sub(last_end)
      table.insert(Table, cap)
   end
   return Table
end

function deepcopy_table(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[deepcopy_table(orig_key)] = deepcopy_table(orig_value)
        end
        setmetatable(copy, deepcopy_table(getmetatable(orig)))
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

local function lang_toolkit_error(msg)
  if string.sub(msg, 1, 9) == "LLT-ERROR" then
    return false, "luajit-lang-toolkit: " .. string.sub(msg, 10)
  else
    error(msg)
  end
end

function ASTtoCODE(tree)
  local success, luacode = pcall(generator, tree)
  if not success then
    return lang_toolkit_error(luacode)
  else
    return luacode
  end
end

function getAST(filename)
  -- filename = "testprogram.lua"
  local ls = lex_setup(reader.file(filename), filename)
  local parse_success, tree = pcall(parse, ast, ls)
  if not parse_success then
    return lang_toolkit_error(tree)
  end
    -- print(tree.body[1].body[2])
    return tree
end

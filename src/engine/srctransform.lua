require 'nn'
require 'utils'
require 'cons'
JSON = (loadfile "JSON.lua")() 

DEBUG = true
local function printf(str)
  if DEBUG then
    print(str)
  end
end

MODIFIED_SOURCE = "require 'engine' \n"

local function neural_interpret(line, lvars, rvars)
  MODIFIED_SOURCE = MODIFIED_SOURCE .. line .. "\n"

  ---- lhs -----
  local linfo
  if lvars[1].kind == "Identifier" then
    -- linfo={"Identifier", lvars[1].name}
    MODIFIED_SOURCE = MODIFIED_SOURCE .. '_nreg({' .. "'Identifier'," .. "'" ..  lvars[1].name .. "'," .. lvars[1].name .. '},'
  elseif lvars[1].kind == "MemberExpression" then
    -- linfo={"MemberExpression", lvars[1].object.name, lvars[1].property.name}
    MODIFIED_SOURCE = MODIFIED_SOURCE .. '_nreg({' .. "'MemberExpression'," .. "'" ..  lvars[1].object.name .. "'," .. lvars[1].object.name .. "," .. lvars[1].property.name .. '},'
  else
    print('[neural_interpret] ERROR: Not Implemented!')
    exit()
  end
  -- print(linfo)

  ---- rhs ----
  MODIFIED_SOURCE = MODIFIED_SOURCE .. "{"
  local rinfo  = {}
  for i=1,#rvars do
    local var = rvars[i]
    if var.kind == "CallExpression" then --array inits
      -- rinfo[i] = {"CallExpression", var.callee.property.name, var.arguments[1].value}
      MODIFIED_SOURCE = MODIFIED_SOURCE .. '{' .. "'CallExpression'," .. "'" ..  var.callee.property.name .. "'," .. var.callee.property.name .. "," ..  var.arguments[1].value .. '},'
    elseif var.kind == "MemberExpression" then
      if var.property.kind == "Literal" then
        -- rinfo[i] = {"MemberExpression", var.object.name, var.property.value}
        MODIFIED_SOURCE = MODIFIED_SOURCE .. '{' .. "'Literal'," .. "'" ..  var.object.name .. "'," .. var.object.name .. "," ..  var.property.value .. '},'
      elseif var.property.kind == "Identifier" then
        -- rinfo[i] = {"MemberExpression", var.object.name, var.property.name}
        MODIFIED_SOURCE = MODIFIED_SOURCE .. '{'  .. "'Identifier'," .. "'" ..  var.object.name .. "'," .. var.object.name .. "," .. var.property.name .. '},'
      end
    elseif var.kind == "Literal" then
      -- rinfo[i] = {"Literal", var.value}
      MODIFIED_SOURCE = MODIFIED_SOURCE .. '{' .. "'Literal'," .. "'" ..  var.value .. "'," .. var.value .. '},'
    elseif var.kind == "Identifier" then
      -- rinfo[i] = {"Identifier", var.name}
      MODIFIED_SOURCE = MODIFIED_SOURCE .. '{' ..  "'Identifier'," .. "'" ..  var.name .. "'," .. var.name .. '},'
    end
    -- print(rinfo)
  end

  MODIFIED_SOURCE = MODIFIED_SOURCE .. "})\n"
end


local function recurse(ast, vars)
  if ast == nil then return end
  if ast.left == nil and ast.right == nil then --only look at leafs
    vars[#vars+1] = ast
  end
  recurse(ast.left, vars)
  recurse(ast.right, vars)
end


local function parse(line)
  skip = false

  for i=1,#reserved_keywords do 
    if string.match(line, reserved_keywords[i]) ~= nil then
      skip = true
    end
  end
  if skip == false then
    if string.match(line, "return") ~= nil then
      -- print('Not Implemented')
      local variable = split(line, " ")
      MODIFIED_SOURCE = MODIFIED_SOURCE .. "_nreg(\'return\'" .. ",\'" .. variable[#variable] .. "\')\n"
      MODIFIED_SOURCE = MODIFIED_SOURCE .. line .. "\n"
    else
      local file = io.open("tmp/tmp.txt", "w")
      file:write(line)
      file:close()
      local ast = getAST("tmp/tmp.txt")
      ast = ast.body[1]
      local last = ast.left[1]; local rast = ast.right[1]
      local lvars = {}; recurse(last, lvars)
      local rvars = {}; recurse(rast, rvars)
      neural_interpret(line, lvars, rvars)
    end
  else
    MODIFIED_SOURCE = MODIFIED_SOURCE .. line .. "\n"
  end
end


function transformer(filename)
  local file = io.open(filename, "r")
  io.input(file)
  local cnt=1
  while true do
    local line = io.read()
    if line == nil then
      break
    end
    parse(line)
    cnt = cnt + 1
  end 
end

local function unittest()
  transformer("testprogram.lua")
  print('Completed program source transformation ...\n\n')
  print(MODIFIED_SOURCE)
  local file = io.open("tmp/modsrc.lua", "w")
  file:write(MODIFIED_SOURCE)
  file:close()
  print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
  
  require('tmp/modsrc.lua')
  program(torch.rand(10))
end

unittest()

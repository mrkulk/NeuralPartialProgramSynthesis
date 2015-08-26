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

MODIFIED_SOURCE = "require 'nn' \n"

local function neural_interpret(line, lvars, rvars)
  -- print(line)
  -- print(lvars)
  -- print(rvars)
  ---- lhs -----
  local linfo
  if lvars[1].kind == "Identifier" then
    linfo={"Identifier", lvars[1].name}
  elseif lvars[1].kind == "MemberExpression" then
    linfo={"MemberExpression", lvars[1].object.name, lvars[1].property.name}
  else
    print('[neural_interpret] ERROR: Not Implemented!')
    exit()
  end
  -- print(linfo)

  ---- rhs ----
  local rinfo  = {}
  for i=1,#rvars do
    local var = rvars[i]
    if var.kind == "CallExpression" then --array inits
      rinfo[i] = {"CallExpression", var.callee.property.name, var.arguments[1].value}
    elseif var.kind == "MemberExpression" then
      if var.property.kind == "Literal" then
        rinfo[i] = {"MemberExpression-Literal", var.object.name, var.property.value}
      elseif var.property.kind == "Identifier" then
        rinfo[i] = {"MemberExpression-Identifier", var.object.name, var.property.name}
      end
    elseif var.kind == "Literal" then
      rinfo[i] = {"Literal", var.value}
    elseif var.kind == "Identifier" then
      rinfo[i] = {"Identifier", var.name}
    end
    -- print(rinfo)
  end

  linfo_serialized = JSON:encode(linfo)
  rinfo_serialized = JSON:encode(rinfo)
  MODIFIED_SOURCE = MODIFIED_SOURCE .. line .. "\n"
  MODIFIED_SOURCE = MODIFIED_SOURCE .. "_nreg(" .. linfo_serialized .. "," .. rinfo_serialized .. ")\n"
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
  printf("-------------------------------\n")
  skip = false

  for i=1,#reserved_keywords do 
    if string.match(line, reserved_keywords[i]) ~= nil then
      skip = true
    end
  end
  if skip == false then
    if string.match(line, "return") ~= nil then
      print('Not Implemented')
    else
      local file = io.open("tmp.txt", "w")
      file:write(line)
      file:close()
      local ast = getAST("tmp.txt")
      ast = ast.body[1]
      local last = ast.left[1]; local rast = ast.right[1]
      lvars = {}; recurse(last, lvars)
      rvars = {}; recurse(rast, rvars)
      neural_interpret(line, lvars, rvars)
    end
  else
    MODIFIED_SOURCE = MODIFIED_SOURCE .. line .. "\n"
  end
end


local function transformer(filename)
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


transformer("testprogram.lua")
print(MODIFIED_SOURCE)
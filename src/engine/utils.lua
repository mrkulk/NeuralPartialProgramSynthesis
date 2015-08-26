
package.path = 'lang/?.lua;?.lua;' .. package.path
local lex_setup = require("lang.lexer")
local reader = require("lang.reader")
local parse = require('lang.parser')
local ast = require('lang.lua-ast').New()
generator = require('lang.luacode-generator')


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

-- print(getAST("testprogram.lua"))
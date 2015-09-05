
rows = 20
cols = 10

VARLIST = {}
MEMCNTR = 1



function syncMemory(lexp, rexp)
  print('##############')
  print(lexp)
  print(rexp) 
  print('============')
  local lhs_cmds, rhs_cmds
  rhs_cmds = {}
  reads = {}; writes={}
  -- process LHS -- 
  local vname, vind, val, vrows, start_col,end_col
  if lexp[1] == "Identifier" then
    vname = lexp[2]; val = lexp[3]
    if type(val) == type(1) then
      vrows = 1; start_col=1;end_col = cols
    else
      local sz = val:size()
      vrows = sz[1]; start_col=1;end_col = sz[2]
    end
  elseif lexp[1] == "MemberExpression" then
    vname = lexp[2];
    vrows = lexp[4]
    if lexp[4] ~= lexp[5] then
      print('ERROR: currently only supporting 1D along row')
      exit()
    end
    start_col = lexp[6]; end_col = lexp[7]
    val = lexp[3][{{vrows,vrows}, {start_col, end_col}}]
    -- vrows = lexp[4]; vcols = lexp[5]
    -- val = lexp[3][vrows][vcols]
  end
  local INDX = MEMCNTR
  if VARLIST[vname] ~= nil then
    INDX = VARLIST[vname].INDX
  end

  -- update writes for LHS
  local w_rkey = torch.zeros(rows)
  local w_ckey = torch.zeros(cols)
  w_rkey[{{INDX, INDX + vrows - 1}}] = 1
  w_ckey[{{start_col,end_col}}] = 1

  local memory = torch.zeros(rows, cols)

  memory[{{INDX, INDX+ vrows - 1}, {start_col,end_col}}] = val
  lhs_cmds = {
    [1] = {
      cmd = "write",
      rkey = w_rkey:clone(), --row key
      ckey = w_ckey:clone(),  --col key
      val = memory
    }
  }

  if VARLIST[vname] == nil then
    VARLIST[vname] = { rkey = w_rkey, ckey = w_ckey, INDX = MEMCNTR }
  end

  MEMCNTR = MEMCNTR + vrows

  -- process RHS --
  local exp = rexp[1]
  if exp[1] == "CallExpression" or exp[1] == "Literal" then
    if #rexp == 1 then
      return
    end
  end
  for i=1,#rexp do
    local exp = rexp[i]
    if exp[1] == "Identifier" then
      -- make a bunch of reads 
      -- print(VARLIST[exp[2]])
      rhs_cmds[#rhs_cmds + 1] = {
        cmd = "read",
        rkey = VARLIST[exp[2]].rkey:clone(),
        ckey = VARLIST[exp[2]].ckey:clone()
      }
      -- print(rhs_cmds)
    end 
  end
  -- print(lhs_cmds[1].rkey, lhs_cmds[1].ckey)
end

function _nreg(lexp, rexp)
    if lexp == "return" then
        print('TODO: return')
    else
      syncMemory(lexp, rexp)
    end
end
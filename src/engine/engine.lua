
rows = 20
cols = 10

VARLIST = {}
MEMCNTR = 1



function syncMemory(BSIZE, lexp, rexp, mode)
  -- print('##############')
  -- print(lexp)
  -- print(rexp) 

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
      vrows = sz[2]; start_col=1;end_col = sz[3]
    end
  elseif lexp[1] == "MemberExpression" then
    vname = lexp[2];
    vrows = lexp[5]
    if lexp[5] ~= lexp[6] then
      print('ERROR: currently only supporting 1D along row')
      exit()
    end
    start_col = lexp[7]; end_col = lexp[8]
    -- print(lexp[3])
    val = lexp[3][{{},{vrows,vrows}, {start_col, end_col}}]
    -- vrows = lexp[4]; vcols = lexp[5]
    -- val = lexp[3][vrows][vcols]
  end
  local INDX = MEMCNTR
  if VARLIST[vname] ~= nil then
    INDX = VARLIST[vname].INDX
  end

  -- update writes for LHS
  local w_rkey = torch.zeros(BSIZE, rows)
  local w_ckey = torch.zeros(BSIZE, cols)
  w_rkey[{{},{INDX, INDX + vrows - 1}}] = 1
  w_ckey[{{},{start_col,end_col}}] = 1

  local memory = torch.zeros(BSIZE, rows, cols)
  -- print(memory[{{}, {INDX, INDX+ vrows - 1}, {start_col,end_col}}])
  -- print(val)

  memory[{{}, {INDX, INDX+ vrows - 1}, {start_col,end_col}}] = val
  lhs_cmds = {
    [1] = {
      cmd = "write",
      rkey = w_rkey:clone(), --row key
      ckey = w_ckey:clone(),  --col key
      val = memory, 
      mode = mode
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
      return lhs_cmds, rhs_cmds
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
    elseif exp[1] == "MemberExpression" then
      local INDX = VARLIST[exp[2]].INDX
      local w_rkey = torch.zeros(BSIZE, rows)
      local w_ckey = torch.zeros(BSIZE, cols)
      vrows = lexp[5]
      if lexp[5] ~= lexp[6] then
        print('ERROR: currently only supporting 1D along row')
        exit()
      end
      start_col = lexp[7]; end_col = lexp[8]
      w_rkey[{{},{INDX, INDX + vrows - 1}}] = 1
      w_ckey[{{},{start_col,end_col}}] = 1
      
      rhs_cmds[#rhs_cmds+1] = {
        cmd = "read",
        rkey = w_rkey,
        ckey = w_ckey
      }

    end
  end
  return lhs_cmds, rhs_cmds
end

function _nreg(BSIZE, lexp, rexp, mode)
    if lexp == "return" then
      local cmd = {
        cmd = "read",
        rkey = VARLIST[rexp].rkey,
        ckey = VARLIST[rexp].ckey,
        ret = true
      }
    else
      lhs_cmds, rhs_cmds = syncMemory(BSIZE, lexp, rexp, mode)
      -- print(lhs_cmds)
      -- print(rhs_cmds)
      if mode == "external" then 
        return 1
      end
    end
end

function _nreg_external(BSIZE, lhs, rhs)
  return _nreg(BSIZE, lhs, rhs, "external")
end

function _nload_data(BSIZE, args)
  local vrows = args:size()[2]
  local start_col = 1; local end_col = cols
  local INDX = MEMCNTR
  local w_rkey = torch.zeros(BSIZE, rows)
  local w_ckey = torch.zeros(BSIZE, cols)
  w_rkey[{{}, {INDX, INDX + vrows - 1}}] = 1
  w_ckey[{{}, {start_col,end_col}}] = 1

  local memory = torch.zeros(BSIZE, rows, cols)
  memory[{{},{INDX, INDX+ vrows - 1}, {start_col,end_col}}] = args
  local cmd = {
    [1] = {
      cmd = "write",
      rkey = w_rkey:clone(), --row key
      ckey = w_ckey:clone(),  --col key
      val = memory
    }
  }

  local vname = "args"
  if VARLIST[vname] == nil then
    VARLIST[vname] = { rkey = w_rkey, ckey = w_ckey, INDX = MEMCNTR }
  end
  MEMCNTR = MEMCNTR + vrows
end

rows = 20
cols = 10
MBANK = torch.zeros(rows,cols)

VARLIST = {}

function syncMemory(lexp, rexp)
  -- process LHS -- 
  local vname, vind, val, vrows, vcols
  if lexp[1] == "Identifier" then
    vname = lexp[2]
    val = lexp[3]
    if type(val) == type(1) then
      vrows = 1; vcols = cols
    else
      local sz = val:size()
      print(sz)
      vrows = sz[1]; vcols = sz[2]
    end
  end
  -- return
  

  print(lexp)
  print(rexp) 
end

function _nreg(lexp, rexp)
    if lexp == "return" then
        print('TODO: return')
    else
        syncMemory(lexp, rexp)
    end
    print('----')
end
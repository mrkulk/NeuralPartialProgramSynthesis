require 'engine' 
function program(arg)
  a=torch.zeros(1,10)
_nreg({'Identifier','a',a},{{'CallExpression','zeros',zeros,1},})
  fac = 30
_nreg({'Identifier','fac',fac},{{'Literal','30',30},})
  mult = 5
_nreg({'Identifier','mult',mult},{{'Literal','5',5},})
  dummy = fac * 4
_nreg({'Identifier','dummy',dummy},{{'Identifier','fac',fac},{'Literal','4',4},})
  for indx=1,10 do
    a[{{1,1},{indx,indx}}]  = arg[{{1,1},{indx,indx}}]*fac + arg[{{1,1},{1,1}}]*mult
_nreg({'MemberExpression','a',a,1,1,indx,indx,},{{'MemberExpression','arg',arg,1,1,indx,indx,},{'Identifier','fac',fac},{'MemberExpression','arg',arg,1,1,1,1,},{'Identifier','mult',mult},})
  end
_nreg('return','a')
  return a
end

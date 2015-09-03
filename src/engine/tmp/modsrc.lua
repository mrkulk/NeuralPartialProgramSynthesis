require 'engine' 
function program(arg)
  a=torch.zeros(1,10)
_nreg({'Identifier','a',a},{{'CallExpression','zeros',zeros,1},})
  fac = 30
_nreg({'Identifier','fac',fac},{{'Literal','30',30},})
  mult = 5
_nreg({'Identifier','mult',mult},{{'Literal','5',5},})
  dummy = fac
_nreg({'Identifier','dummy',dummy},{{'Identifier','fac',fac},})
  for indx=1,10 do
    a[{1,indx}]  = arg[{1,indx}]*fac + mult*arg[{1,1}]
_nreg({'MemberExpression','a',a,1,indx,},{{'MemberExpression','arg',arg,1,indx,},{'Identifier','fac',fac},{'Identifier','mult',mult},{'MemberExpression','arg',arg,1,1,},})
  end
_nreg('return','a')
  return a
end

require 'engine' 
function program(arg)
  a=torch.zeros(10)
_nreg({'Identifier','a',a},{{'CallExpression','zeros',zeros,10},})
  fac = 30
_nreg({'Identifier','fac',fac},{{'Literal','30',30},})
  mult = 5
_nreg({'Identifier','mult',mult},{{'Literal','5',5},})
  for indx=1,10 do
    a[indx] = arg[indx]*fac + mult*arg[1]
_nreg({'MemberExpression','a',a,indx},{{'Identifier','arg',arg,indx},{'Identifier','fac',fac},{'Identifier','mult',mult},{'Literal','arg',arg,1},})
  end
_nreg('return','a')
  return a
end

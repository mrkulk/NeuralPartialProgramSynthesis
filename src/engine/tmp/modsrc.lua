require 'engine' 
function program(arg)
  a=torch.zeros(10)
_nreg('["Identifier","a"]','[["CallExpression","zeros",10]]')
  fac = 30
_nreg('["Identifier","fac"]','[["Literal",30]]')
  mult = 5
_nreg('["Identifier","mult"]','[["Literal",5]]')
  for i=1,10 do
    a[i] = arg[i]*fac + mult*arg[1]
_nreg('["MemberExpression","a","i"]','[["MemberExpression-Identifier","arg","i"],["Identifier","fac"],["Identifier","mult"],["MemberExpression-Literal","arg",1]]')
  end
_nreg('return','a')
  return a
end

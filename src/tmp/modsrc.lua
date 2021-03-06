require 'engine' 
function program(BSIZE, args)
_nload_data(BSIZE,args)
  a=torch.zeros(BSIZE,1,10)
_nreg(BSIZE, {'Identifier','a',a},{{'CallExpression','zeros',zeros},})
  fac = 0.5
_nreg(BSIZE, {'Identifier','fac',fac},{{'Literal','0.5',0.5},})
  mult = 0.2
_nreg(BSIZE, {'Identifier','mult',mult},{{'Literal','0.2',0.2},})
  dummy = fac * 4
_nreg(BSIZE, {'Identifier','dummy',dummy},{{'Identifier','fac',fac},{'Literal','4',4},})
  for indx=1,10 do
    a[{{},{1,1},{indx,indx}}]  = _nreg_external(BSIZE, {'MemberExpression','a',a,{},1,1,indx,indx,}, {{'MemberExpression','args',args,{},1,1,indx,indx,},{'Identifier','fac',fac},{'MemberExpression','args',args,{},1,1,1,1,},{'Identifier','mult',mult},})--args[{{},{1,1},{indx,indx}}]*fac + args[{{},{1,1},{1,1}}]*mult
  end
_nreg(BSIZE, 'return','a')
  return a
end

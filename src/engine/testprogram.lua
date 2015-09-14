function program(BSIZE, args)
  a=torch.zeros(BSIZE,1,10)
  fac = 30
  mult = 5
  dummy = fac * 4
  for indx=1,10 do
    a[{{},{1,1},{indx,indx}}]  = _nreg_external(BSIZE, {'MemberExpression','a',a,{},1,1,indx,indx,}, {{'MemberExpression','args',args,{},1,1,indx,indx,},{'Identifier','fac',fac},{'MemberExpression','args',args,{},1,1,1,1,},{'Identifier','mult',mult},})--args[{{},{1,1},{indx,indx}}]*fac + args[{{},{1,1},{1,1}}]*mult
  end
  return a
end
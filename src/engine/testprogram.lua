function program(arg)
  a=torch.zeros(1,10)
  fac = 30
  mult = 5
  dummy = fac * 4
  for indx=1,10 do
    a[{{1,1},{indx,indx}}]  = arg[{{1,1},{indx,indx}}]*fac + arg[{{1,1},{1,1}}]*mult
  end
  return a
end
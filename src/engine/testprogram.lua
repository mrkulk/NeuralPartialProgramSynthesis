function program(args)
  a=torch.zeros(1,10)
  fac = 30
  mult = 5
  dummy = fac * 4
  for indx=1,10 do
    a[{{1,1},{indx,indx}}]  = args[{{1,1},{indx,indx}}]*fac + args[{{1,1},{1,1}}]*mult
  end
  return a
end
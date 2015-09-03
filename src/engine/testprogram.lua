function program(arg)
  a=torch.zeros(1,10)
  fac = 30
  mult = 5
  dummy = fac
  for indx=1,10 do
    a[{1,indx}]  = arg[{1,indx}]*fac + mult*arg[{1,1}]
  end
  return a
end

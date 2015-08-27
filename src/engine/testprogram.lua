function program(arg)
  a=torch.zeros(10)
  fac = 30
  mult = 5
  for indx=1,10 do
    a[indx] = arg[indx]*fac + mult*arg[1]
  end
  return a
end

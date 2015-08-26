function program(arg)
  a=torch.zeros(10)
  fac = 30
  mult = 5
  for i=1,10 do
    a[i] = arg[i]*fac + mult*arg[1]
  end
  return a
end

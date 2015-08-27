require 'engine' 
function program(arg)
  a=torch.zeros(10)
_nreg({a},{{zeros,10},})
  fac = 30
_nreg({fac},{{30},})
  mult = 5
_nreg({mult},{{5},})
  for indx=1,10 do
    a[indx] = arg[indx]*fac + mult*arg[1]
_nreg({a,indx},{{arg,indx},{fac},{mult},{arg,1},})
  end
_nreg('return','a')
  return a
end

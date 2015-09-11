require 'srctransform.lua'

local function transform_code()
  transformer("testprogram.lua")
  print('Completed program source transformation ...\n\n')
  -- print(MODIFIED_SOURCE)
  local file = io.open("tmp/modsrc.lua", "w")
  file:write(MODIFIED_SOURCE)
  file:close()
end

function main()
	transform_code()
  	require('tmp/modsrc.lua')
  	
  	local BSIZE = 1
  	program(BSIZE, torch.rand(1,10))
end

main()
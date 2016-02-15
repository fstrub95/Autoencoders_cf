require("torch")

---------------------------------------
-- Basic Constant 


math.round = math.round or function(num, idp)
   
  if idp == nil then idp = 4 end

  local mult = 10^(idp or 0)
  return math.floor(num * mult + 0.5) / mult
end

--[[---------------------------------------------------------
	Name: Copy(t, lookup_table)
	Desc: Taken straight from http://lua-users.org/wiki/PitLibTablestuff
		and modified to the new Lua 5.1 code by me.
		Original function by PeterPrade!
-----------------------------------------------------------]]
function table.Copy( t, lookup_table )
	if ( t == nil ) then return nil end

	local copy = {}
	setmetatable( copy, debug.getmetatable( t ) )
	for i, v in pairs( t ) do
		if ( torch.type(v) ~= "table" ) then
			copy[ i ] = v
		else
			lookup_table = lookup_table or {}
			lookup_table[ t ] = copy
			if ( lookup_table[ v ] ) then
				copy[ i ] = lookup_table[ v ] -- we already copied this table. reuse the copy.
			else
				copy[ i ] = table.Copy( v, lookup_table ) -- not yet copied. copy it.
			end
		end
	end
	return copy
end


table.merge = table.merge or function(t1,  t2)
    for k, v in pairs(t2) do
        if (type(v) == "table") and (type(t1[k] or false) == "table") then
            table.merge(t1[k], t2[k])
        else
            if t1[k] == nil then
               t1[k] = v
            end
        end
    end
    return t1
end

--[[---------------------------------------------------------
   Name: table.Count( table )
   Desc: Returns the number of keys in a table
-----------------------------------------------------------]]
function table.Count( t )
   local i = 0
   for _, k in pairs( t ) do i = i + 1 end
   return i
end

--[[---------------------------------------------------------
   Name: table.Random( table )
   Desc: Return a random key
-----------------------------------------------------------]]
function table.Random( t )
   local rk = math.random( 1, table.Count( t ) )
   local i = 1
   for k, v in pairs( t ) do 
      if ( i == rk ) then return v, k end
      i = i + 1 
   end
end

string.starts = string.starts or function (String,Start)
   return string.sub(String,1,string.len(Start))==Start
end

string.ends = string.ends or function(String,End)
   return End=='' or string.sub(String,-string.len(End))==End
end

string.split = string.split or function(str, pat) 
   local t = {}  -- NOTE: use {n = 0} in Lua-5.0
   local fpat = "(.-)" .. pat
   local last_end = 1
   local s, e, cap = str:find(fpat, 1)
   while s do
      if s ~= 1 or cap ~= "" then
    table.insert(t,cap)
      end
      last_end = e+1
      s, e, cap = str:find(fpat, last_end)
   end
   if last_end <= #str then
      cap = str:sub(last_end)
      table.insert(t, cap)
   end
   return t
end


function GetSize(X,dim)
   if torch.isTensor(X) == false then 
      return #X
   else 
      if dim == nil then dim = 1 end
      return X:size(dim)
   end
end

local function GetnElement(X) 
   if torch.isTensor(X)  then 
      return X:nElement()
   elseif torch.type(X) == "table" then 
      local size = 0
      for _, _ in pairs(X) do size = size + 1 end
      return size
   else return nil
   end
end


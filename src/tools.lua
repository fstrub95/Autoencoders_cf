require("torch")

---------------------------------------
-- Basic Constant 

NaN =  1/0 -- -999


function uid()
   return (os.time() .. math.random()):gsub('%.','')
end


function round(num, idp)
   
  if idp == nil then idp = 4 end

  local mult = 10^(idp or 0)
  return math.floor(num * mult + 0.5) / mult
end


function merge(t1,  t2)
    for k, v in pairs(t2) do
        if (type(v) == "table") and (type(t1[k] or false) == "table") then
            merge(t1[k], t2[k])
        else
            if t1[k] == nil then
               t1[k] = v
            end
        end
    end
    return t1
end

function string.starts(String,Start)
   return string.sub(String,1,string.len(Start))==Start
end

function string.ends(String,End)
   return End=='' or string.sub(String,-string.len(End))==End
end

function GetSize(X,dim)
   if torch.isTensor(X) == false then 
      return #X
   else 
      if dim == nil then dim = 1 end
      return X:size(dim)
   end
end

function tensorToCsv(M, outFile)

   print("This can take several minutes")

   local ouputTrain = io.open(outFile, "w")
   io.output(ouputTrain)
   for i = 1, M:size(1) do
      xlua.progress(i, M:size(1))
      local line = ""
      for j = 1, M:size(2) do
         line = line .. M[i][j] .. " " 
      end
      io.write(line .. "\n")
   end

   io.close(ouputTrain)

end

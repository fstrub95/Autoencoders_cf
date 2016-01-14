local AppenderIn, parent = torch.class('nnsparse.AppenderIn')

function AppenderIn:prepareInput(denseInput, sparseInput)
   self.input       = denseInput
   self.sparseInput = sparseInput
end



local AppenderOut, parent = torch.class('nnsparse.AppenderOut', 'nn.Module')

function AppenderOut:__init(appenderIn)
   parent:__init()
   self.appenderIn = appenderIn
   self.output = nil
end


function AppenderOut:updateOutput(input)

   --share input!!!
   local inputToAppend = self.appenderIn.input

   self.prevSize = input:size()
   self.output = self.output:typeAs(input)

   if inputToAppend ~= nil then
      --self.output:resize(input:size(1), input:size(2) + inputToAppend:size(2))

      torch.cat(self.output, input, inputToAppend, 2)
   else
      self.output = input
   end

   return self.output
end

function AppenderOut:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput[{{}, {1, self.prevSize[2]}}] -- truncate the added input
   return  self.gradInput
end


local AppenderSparseOut, parent = torch.class('nnsparse.AppenderSparseOut', 'nn.Module')

function AppenderSparseOut:__init(appenderIn, offset)
   parent:__init()
   self.appenderIn = appenderIn
   self.offset = appenderIn
   self.output = nil
end


function AppenderSparseOut:updateOutput(input)
   assert(torch.type(input) == "table")

   self.output = self.output or {}
   
   if #input ~= #self.output then 
      self.output = {} 
   end

   --share input!!!
   local inputToAppend = self.appenderIn.sparseInput
   assert(torch.type(inputToAppend) == "table")
   
   if inputToAppend ~= nil then
      assert(#inputToAppend == #input)
      
      for k, oneInput in pairs(input) do
         self.output[k] = self.output[k] or oneInput.new()
         self.output[k]:resize(oneInput:size(1) + inputToAppend[k]:size(1) , 2)
         torch.cat(self.output[k], oneInput, inputToAppend[k], 1)   
      end
   
   else
      self.output = input
   end

   return self.output
end

function AppenderSparseOut:updateGradInput(input, gradOutput)
   return nil --return gradOutput:resize(self.prevSize) -- truncate the added input
end

local    AppenderDummy, parent = torch.class('nnsparse.AppenderDummy')
function AppenderDummy:prepareInput() end


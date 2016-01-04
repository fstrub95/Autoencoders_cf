local AppenderIn, parent = torch.class('nn.AppenderIn', 'nn.Module')

function AppenderIn:__init()
   parent.__init(self)
end

function AppenderIn:updateOutput(input)
   
   self.input = self.input or input.new()
   self.input:resizeAs(input)
   
   parent.__init(self)
   
   return input
end

function AppenderIn:updateGradInput(input, gradOutput)
   return gradOutput
end

local AppenderOut, parent = torch.class('nn.AppenderOut', 'nn.Module')

function AppenderOut:__init(appenderIn)
   parent.__init(self)
   self.appenderIn = appenderIn
end

function AppenderOut:updateOutput(input)

   --share input!!!
   local inputToAppend = self.appenderIn.input

   self.prevSize = input:size()

   if inputToAppend ~= nil then
      self.input = self.input or input.new()
      self.input:resize(input:size(1), input:size(2) + inputToAppend:size(2))

      torch.cat(self.input, input, inputToAppend, 2)
   else
      self.input = input
   end

   return self.input
end

function AppenderIn:updateGradInput(input, gradOutput)
   return gradOutput:resize(self.prevSize) -- truncate the added input
end


local AppenderSparseOut, parent = torch.class('nn.AppenderSparseOut', 'nn.Module')

function AppenderSparseOut:__init(appenderIn, offset)
   parent.__init(self)
   self.appenderIn = appenderIn
   self.offset = appenderIn
end

function AppenderOut:updateOutput(input)

   --share input!!!
   local inputToAppend = self.appenderIn.input

   self.prevSize = input:size()

   if inputToAppend ~= nil then
      self.input = self.input or input.new()
      self.input:resize(input:size(1), input:size(2) + inputToAppend:size(2))

      torch.cat(self.input, input, inputToAppend, 2)
   else
      self.input = input
   end

   return self.input
end

function AppenderIn:updateGradInput(input, gradOutput)
   return gradOutput:resize(self.prevSize) -- truncate the added input
end

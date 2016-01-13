
if __config == nil then

   networkType    = {"U"}
   useMetaInfo    = {false}
   batchSize      = {20}
   layerSize1     = {700}
   layerSize2     = {500}
   weightDecay    = {0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005}
   learningRate   = {0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005}
   learningDecay  = {0, 0.1, 0.2, 0.3}
   alpha          = {1}
   beta           = {1, 0.9, 0.8, 0.7, 0.6}
   hideratio      = {0, 0.1, 0.2, 0.3, 0.4}
   __config = {}
   
end

__config = 
{
   layer1 = 
   {      
      layerSize = __layerSize1,
      { 
         criterion = nnsparse.SDAECriterionGPU(nn.MSECriterion(),
         {
            alpha     = __alpha,
            beta      = __beta,
            hideRatio = __hideratio,
         }), 
         noEpoch           = 2, 
         miniBatchSize     = __batchSize,
         learningRate      = __learningRate,  
         learningRateDecay = __learningDecay,
         weightDecay       = __weightDecay,
      },
      
   },
   
   layer2 = 
   {
      layerSize = __layerSize2,
      { 
         criterion = nnsparse.SDAECriterionGPU(nn.MSECriterion(),
         {
            alpha = __alpha,
            beta  = __beta,
            noiseRatio = 0.2,
            noiseStd  = 0.01, 
         }),
         noEpoch = 2, 
         miniBatchSize     = __batchSize,
         learningRate      = __learningRate,  
         learningRateDecay = __learningDecay,
         weightDecay       = __weightDecay,
      },
      
      {
         criterion = nnsparse.SDAECriterionGPU(nn.MSECriterion(),
         {
            alpha     = __alpha,
            beta      = __beta,
            hideRatio = __hideratio,
         }),
         noEpoch = 2,
         miniBatchSize     = __batchSize,
         learningRate      = __learningRate,  
         learningRateDecay = __learningDecay,
         weightDecay       = __weightDecay,
         
      },
      
   },
   
   layer3 = nil
}
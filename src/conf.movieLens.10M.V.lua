dofile("SDAECriterionGPU.lua")

configV = 
{
   useMetadata = false,
   layer1 = 
   {      
      layerSize = 700,
      { 
         criterion = nnsparse.SDAECriterionGPU(nn.MSECriterion(),
         {
            alpha = 0.98453737460077,
            beta  = 0.57107167015783,
            hideRatio = 0.12414132046979,
         }), 
         noEpoch = 20, 
         miniBatchSize = 35,
         learningRate = 0.061001000545609,  
         learningRateDecay = 0.3127887416165,
         weightDecay = 0.056381709873676,
      },
      
   },
}


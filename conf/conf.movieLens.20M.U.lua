config = 
{
   useMetadata = false,
   layer1 = 
   {      
      layerSize = 584,
      { 
         criterion = cfn.SDAECriterionGPU(nn.MSECriterion(),
         {
            alpha = 0.95036246101889,
            beta  = 0.69859368623131,
            hideRatio = 0.25525745830964,
         }), 
         noEpoch = 15, 
         miniBatchSize = 35,
         learningRate = 0.029226935168521,  
         learningRateDecay = 0.30946261772575,
         weightDecay = 0.0087159276267307,
      },
      
   },
 
}

return config

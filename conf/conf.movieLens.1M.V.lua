config = 
{
   useMetadata = false,
   layer1 = 
   {      
      layerSize = 700,
      { 
         criterion = cfn.SDAECriterionGPU(nn.MSECriterion(),
         {
            alpha = 1.107842557215,
            beta  = 0.72031965295577,
            hideRatio = 0.13796888895902,
         }), 
         noEpoch = 15, 
         miniBatchSize = 35,
         learningRate = 0.090054066465632,  
         learningRateDecay = 0.17623840298814,
         weightDecay = 0.025113243590272,
      },
      
   },
   
   layer2 = 
   {
      layerSize = 500,
      { 
         criterion = cfn.SDAECriterionGPU(nn.MSECriterion(),
         {
            alpha = 1,
            beta  = 0.8,
            noiseRatio = 0.2,
            noiseStd  = 0.02, 
         }),
         noEpoch = 5, 
         miniBatchSize = 5,
         learningRate  = 1e-5,  
         learningRateDecay = 0.1,
         weightDecay = 0.2,
         momentum = 0.8
      },
      
      {
         criterion = cfn.SDAECriterionGPU(nn.MSECriterion(),
         {
            alpha = 0.98560892462868,
            beta  = 0.58072139311116,
            hideRatio = 0.12767389068742,
         }),
         noEpoch = 15,
         miniBatchSize = 25,
         learningRate  = 0.015696778637008,
         learningRateDecay = 0.25096496415014,
         weightDecay = 0.045386821936839,
         
      },
      
   },
   
   layer3 = nil

}
return config

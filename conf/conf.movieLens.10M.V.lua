config = 
{
   useMetadata = false,
   layer1 = 
   {      
      layerSize = 770,
      { 
         criterion = cfn.SDAECriterionGPU(nn.MSECriterion(),
         {
            alpha = 0.91210902705444,
            beta  = 0.54054440256139,
            hideRatio = 0.25688179776383,
         }), 
         noEpoch = 20, 
         miniBatchSize = 35,
         learningRate = 0.069180836029247,  
         learningRateDecay = 0.29623809751744,
         weightDecay = 0.052745726597668,
      },
      
   },
}

return config

config = 
{
   layer1 = 
   {      
      layerSize = 529,
      { 
         criterion = cfn.SDAECriterionGPU(nn.MSECriterion(),
         {
            alpha = 0.91331130955368,
            beta  = 0.56809865101241,
            hideRatio = 0.2611993515864,
         }), 
         noEpoch = 20, 
         miniBatchSize = 35,
         learningRate = 0.086912738764659,  
         learningRateDecay = 0.34168719355948,
         weightDecay = 0.010118126263842,
      },
      
   },
}

return config

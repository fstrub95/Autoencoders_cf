config = 
{
   useMetadata = false,
   layer1 = 
   {      
      layerSize = 700,
      { 
         criterion = cfn.SDAECriterionGPU(nn.MSECriterion(),
         {
            alpha = 1.0336864752695,
            beta  = 0.38166233734228,
            hideRatio = 0.25525745830964,
         }), 
         noEpoch = 20, 
         miniBatchSize = 35,
         learningRate = 0.061005359655246,  
         learningRateDecay = 0.54645854830742,
         weightDecay = 0.00570531450212,
      },
      
   },
   
}

return config

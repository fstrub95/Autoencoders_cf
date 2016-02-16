config = 
{
   useMetadata = false,
   layer1 = 
   {      
      layerSize = 584,
      { 
         criterion = cfn.SDAECriterionGPU(nn.MSECriterion(),
         {
            alpha = 0.98905390398577,
            beta  = 0.6047364819312,
            hideRatio = 0.24630973351843,
         }), 
         noEpoch = 15, 
         miniBatchSize = 35,
         learningRate = 0.050508657319437 ,  
         learningRateDecay = 0.35850763596698,
         weightDecay = 0.026518716275071,
      },
      
   },
 
}


return config

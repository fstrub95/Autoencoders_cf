config = 
{
   layer1 = 
   {      
      layerSize = 580,
      { 
         criterion = cfn.SDAECriterionGPU(nn.MSECriterion(),
         {
            alpha = 0.90932733460019,
            beta  = 0.82108359946869,
            hideRatio = 0.26152918060931,
         }), 
         noEpoch = 20, 
         miniBatchSize = 35,
         learningRate = 0.03360762981077,  
         learningRateDecay = 0.54861738152492,
         weightDecay = 0.006019425058427,
      },
      
   },
}

return config

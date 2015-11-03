configV = 
{
   layer1 = 
   {      
      isTied    = false,
      coefLayer = 10,
      { 
         criterion = nn.SDAESparseCriterion(nn.MSECriterion(),
         {
            alpha = 1,
            beta  = 1,
            noiseRatio = 0,
            flipRatio = 0.05,
            flipRange = torch.Tensor{-1, 1},
            hideRatio = 0.15,
         }), 
         noEpoch = 15, 
         miniBatchSize = 20,
         learningRate = 0.03,  
         learningRateDecay = 0.1,
         weightDecay = 0.03,
         momentum = 0.8,
      },
      
   },
   
   layer2 = 
   {
      isTied    = false,
      plot      = false,
      coefLayer = 12,
      { 
         criterion = nn.SDAECriterion(nn.MSECriterion(),
         {
            alpha = 1,
            beta  = 1,
            noiseRatio = 0.8,
            noiseStd  = 0.05, 
            flipRatio = 0,
            flipRange = torch.Tensor{-1, 1},
            hideRatio = 0,
         }),
         noEpoch = 40, 
         miniBatchSize = 2,
         learningRate  = 1e-5,  
         learningRateDecay = 0.1,
         weightDecay = 0.2,
         momentum = 0.8
      },
      
      {
         criterion = nn.SDAESparseCriterion(nn.MSECriterion(),
         {
            alpha = 1.2,
            beta  = 0.8,
            noiseRatio = 0,
            flipRatio = 0.05,
            flipRange = torch.Tensor{-1, 1},
            hideRatio = 0.15,
         }),
         noEpoch = 15,
         miniBatchSize = 20,
         learningRate  = 0.003,
         learningRateDecay = 0.2,
         weightDecay = 0.03,
         momentum = 0.8,
         
      },
      
   },
   
   layer3 = nil

}

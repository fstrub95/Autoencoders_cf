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
            alpha = 1.1074643210508,
            beta  = 0.82831857485386,
            hideRatio = 0.22428934609828,
         }), 
         noEpoch = 20, 
         miniBatchSize = 35,
         learningRate = 0.039843164927637,  
         learningRateDecay = 0.22584167215973,
         weightDecay = 0.080668330929863,
      },
      
   },
}



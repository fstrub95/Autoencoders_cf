# Collaborative Filtering with Stacked Denoising Encoders and sparse Inputs

Collaborative filering consists in predicting the rating of items by a user by using the feedback of all other users. In other words, it try to turn a spare matrix of ratings into a dense matrix of rating. 

The following module tackle the issue by using sparse denoising autoencoders.

You may download the package and run the following command:

```
th main.lua
```

The default behavior will try to predict the rating of the dataset movieLens-1M. (90% training, 10% rating)  

The following option are also available:
```
-file         The relative path to your data file.              Default = ../data/movieLens/ratings-1M.dat
-conf         The relative path to the lua configuration file.  Default = config.template.lua
-ratio        The training ratio                                Default = 0.9                                
-fileType     The data file format (jester/movieLens/classic)   Default = movieLens                       
-seed         The random seed                                   Default = 1234
-out          The path to store the final matrix (csv)          Default = ..
```

One may also change the learninf process or the network architecture by using the file config.template.lua
```lua
configV =          -- ConfigV --> learn Vencoders / configU --> learn Uencoder
{
   layer1 = 
   {      
      isTied    = false, -- tie the autoencoders weight
      coefLayer = 10,    -- reduce the size of hidden layer of the autoencoder by diving the input size by X
      { 
         criterion = nn.SDAESparseCriterion(nn.MSECriterion(), -- define the training loss
         {
            alpha = 1,                       -- prediction hyperparameter 
            beta  = 1,                       -- reconstruction hyperparameter
            noiseRatio = 0,                  -- Gaussian noise ratio
            flipRatio = 0.05,                -- SaltAndPepper Ratio
            flipRange = torch.Tensor{-1, 1}, -- SaltAndPaperRange
            hideRatio = 0.15,                -- Maksing noise ratio
         }), 
         noEpoch = 15,                       -- number of epoch to train the layer
         miniBatchSize = 20,                 -- minibatch size 
         learningRate = 0.03,                -- Learning rate
         learningRateDecay = 0.1,            -- Learning rate decay lrt = lrt / (1+lrt_dec)
         weightDecay = 0.03,                 -- L2 regulizer
         momentum = 0.8,                     -- momentum
      },
      
   },
```

When several layers are stacked, one need to define the learning process of every layer
```lua
configV = 
{
   layer1 = 
   {
      isTied    = false, 
      coefLayer = 10,    
    { Training 1 }
   },
   layer2 =
   {
     isTied    = false, 
     coefLayer = 12,    
     { Training 1 },  --inner layer
     { Training 2 },  --final network
    },
    layer3 =
   { 
     isTied    = false, 
     coefLayer = 14,    
    { Training 1 }, -- inner hidden layer
    { Training 2 }, -- intermediate hidden layer
    { Training 3 }, -- final network
    }
    etc.
}
```

The SVD and ALS-WR algorithms are also provided for benchmarking
